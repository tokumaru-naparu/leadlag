"""
08_signal_today.py（再設計版）

改善点:
  1. 祝日・非営業日は自動スキップ
  2. データを5つのCSVに分離（AI分析用）
  3. カラム名を英語・スネークケースに統一

保存するCSV:
  data/signals.csv     → 毎日のシグナル値（17業種）
  data/returns.csv     → 毎日の実績リターン（17業種）
  data/trades.csv      → 売買記録
  data/performance.csv → 損益・資産推移
  data/market.csv      → 市場環境（米国リターン）
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

try:
    import jpholiday
    HAS_JPHOLIDAY = True
except ImportError:
    HAS_JPHOLIDAY = False
    print("⚠️  jpholiday 未インストール → pip install jpholiday")

# ============================================================
# 設定
# ============================================================

DATA_DIR        = Path(__file__).parent / "data"
INITIAL_CAPITAL = 1_000_000

L      = 60
K      = 3
LAMBDA = 0.9
TOP_N  = 3

US_TICKERS = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
JP_TICKERS = ["1617.T","1618.T","1619.T","1620.T","1621.T","1622.T","1623.T",
              "1624.T","1625.T","1626.T","1627.T","1628.T","1629.T","1630.T",
              "1631.T","1632.T","1633.T"]

JP_KEYS = {
    "1617.T": "food",        "1618.T": "energy",
    "1619.T": "construction","1620.T": "materials",
    "1621.T": "pharma",      "1622.T": "auto",
    "1623.T": "steel",       "1624.T": "machinery",
    "1625.T": "electronics", "1626.T": "it_services",
    "1627.T": "utilities",   "1628.T": "transport",
    "1629.T": "trading",     "1630.T": "retail",
    "1631.T": "banks",       "1632.T": "finance",
    "1633.T": "realestate",
}

JP_NAMES = {
    "1617.T": "食品",        "1618.T": "エネルギー",
    "1619.T": "建設・資材",  "1620.T": "素材・化学",
    "1621.T": "医薬品",      "1622.T": "自動車",
    "1623.T": "鉄鋼・非鉄",  "1624.T": "機械",
    "1625.T": "電機・精密",  "1626.T": "情報通信",
    "1627.T": "電力・ガス",  "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",  "1630.T": "小売",
    "1631.T": "銀行",        "1632.T": "金融",
    "1633.T": "不動産",
}

US_KEYS = {
    "XLB": "materials",    "XLC": "communication",
    "XLE": "energy",       "XLF": "financials",
    "XLI": "industrials",  "XLK": "tech",
    "XLP": "staples",      "XLRE": "realestate",
    "XLU": "utilities",    "XLV": "healthcare",
    "XLY": "discretionary",
}

US_CYCLICAL  = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE = ["XLK", "XLP", "XLU", "XLV"]
JP_CYCLICAL  = ["1618.T", "1625.T", "1629.T", "1631.T"]
JP_DEFENSIVE = ["1617.T", "1621.T", "1627.T", "1630.T"]

# ============================================================
# 営業日チェック
# ============================================================

def is_jp_business_day(date: datetime) -> bool:
    if date.weekday() >= 5:
        return False
    if HAS_JPHOLIDAY and jpholiday.is_holiday(date.date()):
        return False
    return True

# ============================================================
# PCA SUB 関数群
# ============================================================

def standardize(df):
    mu    = df.mean()
    sigma = df.std().replace(0, np.nan)
    return (df - mu) / sigma, mu, sigma

def build_prior_subspace(available_tickers):
    n      = len(available_tickers)
    is_us  = np.array([t in US_TICKERS for t in available_tickers], dtype=float)
    is_jp  = np.array([t in JP_TICKERS for t in available_tickers], dtype=float)
    v1     = np.ones(n) / np.sqrt(n)
    v2_raw = is_us - is_jp
    v2_raw -= np.dot(v2_raw, v1) * v1
    norm2  = np.linalg.norm(v2_raw)
    v2     = v2_raw / norm2 if norm2 > 1e-10 else np.zeros(n)
    v3_raw = np.zeros(n)
    for i, t in enumerate(available_tickers):
        if t in US_CYCLICAL or t in JP_CYCLICAL:
            v3_raw[i] = +1.0
        elif t in US_DEFENSIVE or t in JP_DEFENSIVE:
            v3_raw[i] = -1.0
    v3_raw -= np.dot(v3_raw, v1) * v1
    v3_raw -= np.dot(v3_raw, v2) * v2
    norm3  = np.linalg.norm(v3_raw)
    v3     = v3_raw / norm3 if norm3 > 1e-10 else np.zeros(n)
    return np.column_stack([v1, v2, v3])

def build_C0(V0, cfull):
    D0     = np.diag(np.diag(V0.T @ cfull @ V0))
    C0_raw = V0 @ D0 @ V0.T
    diag_v = np.where(np.diag(C0_raw) > 0, np.diag(C0_raw), 1.0)
    C0     = np.diag(1.0 / np.sqrt(diag_v)) @ C0_raw @ np.diag(1.0 / np.sqrt(diag_v))
    np.fill_diagonal(C0, 1.0)
    return C0

def calc_signal(window_us, window_jp, z_us_today, cfull_matrix, cfull_tickers):
    combined  = pd.concat([window_us, window_jp], axis=1)
    combined  = combined.dropna(axis=1, how="all").fillna(0)
    if len(combined.columns) < K + 1:
        return pd.Series(np.nan, index=JP_TICKERS)
    available = combined.columns.tolist()
    us_avail  = [t for t in available if t in US_TICKERS]
    jp_avail  = [t for t in available if t in JP_TICKERS]
    if not us_avail or not jp_avail:
        return pd.Series(np.nan, index=JP_TICKERS)
    z_comb, _, _ = standardize(combined)
    Ct = z_comb.values.T @ z_comb.values / len(z_comb)
    V0 = build_prior_subspace(available)
    cidx = [cfull_tickers.index(t) for t in available if t in cfull_tickers]
    C0   = build_C0(V0, cfull_matrix[np.ix_(cidx, cidx)]) if len(cidx) == Ct.shape[0] else np.eye(len(available))
    C_reg = (1 - LAMBDA) * Ct + LAMBDA * C0
    evals, evecs = np.linalg.eigh(C_reg)
    V    = evecs[:, np.argsort(evals)[::-1][:K]]
    ui   = [available.index(t) for t in us_avail]
    ji   = [available.index(t) for t in jp_avail]
    _, mu_us, sg = standardize(window_us.reindex(columns=us_avail).fillna(0))
    zt   = ((z_us_today.reindex(us_avail) - mu_us) / sg.replace(0, np.nan)).fillna(0)
    ft   = V[ui, :].T @ zt.values
    sig  = pd.Series(np.nan, index=JP_TICKERS)
    sig.loc[jp_avail] = V[ji, :] @ ft
    return sig

# ============================================================
# CSV ヘルパー
# ============================================================

def load_csv(filename):
    path = DATA_DIR / filename
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return pd.DataFrame()

def save_row(df_existing, row_dict, filename, index_col):
    new_df = pd.DataFrame([row_dict])
    new_df[index_col] = pd.to_datetime(new_df[index_col])
    new_df = new_df.set_index(index_col)
    if len(df_existing) > 0:
        if new_df.index[0] in df_existing.index:
            df_existing = df_existing.drop(new_df.index[0])
        result = pd.concat([df_existing, new_df]).sort_index()
    else:
        result = new_df
    result.to_csv(DATA_DIR / filename, encoding="utf-8-sig")
    return result

# ============================================================
# メイン
# ============================================================

def main():
    today     = datetime.today()
    today_str = today.strftime("%Y-%m-%d")
    wday_ja   = ["月","火","水","木","金","土","日"][today.weekday()]
    wday_en   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][today.weekday()]

    print(f"\n{'='*55}")
    print(f"  PCA SUB シグナル  {today_str}（{wday_ja}曜日）")
    print(f"{'='*55}")

    # 営業日チェック
    if not is_jp_business_day(today):
        reason = "土日" if today.weekday() >= 5 else "祝日"
        print(f"\n⏭️  {reason}のためスキップします\n")
        return

    # データ取得
    print("\n📡 データ取得中...")
    start = (today - timedelta(days=120)).strftime("%Y-%m-%d")
    us_raw = yf.download(US_TICKERS, start=start, auto_adjust=True, progress=False)
    jp_raw = yf.download(JP_TICKERS, start=start, auto_adjust=True, progress=False)

    us_cc = us_raw["Close"].reindex(columns=US_TICKERS).pct_change()
    jp_cc = jp_raw["Close"].reindex(columns=JP_TICKERS).pct_change()
    jp_oc = ((jp_raw["Close"] - jp_raw["Open"]) / jp_raw["Open"]).reindex(columns=JP_TICKERS)

    common = us_cc.index.intersection(jp_cc.index)
    us_cc  = us_cc.loc[common]
    jp_cc  = jp_cc.loc[common]
    jp_oc  = jp_oc.loc[common]

    today_pd = pd.Timestamp(today_str)
    if today_pd not in common:
        print(f"⚠️  今日のデータ未取得。18:00以降に再実行してください")
        return
    print(f"  取得完了: {len(common)} 営業日分")

    # Cfull
    cdf = pd.concat([us_cc, jp_cc], axis=1).dropna(axis=1, how="all").fillna(0)
    z_c, _, _ = standardize(cdf)
    cfull_mat = z_c.values.T @ z_c.values / len(z_c)
    cfull_tickers = cdf.columns.tolist()

    # シグナル計算
    idx       = common.get_loc(today_pd)
    signal    = calc_signal(
        us_cc.iloc[idx-L:idx], jp_cc.iloc[idx-L:idx],
        us_cc.loc[today_pd], cfull_mat, cfull_tickers
    )
    ranked    = signal.dropna().sort_values(ascending=False)
    long_t    = ranked.head(TOP_N).index.tolist()
    short_t   = ranked.tail(TOP_N).index.tolist()
    sig_strength = float(ranked.iloc[0]  - ranked.iloc[-1])
    sig_spread   = float(ranked.iloc[:TOP_N].mean() - ranked.iloc[-TOP_N:].mean())

    # 実績
    oc        = jp_oc.loc[today_pd]
    long_ret  = float(oc[long_t].mean())
    short_ret = float(oc[short_t].mean())
    strat_ret = long_ret - short_ret
    is_correct = int(long_ret > short_ret)

    # 累積
    perf_df   = load_csv("performance.csv")
    prev_cap  = float(perf_df["capital"].iloc[-1])        if len(perf_df) > 0 else INITIAL_CAPITAL
    prev_capl = float(perf_df["capital_long_only"].iloc[-1]) if len(perf_df) > 0 else INITIAL_CAPITAL
    wins      = int(perf_df["is_correct"].sum())           if len(perf_df) > 0 else 0
    days      = len(perf_df)
    r20       = int(perf_df["is_correct"].tail(20).sum())  if len(perf_df) > 0 else 0

    new_cap   = prev_cap  * (1 + strat_ret)
    new_capl  = prev_capl * (1 + long_ret)
    days     += 1
    wins     += is_correct
    r20      += is_correct

    # ── CSV1: signals.csv ────────────────────────────────
    sig_row = {"date": today_str,
               "signal_strength": round(sig_strength, 6),
               "signal_spread":   round(sig_spread,   6)}
    for t in JP_TICKERS:
        sig_row[f"signal_{JP_KEYS[t]}"]  = round(float(signal.get(t, np.nan)), 6)
    for rank, t in enumerate(ranked.index, 1):
        sig_row[f"rank_{JP_KEYS[t]}"] = rank
    save_row(load_csv("signals.csv"), sig_row, "signals.csv", "date")

    # ── CSV2: returns.csv ────────────────────────────────
    ret_row = {"date": today_str}
    for t in JP_TICKERS:
        ret_row[f"oc_return_{JP_KEYS[t]}"] = round(float(oc.get(t, np.nan)), 6)
    save_row(load_csv("returns.csv"), ret_row, "returns.csv", "date")

    # ── CSV3: trades.csv ─────────────────────────────────
    trade_row = {
        "date": today_str, "weekday": wday_en, "month": today.month,
        "long_1":  JP_KEYS.get(long_t[0],  "") if len(long_t)  > 0 else "",
        "long_2":  JP_KEYS.get(long_t[1],  "") if len(long_t)  > 1 else "",
        "long_3":  JP_KEYS.get(long_t[2],  "") if len(long_t)  > 2 else "",
        "short_1": JP_KEYS.get(short_t[0], "") if len(short_t) > 0 else "",
        "short_2": JP_KEYS.get(short_t[1], "") if len(short_t) > 1 else "",
        "short_3": JP_KEYS.get(short_t[2], "") if len(short_t) > 2 else "",
        "signal_strength":  round(sig_strength, 6),
        "signal_spread":    round(sig_spread,   6),
        "long_return":      round(long_ret,      6),
        "short_return":     round(short_ret,     6),
        "strategy_return":  round(strat_ret,     6),
        "long_only_return": round(long_ret,      6),
        "is_correct":       is_correct,
    }
    save_row(load_csv("trades.csv"), trade_row, "trades.csv", "date")

    # ── CSV4: performance.csv ────────────────────────────
    perf_row = {
        "date":              today_str,
        "strategy_return":   round(strat_ret,  6),
        "long_return":       round(long_ret,   6),
        "short_return":      round(short_ret,  6),
        "long_only_return":  round(long_ret,   6),
        "capital":           round(new_cap,    0),
        "capital_long_only": round(new_capl,   0),
        "pnl":               round(new_cap - INITIAL_CAPITAL, 0),
        "pnl_pct":           round((new_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL, 6),
        "is_correct":        is_correct,
        "win_rate_total":    round(wins / days, 4),
        "win_rate_20d":      round(r20 / min(days, 20), 4),
        "total_days":        days,
    }
    save_row(perf_df, perf_row, "performance.csv", "date")

    # ── CSV5: market.csv ─────────────────────────────────
    mkt_row = {"date": today_str}
    for t in US_TICKERS:
        mkt_row[f"us_cc_{US_KEYS[t]}"] = round(float(us_cc.loc[today_pd].get(t, np.nan)), 6)
    save_row(load_csv("market.csv"), mkt_row, "market.csv", "date")

    # ── 表示 ─────────────────────────────────────────────
    print(f"\n📊 シグナル（強度: {sig_strength:.4f}）")
    print(f"\n  🟢 ロング:")
    for t in long_t:
        print(f"     {JP_NAMES[t]:10s}  signal:{signal.get(t,np.nan):+.4f}  実績:{oc.get(t,np.nan)*100:+.2f}%")
    print(f"\n  🔴 ショート:")
    for t in short_t:
        print(f"     {JP_NAMES[t]:10s}  signal:{signal.get(t,np.nan):+.4f}  実績:{oc.get(t,np.nan)*100:+.2f}%")

    print(f"\n📈 損益:  ロング {long_ret*100:+.3f}%  ショート {short_ret*100:+.3f}%  "
          f"戦略 {strat_ret*100:+.3f}%  {'✅' if is_correct else '❌'}")

    print(f"\n💰 資産: {new_cap:>12,.0f}円  "
          f"({(new_cap-INITIAL_CAPITAL)/INITIAL_CAPITAL*100:+.2f}%)")
    print(f"   ロングのみ: {new_capl:>12,.0f}円")

    print(f"\n🏆 {days}日 {wins}勝  勝率:{wins/days*100:.1f}%  "
          f"直近20日:{r20/min(days,20)*100:.1f}%")

    print(f"\n💾 signals / returns / trades / performance / market .csv 保存完了")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()