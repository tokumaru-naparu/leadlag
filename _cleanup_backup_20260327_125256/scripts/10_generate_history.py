"""
10_generate_history.py
過去9ヶ月分の日次詳細データを生成してCSVに保存

08_signal_today.py と同じ構造のCSVを過去分まとめて作る
→ AI分析用の学習データとして使う

保存するCSV（data/history/）:
  signals.csv     → 毎日のシグナル値（17業種）
  returns.csv     → 毎日の実績リターン（17業種）
  trades.csv      → 売買記録
  market.csv      → 市場環境（米国リターン）
  performance.csv → 損益・資産推移
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

DATA_DIR    = Path(__file__).parent / "data"
HISTORY_DIR = DATA_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)

# ============================================================
# パラメータ
# ============================================================

L               = 60
K               = 3
LAMBDA          = 0.9
TOP_N           = 3
INITIAL_CAPITAL = 1_000_000
MONTHS = 120  # 10年

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

WEEKDAY_EN = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

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
    C0     = np.diag(1/np.sqrt(diag_v)) @ C0_raw @ np.diag(1/np.sqrt(diag_v))
    np.fill_diagonal(C0, 1.0)
    return C0

def calc_signal(window_us, window_jp, z_us_today, cfull_mat, cfull_tickers):
    combined = pd.concat([window_us, window_jp], axis=1)
    combined = combined.dropna(axis=1, how="all").fillna(0)
    if len(combined.columns) < K + 1:
        return pd.Series(np.nan, index=JP_TICKERS)
    available = combined.columns.tolist()
    us_avail  = [t for t in available if t in US_TICKERS]
    jp_avail  = [t for t in available if t in JP_TICKERS]
    if not us_avail or not jp_avail:
        return pd.Series(np.nan, index=JP_TICKERS)
    z_comb, _, _ = standardize(combined)
    Ct   = z_comb.values.T @ z_comb.values / len(z_comb)
    V0   = build_prior_subspace(available)
    cidx = [cfull_tickers.index(t) for t in available if t in cfull_tickers]
    C0   = build_C0(V0, cfull_mat[np.ix_(cidx, cidx)]) if len(cidx)==Ct.shape[0] else np.eye(len(available))
    C_reg = (1-LAMBDA)*Ct + LAMBDA*C0
    evals, evecs = np.linalg.eigh(C_reg)
    V    = evecs[:, np.argsort(evals)[::-1][:K]]
    ui   = [available.index(t) for t in us_avail]
    ji   = [available.index(t) for t in jp_avail]
    _, mu_us, sg = standardize(window_us.reindex(columns=us_avail).fillna(0))
    zt   = ((z_us_today.reindex(us_avail) - mu_us) / sg.replace(0, np.nan)).fillna(0)
    ft   = V[ui,:].T @ zt.values
    sig  = pd.Series(np.nan, index=JP_TICKERS)
    sig.loc[jp_avail] = V[ji,:] @ ft
    return sig

# ============================================================
# 1. データ取得
# ============================================================

print("=== 過去9ヶ月分 日次詳細データ生成 ===")
print("データ取得中...")

today       = datetime.today()
fetch_start = (today - timedelta(days=MONTHS*30 + 200)).strftime("%Y-%m-%d")

us_raw = yf.download(US_TICKERS, start=fetch_start, auto_adjust=True, progress=False)
jp_raw = yf.download(JP_TICKERS, start=fetch_start, auto_adjust=True, progress=False)

us_cc = us_raw["Close"].reindex(columns=US_TICKERS).pct_change()
jp_cc = jp_raw["Close"].reindex(columns=JP_TICKERS).pct_change()
jp_oc = ((jp_raw["Close"] - jp_raw["Open"]) / jp_raw["Open"]).reindex(columns=JP_TICKERS)

common = us_cc.index.intersection(jp_cc.index)
us_cc  = us_cc.loc[common]
jp_cc  = jp_cc.loc[common]
jp_oc  = jp_oc.loc[common]

print(f"取得完了: {len(common)} 営業日  "
      f"({common[0].date()} 〜 {common[-1].date()})")

# Cfull
cdf = pd.concat([us_cc, jp_cc], axis=1).dropna(axis=1, how="all").fillna(0)
z_c, _, _ = standardize(cdf)
cfull_mat     = z_c.values.T @ z_c.values / len(z_c)
cfull_tickers = cdf.columns.tolist()

# ============================================================
# 2. 分析対象期間を直近9ヶ月に絞る
# ============================================================

cutoff = today - timedelta(days=MONTHS*30)
cutoff_pd = pd.Timestamp(cutoff.strftime("%Y-%m-%d"))

# L日のウィンドウが確保できる範囲で直近9ヶ月
target_dates = common[common >= cutoff_pd]
print(f"分析対象: {target_dates[0].date()} 〜 {target_dates[-1].date()}  "
      f"({len(target_dates)} 営業日)")

# ============================================================
# 3. 日次ループ
# ============================================================

print("\n日次データ生成中...")

sig_rows   = []
ret_rows   = []
trade_rows = []
mkt_rows   = []
perf_rows  = []

capital      = INITIAL_CAPITAL
capital_long = INITIAL_CAPITAL
wins         = 0
total_days   = 0

for target_date in target_dates:
    idx = common.get_loc(target_date)

    # 翌営業日（リターン評価用）
    if idx + 1 >= len(common):
        continue
    next_date = common[idx + 1]

    # ウィンドウ確保チェック
    if idx < L:
        continue

    # シグナル計算
    signal = calc_signal(
        us_cc.iloc[idx-L:idx],
        jp_cc.iloc[idx-L:idx],
        us_cc.iloc[idx],
        cfull_mat, cfull_tickers
    )

    ranked      = signal.dropna().sort_values(ascending=False)
    long_t      = ranked.head(TOP_N).index.tolist()
    short_t     = ranked.tail(TOP_N).index.tolist()
    sig_strength = float(ranked.iloc[0]  - ranked.iloc[-1]) if len(ranked) > 0 else np.nan
    sig_spread   = float(ranked.iloc[:TOP_N].mean() - ranked.iloc[-TOP_N:].mean()) if len(ranked) >= TOP_N*2 else np.nan

    # 翌日のリターン
    oc        = jp_oc.loc[next_date]
    long_ret  = float(oc[long_t].mean())  if long_t  else np.nan
    short_ret = float(oc[short_t].mean()) if short_t else np.nan
    strat_ret = long_ret - short_ret
    is_correct = int(long_ret > short_ret)

    wday_en = WEEKDAY_EN[next_date.dayofweek]
    month   = next_date.month
    date_str = next_date.strftime("%Y-%m-%d")

    # 累積
    capital      *= (1 + strat_ret)
    capital_long *= (1 + long_ret)
    total_days   += 1
    wins         += is_correct

    # ── signals ──────────────────────────────────────────
    sig_row = {"date": date_str,
               "signal_strength": round(sig_strength, 6),
               "signal_spread":   round(sig_spread,   6)}
    for t in JP_TICKERS:
        sig_row[f"signal_{JP_KEYS[t]}"] = round(float(signal.get(t, np.nan)), 6)
    for rank, t in enumerate(ranked.index, 1):
        sig_row[f"rank_{JP_KEYS[t]}"] = rank
    sig_rows.append(sig_row)

    # ── returns ──────────────────────────────────────────
    ret_row = {"date": date_str}
    for t in JP_TICKERS:
        ret_row[f"oc_return_{JP_KEYS[t]}"] = round(float(oc.get(t, np.nan)), 6)
    ret_rows.append(ret_row)

    # ── trades ───────────────────────────────────────────
    trade_rows.append({
        "date":             date_str,
        "weekday":          wday_en,
        "month":            month,
        "long_1":           JP_KEYS.get(long_t[0],  "") if len(long_t)  > 0 else "",
        "long_2":           JP_KEYS.get(long_t[1],  "") if len(long_t)  > 1 else "",
        "long_3":           JP_KEYS.get(long_t[2],  "") if len(long_t)  > 2 else "",
        "short_1":          JP_KEYS.get(short_t[0], "") if len(short_t) > 0 else "",
        "short_2":          JP_KEYS.get(short_t[1], "") if len(short_t) > 1 else "",
        "short_3":          JP_KEYS.get(short_t[2], "") if len(short_t) > 2 else "",
        "signal_strength":  round(sig_strength, 6),
        "signal_spread":    round(sig_spread,   6),
        "long_return":      round(long_ret,     6),
        "short_return":     round(short_ret,    6),
        "strategy_return":  round(strat_ret,    6),
        "long_only_return": round(long_ret,     6),
        "is_correct":       is_correct,
    })

    # ── market ───────────────────────────────────────────
    mkt_row = {"date": date_str}
    us_today = us_cc.iloc[idx]
    for t in US_TICKERS:
        mkt_row[f"us_cc_{US_KEYS[t]}"] = round(float(us_today.get(t, np.nan)), 6)
    mkt_rows.append(mkt_row)

    # ── performance ──────────────────────────────────────
    recent_20 = [r["is_correct"] for r in trade_rows[-20:]]
    perf_rows.append({
        "date":              date_str,
        "strategy_return":   round(strat_ret,  6),
        "long_return":       round(long_ret,   6),
        "short_return":      round(short_ret,  6),
        "long_only_return":  round(long_ret,   6),
        "capital":           round(capital,    0),
        "capital_long_only": round(capital_long, 0),
        "pnl":               round(capital - INITIAL_CAPITAL, 0),
        "pnl_pct":           round((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL, 6),
        "is_correct":        is_correct,
        "win_rate_total":    round(wins / total_days, 4),
        "win_rate_20d":      round(sum(recent_20) / len(recent_20), 4) if recent_20 else 0,
        "total_days":        total_days,
    })

    if total_days % 20 == 0:
        print(f"  {total_days}日完了... 現在資産: {capital:,.0f}円")

# ============================================================
# 4. CSV保存
# ============================================================

def save(rows, filename):
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)
    df.to_csv(HISTORY_DIR / filename, encoding="utf-8-sig")
    print(f"  [SAVE] history/{filename}  ({len(df)}行)")

print("\nCSV保存中...")
save(sig_rows,   "signals.csv")
save(ret_rows,   "returns.csv")
save(trade_rows, "trades.csv")
save(mkt_rows,   "market.csv")
save(perf_rows,  "performance.csv")

# ============================================================
# 5. サマリー表示
# ============================================================

final_capital = perf_rows[-1]["capital"]
total_pnl_pct = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
win_rate      = wins / total_days * 100

print(f"\n=== 9ヶ月サマリー ===")
print(f"期間     : {sig_rows[0]['date']} 〜 {sig_rows[-1]['date']}")
print(f"営業日数 : {total_days} 日")
print(f"初期資産 : {INITIAL_CAPITAL:>12,} 円")
print(f"最終資産 : {final_capital:>12,.0f} 円")
print(f"総損益   : {final_capital-INITIAL_CAPITAL:>+12,.0f} 円  ({total_pnl_pct:+.2f}%)")
print(f"勝率     : {wins}/{total_days}日  ({win_rate:.1f}%)")
print(f"\n保存先: {HISTORY_DIR}")
print("\n次のステップ: 11_analyze_patterns.py でAI分析を実行します")