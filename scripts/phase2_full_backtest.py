"""
phase2_full_backtest.py
Phase 1 で選定した個別株を使って、全期間（2010〜現在）でバックテストを実行

INPUT:
  - output/reports/phase1_ic_ranking.csv（Phase 1 の結果）
  - yfinance で全履歴データを取得

OUTPUT:
  - data/processed/signals_v2.csv (個別株版)
  - data/processed/trades_v2.csv
  - data/processed/performance_v2.csv
  - output/reports/phase2_summary.txt (成績サマリー)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402

# ================================================================
# 定数
# ================================================================

L = 60
K = 3
K0 = 3
LAMBDA = 0.9
TOP_N = 3
INITIAL_CAPITAL = 1_000_000

US_TICKERS = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]

US_CYCLICAL  = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE = ["XLK", "XLP", "XLU", "XLV"]

CFULL_START = "2010-01-01"
CFULL_END   = "2014-12-31"

WEEKDAY_EN = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

N_US = len(US_TICKERS)
N_JP = 17
N = N_US + N_JP

# JP セクターキー（順序固定）
JP_SECTOR_KEYS = [
    "food", "energy", "construction", "materials", "pharma", "auto", "steel",
    "machinery", "electronics", "it_services", "utilities", "transport",
    "trading", "retail", "banks", "finance", "realestate"
]

JP_SECTOR_NAMES = {
    "food": "食品", "energy": "エネルギー", "construction": "建設", "materials": "素材化学",
    "pharma": "医薬品", "auto": "自動車", "steel": "鉄鋼非鉄", "machinery": "機械",
    "electronics": "電機精密", "it_services": "情報通信", "utilities": "電力ガス",
    "transport": "運輸物流", "trading": "商社卸売", "retail": "小売",
    "banks": "銀行", "finance": "金融", "realestate": "不動産",
}


# ================================================================
# Phase 1 結果を読み込み → JP ティッカー辞書を作成
# ================================================================

def load_phase1_results() -> dict[str, int]:
    """Phase 1 の結果から sector_key → ticker を作成。
    各セクターで、A/B 評価の最高 IC を持つ銘柄を選択。
    A/B がなければ C を選択。
    """
    csv_path = REPORTS_DIR / "phase1_ic_ranking.csv"
    df = pd.read_csv(csv_path)
    result = {}

    for sector_key in JP_SECTOR_KEYS:
        sector_df = df[df["sector"] == sector_key].copy()

        # A/B 評価を優先
        ab_df = sector_df[sector_df["decision"].isin(["A", "B"])]
        if len(ab_df) > 0:
            best_row = ab_df.loc[ab_df["IC_full"].idxmax()]
        else:
            # A/B がなければ C を使う
            c_df = sector_df[sector_df["decision"] == "C"]
            if len(c_df) > 0:
                best_row = c_df.loc[c_df["IC_full"].idxmax()]
            else:
                # C もなければスキップ（警告は出さない）
                continue

        result[sector_key] = int(best_row["ticker"])

    return result


# ================================================================
# PCA SUB 関数群
# ================================================================

def standardize(df: pd.DataFrame):
    mu    = df.mean()
    sigma = df.std().replace(0, np.nan)
    z     = (df - mu) / sigma
    return z, mu, sigma


def build_prior_subspace() -> np.ndarray:
    """固定V0（28×3）を作る。"""
    v1 = np.ones(N) / np.sqrt(N)

    v2_raw = np.zeros(N)
    v2_raw[:N_US] = +1.0
    v2_raw[N_US:] = -1.0
    v2_raw -= np.dot(v2_raw, v1) * v1
    v2 = v2_raw / np.linalg.norm(v2_raw)

    all_tickers_us = US_TICKERS
    all_tickers_jp = JP_SECTOR_KEYS

    v3_raw = np.zeros(N)
    for i, t in enumerate(all_tickers_us):
        if t in US_CYCLICAL:
            v3_raw[i] = +1.0
        elif t in US_DEFENSIVE:
            v3_raw[i] = -1.0

    jp_cyclical  = ["energy", "auto", "trading", "banks"]
    jp_defensive = ["food", "pharma", "utilities", "retail"]

    for i, t in enumerate(all_tickers_jp):
        if t in jp_cyclical:
            v3_raw[N_US + i] = +1.0
        elif t in jp_defensive:
            v3_raw[N_US + i] = -1.0

    v3_raw -= np.dot(v3_raw, v1) * v1
    v3_raw -= np.dot(v3_raw, v2) * v2
    norm3 = np.linalg.norm(v3_raw)
    v3 = v3_raw / norm3 if norm3 > 1e-10 else np.zeros(N)

    return np.column_stack([v1, v2, v3])


def build_C0(V0: np.ndarray, cfull: np.ndarray) -> np.ndarray:
    D0     = np.diag(np.diag(V0.T @ cfull @ V0))
    C0_raw = V0 @ D0 @ V0.T
    diag_v = np.where(np.diag(C0_raw) > 0, np.diag(C0_raw), 1.0)
    C0     = np.diag(1.0 / np.sqrt(diag_v)) @ C0_raw @ np.diag(1.0 / np.sqrt(diag_v))
    np.fill_diagonal(C0, 1.0)
    return C0


def calc_pca_sub_signal(
    window_us: pd.DataFrame,
    window_jp: pd.DataFrame,
    z_us_today: pd.Series,
    C0: np.ndarray,
    V0: np.ndarray,
) -> pd.Series:
    """PCA SUB シグナルを計算。"""
    window_us = window_us.reindex(columns=US_TICKERS)
    window_jp = window_jp.reindex(columns=range(N_JP))
    combined  = pd.concat([window_us, window_jp], axis=1).fillna(0.0)

    if len(combined) < K + 1:
        return pd.Series(np.nan, index=JP_SECTOR_KEYS)

    z_comb, _, _ = standardize(combined)
    z_comb = z_comb.fillna(0.0)
    Ct = z_comb.values.T @ z_comb.values / len(z_comb)

    C_reg = (1 - LAMBDA) * Ct + LAMBDA * C0

    evals, evecs = np.linalg.eigh(C_reg)
    idx = np.argsort(evals)[::-1]
    V = evecs[:, idx[:K]]

    V_us = V[:N_US, :]
    V_jp = V[N_US:, :]

    _, mu_us, sigma_us = standardize(window_us)
    sigma_us = sigma_us.replace(0, np.nan)
    z_today = (z_us_today - mu_us) / sigma_us
    z_today = z_today.fillna(0)

    ft = V_us.T @ z_today.values
    signal = V_jp @ ft

    return pd.Series(signal, index=JP_SECTOR_KEYS)


# ================================================================
# メイン
# ================================================================

def main() -> None:
    print("=" * 70)
    print("Phase 2: フルバックテスト（個別株版）")
    print("=" * 70)

    # Phase 1 結果を読み込み
    jp_ticker_map = load_phase1_results()
    print(f"\n選定株（17銘柄）:")
    for sector in JP_SECTOR_KEYS:
        ticker = jp_ticker_map.get(sector, "N/A")
        name = JP_SECTOR_NAMES.get(sector, "")
        print(f"  {name:<12} {sector:<15} {ticker}")

    # ETF データ取得
    print(f"\n【1】米国セクターETFデータ取得...")
    us_raw = yf.download(
        US_TICKERS,
        start="2010-01-01",
        auto_adjust=True,
        progress=False,
    )

    # 個別株データ取得
    print(f"【2】JP個別株データ取得...")
    jp_tickers = [f"{int(ticker)}.T" for ticker in jp_ticker_map.values()]
    jp_raw = yf.download(
        jp_tickers,
        start="2010-01-01",
        auto_adjust=True,
        progress=False,
    )

    # リターン計算
    print(f"【3】リターン計算...")
    us_cc = us_raw["Close"].reindex(columns=US_TICKERS).pct_change()

    # JP側は列の順序がランダムな可能性があるので、ティッカー順で正規化
    jp_close = jp_raw["Close"].reindex(columns=jp_tickers)
    jp_open = jp_raw["Open"].reindex(columns=jp_tickers)
    jp_oc = (jp_close - jp_open) / jp_open
    jp_cc = jp_close.pct_change()

    # インデックスをリセット（0〜16）
    jp_oc.columns = range(N_JP)
    jp_cc.columns = range(N_JP)

    common = us_cc.index.intersection(jp_oc.index).intersection(jp_cc.index)
    us_cc  = us_cc.loc[common]
    jp_oc  = jp_oc.loc[common]
    jp_cc  = jp_cc.loc[common]

    print(f"  共通営業日: {len(common)}日 ({common[0].date()} ~ {common[-1].date()})")

    # Cfull/C0/V0 構築
    print(f"【4】Cfull/C0/V0構築...")
    mask = (us_cc.index >= CFULL_START) & (us_cc.index <= CFULL_END)
    us_c = us_cc.loc[mask]
    jp_c = jp_cc.loc[mask]
    combined_cfull = pd.concat([us_c, jp_c], axis=1).fillna(0.0)
    z_c, _, _ = standardize(combined_cfull)
    z_c = z_c.fillna(0.0)
    cfull = z_c.values.T @ z_c.values / len(z_c)

    V0 = build_prior_subspace()
    C0 = build_C0(V0, cfull)

    # バックテスト実行
    print(f"【5】バックテスト実行...")

    sig_rows  = []
    trd_rows  = []
    perf_rows = []

    capital = INITIAL_CAPITAL
    peak    = INITIAL_CAPITAL
    wins    = 0
    total   = 0

    for i in range(L, len(common) - 1):
        today    = common[i]
        tomorrow = common[i + 1]

        window_us  = us_cc.iloc[i - L : i]
        window_jp  = jp_cc.iloc[i - L : i]
        z_us_today = us_cc.loc[today]

        signal = calc_pca_sub_signal(window_us, window_jp, z_us_today, C0, V0)
        ranked = signal.dropna().sort_values(ascending=False)

        long_sectors  = ranked.head(TOP_N).index.tolist()
        short_sectors = ranked.tail(TOP_N).index.tolist()

        sig_strength = float(ranked.iloc[0] - ranked.iloc[-1]) if len(ranked) > 0 else np.nan
        sig_spread   = float(ranked.iloc[:TOP_N].mean() - ranked.iloc[-TOP_N:].mean()) if len(ranked) >= TOP_N * 2 else np.nan

        oc_row = jp_oc.loc[tomorrow]
        long_ret  = float(oc_row[[JP_SECTOR_KEYS.index(s) for s in long_sectors]].mean()) if long_sectors else np.nan
        short_ret = float(oc_row[[JP_SECTOR_KEYS.index(s) for s in short_sectors]].mean()) if short_sectors else np.nan

        if np.isnan(long_ret) or np.isnan(short_ret):
            continue

        strat_ret = long_ret - short_ret
        is_correct = int(long_ret > short_ret)

        capital *= (1 + strat_ret)
        if capital > peak:
            peak = capital
        wins += is_correct
        total += 1

        date_str = tomorrow.strftime("%Y-%m-%d")

        # signals 行
        sig_row = {"date": date_str, "signal_strength": round(sig_strength, 6), "signal_spread": round(sig_spread, 6)}
        for sector in JP_SECTOR_KEYS:
            sig_row[f"signal_{sector}"] = round(float(signal.get(sector, np.nan)), 6)
        for rank, sector in enumerate(ranked.index, 1):
            sig_row[f"rank_{sector}"] = rank
        sig_rows.append(sig_row)

        # trades 行
        trd_rows.append({
            "date": date_str,
            "weekday": WEEKDAY_EN[tomorrow.dayofweek],
            "month": tomorrow.month,
            "long_1": long_sectors[0] if len(long_sectors) > 0 else "",
            "long_2": long_sectors[1] if len(long_sectors) > 1 else "",
            "long_3": long_sectors[2] if len(long_sectors) > 2 else "",
            "short_1": short_sectors[0] if len(short_sectors) > 0 else "",
            "short_2": short_sectors[1] if len(short_sectors) > 1 else "",
            "short_3": short_sectors[2] if len(short_sectors) > 2 else "",
            "signal_strength": round(sig_strength, 6),
            "signal_spread": round(sig_spread, 6),
            "long_return": round(long_ret, 6),
            "short_return": round(short_ret, 6),
            "strategy_return": round(strat_ret, 6),
            "long_only_return": round(long_ret, 6),
            "is_correct": is_correct,
        })

        # performance 行
        recent_20 = [r["is_correct"] for r in trd_rows[-20:]]
        win_rate_20d = sum(recent_20) / len(recent_20) if recent_20 else 0.0
        perf_rows.append({
            "date": date_str,
            "strategy_return": round(strat_ret, 6),
            "long_return": round(long_ret, 6),
            "short_return": round(short_ret, 6),
            "long_only_return": round(long_ret, 6),
            "capital": round(capital, 0),
            "capital_long_only": round(capital, 0),
            "pnl": round(capital - INITIAL_CAPITAL, 0),
            "pnl_pct": round((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL, 6),
            "is_correct": is_correct,
            "win_rate_total": round(wins / total, 4),
            "win_rate_20d": round(win_rate_20d, 4),
            "total_days": total,
        })

        if total % 500 == 0:
            print(f"  {total}日完了...")

    # CSV保存
    print(f"【6】結果保存...")
    pd.DataFrame(sig_rows).to_csv(PROCESSED_DIR / "signals_v2.csv", index=False)
    pd.DataFrame(trd_rows).to_csv(PROCESSED_DIR / "trades_v2.csv", index=False)
    pd.DataFrame(perf_rows).to_csv(PROCESSED_DIR / "performance_v2.csv", index=False)

    # サマリー計算
    perf_df = pd.DataFrame(perf_rows)
    rets = perf_df["strategy_return"].dropna()
    cum = (1 + rets).cumprod()
    annual_ret = rets.mean() * 252 * 100
    annual_vol = rets.std() * np.sqrt(252) * 100
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    mdd = ((cum / cum.cummax()) - 1).min() * 100
    win_rate = wins / total * 100

    summary = f"""
======================================================================
Phase 2: バックテスト結果（個別株版）
======================================================================

期間: {common[L].date()} ~ {common[-1].date()}
営業日数: {total}日

【成績】
  年率リターン:      {annual_ret:+.2f}%
  年率ボラティリティ: {annual_vol:.2f}%
  Sharpe レシオ:    {sharpe:.2f}
  最大ドローダウン:  {mdd:.2f}%
  最終資産:          {capital:,.0f}円（初期 {INITIAL_CAPITAL:,.0f}円）
  総損益:            {capital - INITIAL_CAPITAL:+,.0f}円
  勝率:              {win_rate:.1f}%

【比較】
  ETF版（既存）:     年率 +20.59%
  個別株版（新規）:  年率 {annual_ret:+.2f}%

======================================================================
"""

    output_path = REPORTS_DIR / "phase2_summary.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)
    print(f"詳細: {PROCESSED_DIR / 'signals_v2.csv'}")
    print(f"      {PROCESSED_DIR / 'trades_v2.csv'}")
    print(f"      {PROCESSED_DIR / 'performance_v2.csv'}")


if __name__ == "__main__":
    main()
