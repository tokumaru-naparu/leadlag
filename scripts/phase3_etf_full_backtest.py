"""
phase3_etf_full_backtest.py
ETF版を 2010年〜現在の全期間でバックテスト（個別株版と同期間での公正比較用）

INPUT:
  - yfinance で US ETF（XLB〜XLY）+ JP ETF（1617.T〜1633.T）を 2010-01-01 から取得

OUTPUT:
  - data/processed/signals_etf_full.csv
  - data/processed/trades_etf_full.csv
  - data/processed/performance_etf_full.csv
  - output/reports/phase3_comparison.txt（ETF版 vs 個別株版の公正比較）
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
# 定数（PCA SUB 論文パラメータ・変更禁止）
# ================================================================

L = 60
K = 3
LAMBDA = 0.9
TOP_N = 3
INITIAL_CAPITAL = 1_000_000

CFULL_START = "2010-01-01"
CFULL_END   = "2014-12-31"

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

US_CYCLICAL  = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE = ["XLK", "XLP", "XLU", "XLV"]
JP_CYCLICAL  = ["1618.T", "1625.T", "1629.T", "1631.T"]
JP_DEFENSIVE = ["1617.T", "1621.T", "1627.T", "1630.T"]

WEEKDAY_EN = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

N_US = len(US_TICKERS)
N_JP = len(JP_TICKERS)
N    = N_US + N_JP


# ================================================================
# PCA SUB 関数群
# ================================================================

def standardize(df: pd.DataFrame):
    mu    = df.mean()
    sigma = df.std().replace(0, np.nan)
    return (df - mu) / sigma, mu, sigma


def build_prior_subspace() -> np.ndarray:
    v1 = np.ones(N) / np.sqrt(N)

    v2_raw = np.zeros(N)
    v2_raw[:N_US] = +1.0
    v2_raw[N_US:] = -1.0
    v2_raw -= np.dot(v2_raw, v1) * v1
    v2 = v2_raw / np.linalg.norm(v2_raw)

    all_tickers = US_TICKERS + JP_TICKERS
    v3_raw = np.zeros(N)
    for i, t in enumerate(all_tickers):
        if t in US_CYCLICAL or t in JP_CYCLICAL:
            v3_raw[i] = +1.0
        elif t in US_DEFENSIVE or t in JP_DEFENSIVE:
            v3_raw[i] = -1.0

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
    window_us = window_us.reindex(columns=US_TICKERS)
    window_jp = window_jp.reindex(columns=JP_TICKERS)
    combined  = pd.concat([window_us, window_jp], axis=1).fillna(0.0)

    if len(combined) < K + 1:
        return pd.Series(np.nan, index=JP_TICKERS)

    z_comb, _, _ = standardize(combined)
    z_comb = z_comb.fillna(0.0)
    Ct = z_comb.values.T @ z_comb.values / len(z_comb)

    C_reg = (1 - LAMBDA) * Ct + LAMBDA * C0

    evals, evecs = np.linalg.eigh(C_reg)
    V = evecs[:, np.argsort(evals)[::-1][:K]]

    V_us = V[:N_US, :]
    V_jp = V[N_US:, :]

    _, mu_us, sigma_us = standardize(window_us)
    sigma_us = sigma_us.replace(0, np.nan)
    z_today = (z_us_today - mu_us) / sigma_us
    z_today = z_today.fillna(0)

    ft = V_us.T @ z_today.values
    return pd.Series(V_jp @ ft, index=JP_TICKERS)


def calc_stats(rets: pd.Series) -> dict:
    cum = (1 + rets).cumprod()
    n_years = len(rets) / 252
    annual_ret = ((cum.iloc[-1]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    annual_vol = rets.std() * np.sqrt(252) * 100
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    mdd = ((cum / cum.cummax()) - 1).min() * 100
    win_rate = (rets > 0).mean() * 100
    return {
        "annual_ret": annual_ret,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
        "final_capital": (cum.iloc[-1] * INITIAL_CAPITAL),
    }


# ================================================================
# メイン
# ================================================================

def main() -> None:
    print("=" * 70)
    print("Phase 3: ETF版フルバックテスト（2010〜現在）")
    print("=" * 70)

    # データ取得
    print("\n【1】US ETF データ取得...")
    us_raw = yf.download(US_TICKERS, start="2010-01-01", auto_adjust=True, progress=False)

    print("【2】JP ETF データ取得...")
    jp_raw = yf.download(JP_TICKERS, start="2010-01-01", auto_adjust=True, progress=False)

    # リターン計算
    print("【3】リターン計算...")
    us_cc = us_raw["Close"].reindex(columns=US_TICKERS).pct_change()
    jp_cc = jp_raw["Close"].reindex(columns=JP_TICKERS).pct_change()
    jp_oc = ((jp_raw["Close"] - jp_raw["Open"]) / jp_raw["Open"]).reindex(columns=JP_TICKERS)

    common = us_cc.index.intersection(jp_cc.index).intersection(jp_oc.index)
    us_cc = us_cc.loc[common]
    jp_cc = jp_cc.loc[common]
    jp_oc = jp_oc.loc[common]

    print(f"  共通営業日: {len(common)}日 ({common[0].date()} ~ {common[-1].date()})")

    # C0/V0 構築
    print("【4】Cfull/C0/V0 構築...")
    mask = (us_cc.index >= CFULL_START) & (us_cc.index <= CFULL_END)
    combined_cfull = pd.concat([us_cc.loc[mask], jp_cc.loc[mask]], axis=1).fillna(0.0)
    z_c, _, _ = standardize(combined_cfull)
    z_c = z_c.fillna(0.0)
    cfull = z_c.values.T @ z_c.values / len(z_c)

    V0 = build_prior_subspace()
    C0 = build_C0(V0, cfull)

    # バックテスト
    print("【5】バックテスト実行...")
    sig_rows, trd_rows, perf_rows = [], [], []
    capital = float(INITIAL_CAPITAL)
    peak    = float(INITIAL_CAPITAL)
    wins = total = 0

    for i in range(L, len(common) - 1):
        today    = common[i]
        tomorrow = common[i + 1]

        signal = calc_pca_sub_signal(
            us_cc.iloc[i - L : i],
            jp_cc.iloc[i - L : i],
            us_cc.loc[today],
            C0, V0,
        )
        ranked      = signal.dropna().sort_values(ascending=False)
        long_t      = ranked.head(TOP_N).index.tolist()
        short_t     = ranked.tail(TOP_N).index.tolist()
        sig_strength = float(ranked.iloc[0] - ranked.iloc[-1]) if len(ranked) > 0 else np.nan
        sig_spread   = float(ranked.iloc[:TOP_N].mean() - ranked.iloc[-TOP_N:].mean()) if len(ranked) >= TOP_N * 2 else np.nan

        oc_row    = jp_oc.loc[tomorrow]
        long_ret  = float(oc_row[long_t].mean())  if long_t  else np.nan
        short_ret = float(oc_row[short_t].mean()) if short_t else np.nan

        if np.isnan(long_ret) or np.isnan(short_ret):
            continue

        strat_ret  = long_ret - short_ret
        is_correct = int(long_ret > short_ret)

        capital *= (1 + strat_ret)
        if capital > peak:
            peak = capital
        wins  += is_correct
        total += 1

        date_str = tomorrow.strftime("%Y-%m-%d")

        # signals 行
        sig_row = {"date": date_str, "signal_strength": round(sig_strength, 6), "signal_spread": round(sig_spread, 6)}
        for t in JP_TICKERS:
            sig_row[f"signal_{JP_KEYS[t]}"] = round(float(signal.get(t, np.nan)), 6)
        for rank, t in enumerate(ranked.index, 1):
            sig_row[f"rank_{JP_KEYS[t]}"] = rank
        sig_rows.append(sig_row)

        # trades 行
        trd_rows.append({
            "date": date_str,
            "weekday": WEEKDAY_EN[tomorrow.dayofweek],
            "month": tomorrow.month,
            "long_1":  JP_KEYS.get(long_t[0],  "") if len(long_t)  > 0 else "",
            "long_2":  JP_KEYS.get(long_t[1],  "") if len(long_t)  > 1 else "",
            "long_3":  JP_KEYS.get(long_t[2],  "") if len(long_t)  > 2 else "",
            "short_1": JP_KEYS.get(short_t[0], "") if len(short_t) > 0 else "",
            "short_2": JP_KEYS.get(short_t[1], "") if len(short_t) > 1 else "",
            "short_3": JP_KEYS.get(short_t[2], "") if len(short_t) > 2 else "",
            "signal_strength": round(sig_strength, 6),
            "signal_spread":   round(sig_spread,   6),
            "long_return":     round(long_ret,     6),
            "short_return":    round(short_ret,    6),
            "strategy_return": round(strat_ret,    6),
            "long_only_return":round(long_ret,     6),
            "is_correct":      is_correct,
        })

        # performance 行
        recent_20 = [r["is_correct"] for r in trd_rows[-20:]]
        perf_rows.append({
            "date":              date_str,
            "strategy_return":   round(strat_ret,  6),
            "long_return":       round(long_ret,   6),
            "short_return":      round(short_ret,  6),
            "long_only_return":  round(long_ret,   6),
            "capital":           round(capital,    0),
            "capital_long_only": round(capital,    0),
            "pnl":               round(capital - INITIAL_CAPITAL, 0),
            "pnl_pct":           round((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL, 6),
            "is_correct":        is_correct,
            "win_rate_total":    round(wins / total, 4),
            "win_rate_20d":      round(sum(recent_20) / len(recent_20), 4) if recent_20 else 0.0,
            "total_days":        total,
        })

        if total % 500 == 0:
            print(f"  {total}日完了...")

    # 保存
    print("【6】結果保存...")
    pd.DataFrame(sig_rows).to_csv(PROCESSED_DIR / "signals_etf_full.csv", index=False)
    pd.DataFrame(trd_rows).to_csv(PROCESSED_DIR / "trades_etf_full.csv", index=False)
    pd.DataFrame(perf_rows).to_csv(PROCESSED_DIR / "performance_etf_full.csv", index=False)

    # 公正比較
    print("【7】公正比較（同期間）...")
    etf_rets = pd.DataFrame(perf_rows)["strategy_return"].dropna()
    v2_path = PROCESSED_DIR / "performance_v2.csv"
    v2_rets  = pd.read_csv(v2_path)["strategy_return"].dropna() if v2_path.exists() else None

    etf_stats = calc_stats(etf_rets)

    report = []
    report.append("=" * 70)
    report.append("Phase 3: 公正比較レポート（同期間: 2010〜2026）")
    report.append("=" * 70)
    report.append("")
    report.append("【ETF版（全期間）】")
    report.append(f"  期間:      {perf_rows[0]['date']} ~ {perf_rows[-1]['date']}")
    report.append(f"  日数:      {total}日")
    report.append(f"  年率:      {etf_stats['annual_ret']:+.2f}%")
    report.append(f"  ボラ:      {etf_stats['annual_vol']:.2f}%")
    report.append(f"  Sharpe:    {etf_stats['sharpe']:.2f}")
    report.append(f"  最大DD:    {etf_stats['mdd']:.2f}%")
    report.append(f"  勝率:      {etf_stats['win_rate']:.1f}%")
    report.append(f"  最終資産:  {etf_stats['final_capital']:,.0f}円")

    if v2_rets is not None:
        v2_stats = calc_stats(v2_rets)
        # 同期間に揃える（開始が遅い方に合わせる）
        etf_df = pd.DataFrame(perf_rows)[["date", "strategy_return"]]
        etf_df["date"] = pd.to_datetime(etf_df["date"])
        v2_df = pd.read_csv(v2_path, parse_dates=["date"])[["date", "strategy_return"]]
        common_start = max(etf_df["date"].min(), v2_df["date"].min())
        etf_trim = etf_df[etf_df["date"] >= common_start]["strategy_return"].dropna()
        v2_trim  = v2_df[v2_df["date"]  >= common_start]["strategy_return"].dropna()
        etf_trim_stats = calc_stats(etf_trim)
        v2_trim_stats  = calc_stats(v2_trim)

        report.append("")
        report.append(f"【個別株版（同期間: {common_start.date()}〜）】")
        report.append(f"  日数:      {len(v2_trim)}日")
        report.append(f"  年率:      {v2_trim_stats['annual_ret']:+.2f}%")
        report.append(f"  ボラ:      {v2_trim_stats['annual_vol']:.2f}%")
        report.append(f"  Sharpe:    {v2_trim_stats['sharpe']:.2f}")
        report.append(f"  最大DD:    {v2_trim_stats['mdd']:.2f}%")
        report.append(f"  勝率:      {v2_trim_stats['win_rate']:.1f}%")

        report.append("")
        report.append("【公正比較（同期間）】")
        report.append(f"{'指標':<20} {'ETF版':>12} {'個別株版':>12} {'差異':>10}")
        report.append("-" * 56)
        for label, e, v in [
            ("年率リターン(%)",  etf_trim_stats["annual_ret"],  v2_trim_stats["annual_ret"]),
            ("ボラティリティ(%)", etf_trim_stats["annual_vol"],  v2_trim_stats["annual_vol"]),
            ("Sharpe",          etf_trim_stats["sharpe"],       v2_trim_stats["sharpe"]),
            ("最大DD(%)",        etf_trim_stats["mdd"],          v2_trim_stats["mdd"]),
            ("勝率(%)",          etf_trim_stats["win_rate"],     v2_trim_stats["win_rate"]),
        ]:
            diff = e - v
            report.append(f"  {label:<18} {e:>12.2f} {v:>12.2f} {diff:>+10.2f}")

        report.append("")
        # 判定
        etf_better = (etf_trim_stats["sharpe"] >= v2_trim_stats["sharpe"] * 1.1
                      and etf_trim_stats["mdd"] > v2_trim_stats["mdd"])  # DDはマイナスなので大きい方が浅い
        report.append("【判定】")
        if etf_better:
            report.append("  → ETF版が統計的に優位（Sharpe >= 1.1x かつ DD 浅い）")
            report.append("  → ただし流動性問題（スプレッド/逆日歩）は依然残る")
            report.append("  → 実運用 or ペーパートレード継続を検討すること")
        else:
            report.append("  → 個別株版に切り替え推奨")
            report.append("  → 次のステップ: update_daily_pipeline.py を v2 対応に更新")

    report_text = "\n".join(report)
    print("\n" + report_text)

    output_path = REPORTS_DIR / "phase3_comparison.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n保存: {output_path}")


if __name__ == "__main__":
    main()
