"""
step0_calc_full_signals.py
ETF版シグナルを 2010-01-01 から全期間で計算する

INPUT:
  - yfinance: US ETF (XLB〜XLY) + JP ETF (1617.T〜1633.T)

OUTPUT:
  - data/processed/signals_etf_full.csv（2010-04〜現在）
    カラム: date, signal_strength, signal_spread,
            signal_food, ..., signal_realestate,
            rank_food, ..., rank_realestate

NOTE:
  - Phase 1 の IC テストはこのファイルを入力として使う
  - signals.csv（2021〜のみ）は上書きしない
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR  # noqa: E402

# ================================================================
# 固定パラメータ（変更禁止）
# ================================================================

L = 60
K = 3
LAMBDA = 0.9
TOP_N = 3

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

N_US = len(US_TICKERS)
N_JP = len(JP_TICKERS)
N    = N_US + N_JP


# ================================================================
# PCA SUB 関数（論文準拠）
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


def calc_signal(window_us, window_jp, z_us_today, C0, V0) -> pd.Series:
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

    _, mu_us, sigma_us = standardize(window_us)
    z_today = (z_us_today - mu_us) / sigma_us.replace(0, np.nan)
    z_today = z_today.fillna(0)

    ft = V[:N_US, :].T @ z_today.values
    return pd.Series(V[N_US:, :] @ ft, index=JP_TICKERS)


# ================================================================
# メイン
# ================================================================

def main() -> None:
    out_path = PROCESSED_DIR / "signals_etf_full.csv"

    print("=" * 60)
    print("Step 0: ETF版フルシグナル計算（2010〜現在）")
    print("=" * 60)

    # データ取得
    print("\n[1] US ETF データ取得...")
    us_raw = yf.download(US_TICKERS, start="2010-01-01", auto_adjust=True, progress=False)
    print(f"  {len(us_raw)}日取得")

    print("[2] JP ETF データ取得...")
    jp_raw = yf.download(JP_TICKERS, start="2010-01-01", auto_adjust=True, progress=False)
    print(f"  {len(jp_raw)}日取得")

    # リターン計算
    print("[3] リターン計算...")
    us_cc = us_raw["Close"].reindex(columns=US_TICKERS).pct_change()
    jp_cc = jp_raw["Close"].reindex(columns=JP_TICKERS).pct_change()

    common = us_cc.index.intersection(jp_cc.index)
    us_cc  = us_cc.loc[common]
    jp_cc  = jp_cc.loc[common]
    print(f"  共通営業日: {len(common)}日 ({common[0].date()} ~ {common[-1].date()})")

    # Cfull / C0 / V0
    print("[4] Cfull/C0/V0 構築...")
    mask = (us_cc.index >= CFULL_START) & (us_cc.index <= CFULL_END)
    combined_cfull = pd.concat([us_cc.loc[mask], jp_cc.loc[mask]], axis=1).fillna(0.0)
    z_c, _, _ = standardize(combined_cfull)
    cfull = z_c.fillna(0.0).values.T @ z_c.fillna(0.0).values / len(z_c)
    V0 = build_prior_subspace()
    C0 = build_C0(V0, cfull)
    print(f"  Cfull 期間: {CFULL_START} ~ {CFULL_END}")

    # シグナル計算
    print("[5] シグナル計算中...")
    rows = []
    for i in range(L, len(common) - 1):
        today    = common[i]
        tomorrow = common[i + 1]

        signal = calc_signal(
            us_cc.iloc[i - L : i],
            jp_cc.iloc[i - L : i],
            us_cc.loc[today],
            C0, V0,
        )
        ranked = signal.dropna().sort_values(ascending=False)

        sig_strength = float(ranked.iloc[0]  - ranked.iloc[-1])        if len(ranked) > 0      else np.nan
        sig_spread   = float(ranked.iloc[:TOP_N].mean() - ranked.iloc[-TOP_N:].mean()) if len(ranked) >= TOP_N * 2 else np.nan

        row = {
            "date":             tomorrow.strftime("%Y-%m-%d"),
            "signal_strength":  round(sig_strength, 6),
            "signal_spread":    round(sig_spread,   6),
        }
        for t in JP_TICKERS:
            row[f"signal_{JP_KEYS[t]}"] = round(float(signal.get(t, np.nan)), 6)
        for rank, t in enumerate(ranked.index, 1):
            row[f"rank_{JP_KEYS[t]}"] = rank
        rows.append(row)

        if (i - L) % 500 == 0:
            print(f"  {i - L + 1}/{len(common) - L - 1} 日完了...")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print(f"\n[完了] {len(df)}日分のシグナルを保存しました")
    print(f"  保存先: {out_path}")
    print(f"  期間:   {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    print("\n次のステップ: python scripts/phase1_ic_test.py")


if __name__ == "__main__":
    main()
