"""
06_backtest_double.py
DOUBLE 戦略のバックテスト

MOM と PCA SUB の2つのシグナルを組み合わせる
2×2のダブルソート:
  MOM High × PCA SUB High → ロング
  MOM Low  × PCA SUB Low  → ショート
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ============================================================
# パラメータ（論文と同じ値）
# ============================================================

L    = 60
K    = 3
LAMBDA = 0.9
Q    = 0.3

N_US = 11
N_JP = 17

US_TICKERS = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
JP_TICKERS = ["1617.T","1618.T","1619.T","1620.T","1621.T","1622.T","1623.T",
              "1624.T","1625.T","1626.T","1627.T","1628.T","1629.T","1630.T",
              "1631.T","1632.T","1633.T"]

US_CYCLICAL  = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE = ["XLK", "XLP", "XLU", "XLV"]
JP_CYCLICAL  = ["1618.T", "1625.T", "1629.T", "1631.T"]
JP_DEFENSIVE = ["1617.T", "1621.T", "1627.T", "1630.T"]

CFULL_START = "2010-01-01"
CFULL_END   = "2014-12-31"

# ============================================================
# 1. データ読み込み
# ============================================================

print("=== データ読み込み ===")

us_cc  = pd.read_csv(DATA_DIR / "us_cc_returns.csv", index_col=0, parse_dates=True)
jp_oc  = pd.read_csv(DATA_DIR / "jp_oc_returns.csv", index_col=0, parse_dates=True)
jp_raw = pd.read_csv(DATA_DIR / "jp_etf.csv",
                     header=[0, 1], index_col=0, parse_dates=True)
jp_cc  = jp_raw["Close"].pct_change()

common_dates = us_cc.index.intersection(jp_oc.index).intersection(jp_cc.index)
us_cc = us_cc.reindex(columns=US_TICKERS).loc[common_dates]
jp_cc = jp_cc.reindex(columns=JP_TICKERS).loc[common_dates]
jp_oc = jp_oc.reindex(columns=JP_TICKERS).loc[common_dates]

print(f"共通営業日数: {len(common_dates)}")

# ============================================================
# 2. 共通関数（05から流用）
# ============================================================

def standardize(df: pd.DataFrame):
    mu    = df.mean()
    sigma = df.std().replace(0, np.nan)
    z     = (df - mu) / sigma
    return z, mu, sigma

def build_prior_subspace(available_tickers: list) -> np.ndarray:
    n     = len(available_tickers)
    is_us = np.array([t in US_TICKERS for t in available_tickers], dtype=float)
    is_jp = np.array([t in JP_TICKERS for t in available_tickers], dtype=float)
    v1    = np.ones(n) / np.sqrt(n)
    v2_raw = is_us - is_jp
    v2_raw = v2_raw - np.dot(v2_raw, v1) * v1
    norm2  = np.linalg.norm(v2_raw)
    v2     = v2_raw / norm2 if norm2 > 1e-10 else np.zeros(n)
    v3_raw = np.zeros(n)
    for i, t in enumerate(available_tickers):
        if t in US_CYCLICAL or t in JP_CYCLICAL:
            v3_raw[i] = +1.0
        elif t in US_DEFENSIVE or t in JP_DEFENSIVE:
            v3_raw[i] = -1.0
    v3_raw = v3_raw - np.dot(v3_raw, v1) * v1 - np.dot(v3_raw, v2) * v2
    norm3  = np.linalg.norm(v3_raw)
    v3     = v3_raw / norm3 if norm3 > 1e-10 else np.zeros(n)
    return np.column_stack([v1, v2, v3])

def build_C0(V0, cfull):
    D0     = np.diag(np.diag(V0.T @ cfull @ V0))
    C0_raw = V0 @ D0 @ V0.T
    diag_v = np.where(np.diag(C0_raw) > 0, np.diag(C0_raw), 1.0)
    D_inv  = np.diag(1.0 / np.sqrt(diag_v))
    C0     = D_inv @ C0_raw @ D_inv
    np.fill_diagonal(C0, 1.0)
    return C0

# Cfull の計算
cfull_mask    = (common_dates >= CFULL_START) & (common_dates <= CFULL_END)
cfull_dates   = common_dates[cfull_mask]
combined_cfull = pd.concat(
    [us_cc.loc[cfull_dates], jp_cc.loc[cfull_dates]], axis=1
).dropna(axis=1, how="all").fillna(0)
z_cfull, _, _ = standardize(combined_cfull)
cfull_matrix  = z_cfull.values.T @ z_cfull.values / len(z_cfull)
cfull_tickers = combined_cfull.columns.tolist()

def calc_pca_sub_signal(window_us, window_jp, z_us_today,
                        lam=LAMBDA, k=K):
    combined  = pd.concat([window_us, window_jp], axis=1)
    combined  = combined.dropna(axis=1, how="all").fillna(0)
    if len(combined.columns) < k + 1:
        return pd.Series(np.nan, index=JP_TICKERS)
    available = combined.columns.tolist()
    us_avail  = [t for t in available if t in US_TICKERS]
    jp_avail  = [t for t in available if t in JP_TICKERS]
    if not us_avail or not jp_avail:
        return pd.Series(np.nan, index=JP_TICKERS)
    z_combined, _, _ = standardize(combined)
    Ct        = z_combined.values.T @ z_combined.values / len(z_combined)
    V0_avail  = build_prior_subspace(available)
    cfull_idx = [cfull_tickers.index(t) for t in available if t in cfull_tickers]
    if len(cfull_idx) == Ct.shape[0]:
        C0_avail = build_C0(V0_avail, cfull_matrix[np.ix_(cfull_idx, cfull_idx)])
    else:
        C0_avail = np.eye(len(available))
    C_reg = (1 - lam) * Ct + lam * C0_avail
    eigenvalues, eigenvectors = np.linalg.eigh(C_reg)
    idx   = np.argsort(eigenvalues)[::-1]
    V     = eigenvectors[:, idx[:k]]
    us_idx = [available.index(t) for t in us_avail]
    jp_idx = [available.index(t) for t in jp_avail]
    V_us, V_jp = V[us_idx, :], V[jp_idx, :]
    w_us_avail   = window_us.reindex(columns=us_avail).fillna(0)
    _, mu_us, sg = standardize(w_us_avail)
    z_today = ((z_us_today.reindex(us_avail) - mu_us) / sg.replace(0, np.nan)).fillna(0)
    ft      = V_us.T @ z_today.values
    signal  = pd.Series(np.nan, index=JP_TICKERS)
    signal.loc[jp_avail] = V_jp @ ft
    return signal

def calc_performance(strategy_returns: pd.Series, name: str):
    r    = strategy_returns.dropna()
    ar   = r.mean() * 252
    risk = r.std()  * np.sqrt(252)
    rr   = ar / risk if risk > 0 else 0.0
    cum  = (1 + r).cumprod()
    mdd  = ((cum / cum.cummax()) - 1).min()
    return {"Strategy": name,
            "AR (%)":   round(ar   * 100, 2),
            "RISK (%)": round(risk * 100, 2),
            "R/R":      round(rr,          2),
            "MDD (%)":  round(mdd  * 100, 2)}, cum

# ============================================================
# 3. DOUBLE ロングショートの計算関数
# ============================================================

def calc_double_longshort(
    signal_mom: pd.Series,
    signal_pca: pd.Series,
    next_returns: pd.Series,
) -> float:
    """
    2×2ダブルソート:
      各シグナルをメディアンで High/Low に二分割
      High × High → ロング
      Low  × Low  → ショート
    """
    # 両方のシグナルがある銘柄だけ使う
    valid = signal_mom.dropna().index.intersection(
            signal_pca.dropna().index)
    if len(valid) < 2:
        return np.nan

    s_mom = signal_mom.loc[valid]
    s_pca = signal_pca.loc[valid]

    # メディアンで High/Low に分類
    med_mom = s_mom.median()
    med_pca = s_pca.median()

    mom_high = s_mom[s_mom >= med_mom].index
    mom_low  = s_mom[s_mom <  med_mom].index
    pca_high = s_pca[s_pca >= med_pca].index
    pca_low  = s_pca[s_pca <  med_pca].index

    # High × High → ロング
    long_stocks  = mom_high.intersection(pca_high)
    # Low × Low   → ショート
    short_stocks = mom_low.intersection(pca_low)

    if len(long_stocks) == 0 or len(short_stocks) == 0:
        return np.nan

    long_ret  = next_returns[long_stocks].mean()
    short_ret = next_returns[short_stocks].mean()

    return long_ret - short_ret

# ============================================================
# 4. MOM シグナルの計算関数（03から流用）
# ============================================================

def calc_mom_signal(window_jp_cc: pd.DataFrame) -> pd.Series:
    return window_jp_cc.mean()

# ============================================================
# 5. DOUBLE 戦略の実行
# ============================================================

print("\n=== DOUBLE 戦略バックテスト ===")
print(f"ウィンドウ: {L}日, λ={LAMBDA}, ダブルソート（メディアン分割）")

double_returns = []
dates = []

for i in range(L, len(common_dates) - 1):
    today    = common_dates[i]
    tomorrow = common_dates[i + 1]

    window_us = us_cc.iloc[i - L : i]
    window_jp = jp_cc.iloc[i - L : i]
    z_us_today = us_cc.loc[today]

    # MOM シグナル
    sig_mom = calc_mom_signal(window_jp)

    # PCA SUB シグナル
    sig_pca = calc_pca_sub_signal(window_us, window_jp, z_us_today)

    # 翌日リターン
    next_ret = jp_oc.loc[tomorrow]

    # DOUBLE ロングショート
    ret = calc_double_longshort(sig_mom, sig_pca, next_ret)
    double_returns.append(ret)
    dates.append(tomorrow)

    if i % 500 == 0:
        print(f"  進捗: {i}/{len(common_dates)} 日完了...")

double_series = pd.Series(double_returns, index=dates, name="DOUBLE").dropna()

# ============================================================
# 6. 結果表示
# ============================================================

perf, _ = calc_performance(double_series, "DOUBLE")

print("\n=== DOUBLE パフォーマンス ===")
print(f"{'指標':<15} {'結果':>10}  {'論文値':>10}")
print("-" * 40)
print(f"{'年率リターン':<15} {perf['AR (%)']:>9.2f}%  {'18.86%':>10}")
print(f"{'年率リスク':<15} {perf['RISK (%)']:>9.2f}%  {'11.16%':>10}")
print(f"{'R/R':<15} {perf['R/R']:>10.2f}  {'1.69':>10}")
print(f"{'最大ドローダウン':<13} {perf['MDD (%)']:>9.2f}%  {'-12.10%':>10}")

double_series.to_csv(DATA_DIR / "strategy_double.csv")
print(f"\n💾 data/strategy_double.csv に保存しました")
print("\n次のステップ: 07_summary.py で4戦略を比較します")