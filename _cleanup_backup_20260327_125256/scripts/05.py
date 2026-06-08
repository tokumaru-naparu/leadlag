"""
05_backtest_pca_sub.py
PCA SUB 戦略のバックテスト（論文のメイン手法）

PCA PLAINとの違い:
  PCA PLAIN → 相関行列をそのままPCA（ノイズを拾いやすい）
  PCA SUB   → 「経済的に意味のある3方向」に相関行列を縮約してからPCA
              → ノイズが減って予測精度が上がる

3つの事前方向（部分空間）:
  v1: グローバルファクター     （全業種が同じ方向に動く）
  v2: 国スプレッドファクター   （米国 vs 日本）
  v3: シクリカル/ディフェンシブ（景気敏感 vs 景気防衛）

論文パラメータ:
  λ = 0.9  （正則化の強さ。1に近いほど事前情報を強く使う）
  K = 3    （使う主成分の数）
  L = 60   （ローリングウィンドウ）
  Cfullの推定期間: 2010-01-01 〜 2014-12-31
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ============================================================
# パラメータ（論文と同じ値）
# ============================================================

L      = 60    # ローリングウィンドウ
K      = 3     # 主成分数
K0     = 3     # 事前部分空間の次元
LAMBDA = 0.9   # 正則化パラメータ（論文: λ=0.9）
Q      = 0.3   # ロング/ショート比率

N_US = 11
N_JP = 17
N    = N_US + N_JP   # 28業種合計

# Cfull（事前相関行列）の推定期間（論文と同じ）
CFULL_START = "2010-01-01"
CFULL_END   = "2014-12-31"

# ============================================================
# シクリカル/ディフェンシブのラベル（論文と同じ）
# ============================================================

US_TICKERS = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
JP_TICKERS = ["1617.T","1618.T","1619.T","1620.T","1621.T","1622.T","1623.T",
              "1624.T","1625.T","1626.T","1627.T","1628.T","1629.T","1630.T",
              "1631.T","1632.T","1633.T"]

# 論文記載のラベル
US_CYCLICAL   = ["XLB", "XLE", "XLF", "XLRE"]          # 景気敏感
US_DEFENSIVE  = ["XLK", "XLP", "XLU", "XLV"]           # 景気防衛
JP_CYCLICAL   = ["1618.T", "1625.T", "1629.T", "1631.T"]
JP_DEFENSIVE  = ["1617.T", "1621.T", "1627.T", "1630.T"]

# ============================================================
# 1. データ読み込み
# ============================================================

print("=== データ読み込み ===")

us_cc = pd.read_csv(DATA_DIR / "us_cc_returns.csv", index_col=0, parse_dates=True)
jp_oc = pd.read_csv(DATA_DIR / "jp_oc_returns.csv", index_col=0, parse_dates=True)

jp_raw = pd.read_csv(DATA_DIR / "jp_etf.csv",
                     header=[0, 1], index_col=0, parse_dates=True)
jp_cc = jp_raw["Close"].pct_change()

# 共通営業日に揃える
common_dates = us_cc.index.intersection(jp_oc.index).intersection(jp_cc.index)
us_cc = us_cc.loc[common_dates]
jp_oc = jp_oc.loc[common_dates]
jp_cc = jp_cc.loc[common_dates]

# ティッカー順序を論文と揃える（V0の符号ベクトルに影響する）
us_cc = us_cc.reindex(columns=US_TICKERS)
jp_cc = jp_cc.reindex(columns=JP_TICKERS)
jp_oc = jp_oc.reindex(columns=JP_TICKERS)

print(f"共通営業日数: {len(common_dates)}")
print(f"期間: {common_dates[0].date()} 〜 {common_dates[-1].date()}")

# ============================================================
# 2. 標準化関数
# ============================================================

def standardize(window_df: pd.DataFrame):
    mu    = window_df.mean()
    sigma = window_df.std().replace(0, np.nan)
    z     = (window_df - mu) / sigma
    return z, mu, sigma


# ============================================================
# 3. 事前部分空間 V0 の構築（論文 3.1節）
# ============================================================

def build_prior_subspace() -> np.ndarray:
    """
    3つの直交ベクトルからなる事前部分空間 V0 を作る

    v1: グローバルファクター（全銘柄に等しい重み）
    v2: 国スプレッド（米国=正、日本=負）をv1に直交化
    v3: シクリカル/ディフェンシブをv1,v2に直交化

    返り値:
        V0: shape (N, K0) = (28, 3) の列直交行列
    """

    # ── v1: グローバルファクター ────────────────────────
    v1 = np.ones(N)
    v1 = v1 / np.linalg.norm(v1)

    # ── v2: 国スプレッド（米国=+1, 日本=-1） ───────────
    v2_raw = np.zeros(N)
    v2_raw[:N_US] = +1.0    # 米国業種
    v2_raw[N_US:] = -1.0    # 日本業種

    # v1に直交化（グラム・シュミット法）
    # 「v1の成分を引き算してv1と直交にする」
    v2_raw = v2_raw - np.dot(v2_raw, v1) * v1
    v2 = v2_raw / np.linalg.norm(v2_raw)

    # ── v3: シクリカル(+1) / ディフェンシブ(-1) ─────────
    v3_raw = np.zeros(N)

    all_tickers = US_TICKERS + JP_TICKERS
    for i, t in enumerate(all_tickers):
        if t in US_CYCLICAL or t in JP_CYCLICAL:
            v3_raw[i] = +1.0
        elif t in US_DEFENSIVE or t in JP_DEFENSIVE:
            v3_raw[i] = -1.0
        # ラベルなし業種は 0 のまま

    # v1, v2 に直交化
    v3_raw = v3_raw - np.dot(v3_raw, v1) * v1
    v3_raw = v3_raw - np.dot(v3_raw, v2) * v2
    v3 = v3_raw / np.linalg.norm(v3_raw)

    V0 = np.column_stack([v1, v2, v3])   # shape: (28, 3)
    return V0


V0 = build_prior_subspace()
print(f"\n事前部分空間 V0: shape={V0.shape}")
print("列直交確認（V0^T @ V0 ≈ 単位行列になるはず）:")
print(np.round(V0.T @ V0, 4))


# ============================================================
# 4. ターゲット行列 C0 の構築（論文 式10〜12）
# ============================================================

def build_C0(V0: np.ndarray, cfull: np.ndarray) -> np.ndarray:
    """
    事前エクスポージャー行列 C0 を作る

    C0 は「事前部分空間の方向だけを残した相関行列」

    手順:
      D0 = diag(V0^T @ Cfull @ V0)   ← 事前方向の固有値
      C0_raw = V0 @ D0 @ V0^T         ← 低ランク行列に戻す
      C0 = 対角を1に正規化             ← 相関行列の形に整える
    """
    # D0: 事前方向に沿った固有値（対角行列）
    D0 = np.diag(np.diag(V0.T @ cfull @ V0))   # shape: (K0, K0)

    # C0_raw: 低ランク行列
    C0_raw = V0 @ D0 @ V0.T                     # shape: (N, N)

    # 対角要素で正規化して相関行列に変換
    diag_vals = np.diag(C0_raw)
    diag_vals = np.where(diag_vals > 0, diag_vals, 1.0)   # 0除算防止
    D_inv_sqrt = np.diag(1.0 / np.sqrt(diag_vals))
    C0 = D_inv_sqrt @ C0_raw @ D_inv_sqrt

    # 対角を強制的に1にする
    np.fill_diagonal(C0, 1.0)

    return C0


# Cfull の計算（2010〜2014年の全データで推定）
print("\n=== Cfull（事前相関行列）の計算 ===")
cfull_mask = (common_dates >= CFULL_START) & (common_dates <= CFULL_END)
cfull_dates = common_dates[cfull_mask]

us_cfull = us_cc.loc[cfull_dates]
jp_cfull = jp_cc.loc[cfull_dates]
us_cfull = us_cfull.reindex(columns=US_TICKERS)
jp_cfull = jp_cfull.reindex(columns=JP_TICKERS)
combined_cfull = pd.concat([us_cfull, jp_cfull], axis=1).fillna(0.0)

if len(combined_cfull) == 0:
    raise ValueError("Cfull estimation data is empty. Check input files and date range.")

z_cfull, _, _ = standardize(combined_cfull)
z_cfull = z_cfull.fillna(0.0)
cfull_matrix = (z_cfull.values.T @ z_cfull.values) / len(z_cfull)

C0 = build_C0(V0, cfull_matrix)
print(f"Cfull 推定期間: {cfull_dates[0].date()} 〜 {cfull_dates[-1].date()}")
print(f"C0 shape: {C0.shape}, 対角確認（全部1になるはず）: min={np.diag(C0).min():.4f}")


# ============================================================
# 5. PCA SUB シグナルの計算関数（論文 式13〜19）
# ============================================================

def calc_pca_sub_signal(
    window_us: pd.DataFrame,
    window_jp: pd.DataFrame,
    z_us_today: pd.Series,
    C0: np.ndarray,
    V0: np.ndarray,
    lam: float = LAMBDA,
    k: int = K,
) -> pd.Series:
    """
    部分空間正則化PCAで日本業種のシグナルを計算する

    PCA PLAINとの違い:
      C_reg = (1-λ)×Ct + λ×C0
      ↑ λ=0.9 なので C0 を90%使う（事前情報を強く反映）
    """
    # ── Step1: 日米を結合して相関行列を計算 ─────────────
    # C0 (28x28) と次元をそろえるため、常に同じ銘柄順で結合する。
    window_us = window_us.reindex(columns=US_TICKERS)
    window_jp = window_jp.reindex(columns=JP_TICKERS)
    combined = pd.concat([window_us, window_jp], axis=1)

    # 先頭行のpct_change由来NaNや上場前期間は0で埋めて次元を維持する。
    combined = combined.fillna(0.0)

    if len(combined) < k + 1:
        return pd.Series(np.nan, index=window_jp.columns)

    z_combined, _, _ = standardize(combined)
    z_combined = z_combined.fillna(0.0)
    Ct = z_combined.values.T @ z_combined.values / len(z_combined)

    # ── Step2: 正則化相関行列を計算（論文 式13） ─────────
    # C_reg = (1-λ)×Ct + λ×C0
    C_reg = (1 - lam) * Ct + lam * C0

    # ── Step3: 固有分解して上位K個を取り出す ─────────────
    eigenvalues, eigenvectors = np.linalg.eigh(C_reg)
    idx = np.argsort(eigenvalues)[::-1]
    V = eigenvectors[:, idx[:k]]   # shape: (N, K)

    V_us = V[:N_US, :]    # 米国ブロック
    V_jp = V[N_US:, :]    # 日本ブロック

    # ── Step4: 当日米国ショックをファクタースコアに変換 ──
    _, mu_us, sigma_us = standardize(window_us)
    sigma_us = sigma_us.replace(0, np.nan)
    z_today = (z_us_today - mu_us) / sigma_us
    z_today = z_today.fillna(0)

    ft = V_us.T @ z_today.values   # shape: (K,)

    # ── Step5: 日本側ローディングで復元 → シグナル ───────
    signal = V_jp @ ft

    return pd.Series(signal, index=window_jp.columns)


# ============================================================
# 6. パフォーマンス指標・ロングショート関数（前ファイルから流用）
# ============================================================

def calc_performance(strategy_returns: pd.Series, name: str):
    r    = strategy_returns.dropna()
    ar   = r.mean() * 252
    risk = r.std()  * np.sqrt(252)
    rr   = ar / risk if risk > 0 else 0.0
    cumulative = (1 + r).cumprod()
    mdd  = ((cumulative / cumulative.cummax()) - 1).min()
    return {
        "Strategy": name,
        "AR (%)":   round(ar   * 100, 2),
        "RISK (%)": round(risk * 100, 2),
        "R/R":      round(rr,          2),
        "MDD (%)":  round(mdd  * 100, 2),
    }, cumulative


def calc_longshort(signal: pd.Series, next_returns: pd.Series, q: float = Q) -> float:
    valid_signal = signal.dropna()
    valid_returns = next_returns.dropna()
    tradable = valid_signal.index.intersection(valid_returns.index)

    if len(tradable) == 0:
        return np.nan

    valid = valid_signal.loc[tradable]
    upper = valid.quantile(1 - q)
    lower = valid.quantile(q)
    long_stocks = valid[valid > upper].index
    short_stocks = valid[valid < lower].index

    if len(long_stocks) == 0 or len(short_stocks) == 0:
        return np.nan

    return valid_returns.loc[long_stocks].mean() - valid_returns.loc[short_stocks].mean()


# ============================================================
# 7. PCA SUB 戦略の実行
# ============================================================

print("\n=== PCA SUB 戦略バックテスト ===")
print(f"ウィンドウ: {L}日, 主成分数: {K}, λ={LAMBDA}, ロング/ショート: 上下{int(Q*100)}%")

pca_sub_returns = []
dates = []

for i in range(L, len(common_dates) - 1):
    today    = common_dates[i]
    tomorrow = common_dates[i + 1]

    window_us  = us_cc.iloc[i - L : i]
    window_jp  = jp_cc.iloc[i - L : i]
    z_us_today = us_cc.loc[today]

    signal  = calc_pca_sub_signal(window_us, window_jp, z_us_today, C0, V0)
    next_ret = jp_oc.loc[tomorrow]

    ret = calc_longshort(signal, next_ret)
    pca_sub_returns.append(ret)
    dates.append(tomorrow)

    if i % 500 == 0:
        print(f"  進捗: {i}/{len(common_dates)} 日完了...")

pca_sub_series = pd.Series(pca_sub_returns, index=dates, name="PCA_SUB")
pca_sub_series = pca_sub_series.dropna()

# ============================================================
# 8. 結果表示
# ============================================================

perf, cumret = calc_performance(pca_sub_series, "PCA_SUB")

print("\n=== PCA SUB パフォーマンス ===")
print(f"{'指標':<15} {'結果':>10}  {'論文値':>10}")
print("-" * 40)
print(f"{'年率リターン':<15} {perf['AR (%)']:>9.2f}%  {'23.79%':>10}")
print(f"{'年率リスク':<15} {perf['RISK (%)']:>9.2f}%  {'10.70%':>10}")
print(f"{'R/R':<15} {perf['R/R']:>10.2f}  {'2.22':>10}")
print(f"{'最大ドローダウン':<13} {perf['MDD (%)']:>9.2f}%  {'-9.58%':>10}")

pca_sub_series.to_csv(DATA_DIR / "strategy_pca_sub.csv")
print(f"\n💾 data/strategy_pca_sub.csv に保存しました")
print("\n次のステップ: 06_backtest_double.py でDOUBLE戦略を実装します")