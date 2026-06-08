"""
04_backtest_pca_plain.py
PCA PLAIN 戦略のバックテスト

論文のベースライン戦略その2:
  日米28業種の相関行列にPCAをかけて
  米国の当日リターンから日本の翌日リターンを予測する
  （正則化なし = λ=0）

MOMとの違い:
  MOM       → 日本のデータだけ使う（過去の平均）
  PCA PLAIN → 米国の「今日の動き」を使って日本の「明日」を予測する
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ============================================================
# パラメータ（論文と同じ値）
# ============================================================

L = 60    # ローリングウィンドウ: 60営業日
K = 3     # 使う主成分の数（上位3個）
Q = 0.3   # 上位・下位 30% をロング・ショート

N_US = 11   # 米国業種数
N_JP = 17   # 日本業種数

# ============================================================
# 1. データ読み込み
# ============================================================

print("=== データ読み込み ===")

# 米国 Close-to-Close（シグナルの材料）
us_cc = pd.read_csv(DATA_DIR / "us_cc_returns.csv", index_col=0, parse_dates=True)

# 日本 Open-to-Close（戦略の評価対象）
jp_oc = pd.read_csv(DATA_DIR / "jp_oc_returns.csv", index_col=0, parse_dates=True)

# 日本 Close-to-Close（PCAの相関行列を計算するのに必要）
jp_raw = pd.read_csv(DATA_DIR / "jp_etf.csv",
                     header=[0, 1], index_col=0, parse_dates=True)
jp_cc = jp_raw["Close"].pct_change()

# 共通営業日に揃える
common_dates = us_cc.index.intersection(jp_oc.index).intersection(jp_cc.index)
us_cc = us_cc.loc[common_dates]
jp_oc = jp_oc.loc[common_dates]
jp_cc = jp_cc.loc[common_dates]

print(f"共通営業日数: {len(common_dates)}")
print(f"期間: {common_dates[0].date()} 〜 {common_dates[-1].date()}")

# 米国・日本のティッカー一覧を保存
us_tickers = us_cc.columns.tolist()
jp_tickers = jp_cc.columns.tolist()

# ============================================================
# 2. 標準化リターンの計算関数
# ============================================================

def standardize(window_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    ウィンドウ内のリターンを標準化する（平均0・標準偏差1に変換）

    論文式(8)(9):
      z_i,τ = (r_i,τ - μ_i) / σ_i

    引数:
        window_df: 過去L日分のリターン（行=日付、列=銘柄）

    返り値:
        z:     標準化済みリターン
        mu:    各銘柄の平均
        sigma: 各銘柄の標準偏差
    """
    mu    = window_df.mean()
    sigma = window_df.std()
    sigma = sigma.replace(0, np.nan)   # 0除算防止
    z = (window_df - mu) / sigma
    return z, mu, sigma


# ============================================================
# 3. PCA PLAIN シグナルの計算関数
# ============================================================

def calc_pca_plain_signal(
    window_us: pd.DataFrame,   # 過去L日の米国CC リターン
    window_jp: pd.DataFrame,   # 過去L日の日本CC リターン
    z_us_today: pd.Series,     # 当日の米国標準化リターン
    k: int = K,
) -> pd.Series:
    """
    正則化なしPCAで日本業種のシグナルを計算する

    手順:
      1. 日米28業種を縦に並べた相関行列を作る
      2. 上位K個の固有ベクトルを取り出す
      3. 米国ブロック・日本ブロックに分割
      4. 米国の当日ショックをファクタースコアに変換
      5. 日本側ローディングで復元 → シグナル完成

    論文式(13)〜(19)より λ=0 の場合
    """
    # ── Step1: 日米を結合して標準化 ─────────────────────
    combined = pd.concat([window_us, window_jp], axis=1).dropna()
    if len(combined) < k + 1:
        return pd.Series(np.nan, index=window_jp.columns)

    z_combined, _, _ = standardize(combined)

    # ── Step2: 相関行列を計算してPCA ────────────────────
    # 相関行列 = 標準化済みリターンの共分散行列（÷ サンプル数）
    corr = z_combined.T @ z_combined / len(z_combined)

    # 固有値分解（eigenvalues=固有値, eigenvectors=固有ベクトル）
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # 大きい順に並べ替えて上位K個を取り出す
    idx = np.argsort(eigenvalues)[::-1]
    V = eigenvectors[:, idx[:k]]   # shape: (N_US+N_JP, K)

    # ── Step3: 米国・日本ブロックに分割 ─────────────────
    V_us = V[:N_US, :]    # 米国ブロック: (N_US, K)
    V_jp = V[N_US:, :]    # 日本ブロック: (N_JP, K)

    # ── Step4: 当日米国ショックをファクタースコアに変換 ──
    # z_us_today を標準化（ウィンドウの平均・標準偏差を使う）
    _, mu_us, sigma_us = standardize(window_us)
    sigma_us = sigma_us.replace(0, np.nan)
    z_today = (z_us_today - mu_us) / sigma_us
    z_today = z_today.fillna(0)

    # ファクタースコア: ft = V_us^T × z_us_today
    # （米国28次元 → K次元に圧縮）
    ft = V_us.T @ z_today.values   # shape: (K,)

    # ── Step5: 日本側ローディングで復元 → シグナル ───────
    # z_hat_jp = V_jp × ft
    # （K次元 → 日本17次元に復元）
    signal = V_jp @ ft   # shape: (N_JP,)

    return pd.Series(signal, index=window_jp.columns)


# ============================================================
# 4. パフォーマンス指標の計算関数（03から流用）
# ============================================================

def calc_performance(strategy_returns: pd.Series, name: str) -> dict:
    r = strategy_returns.dropna()
    ar   = r.mean() * 252
    risk = r.std()  * np.sqrt(252)
    rr   = ar / risk if risk > 0 else 0.0
    cumulative = (1 + r).cumprod()
    peak       = cumulative.cummax()
    drawdown   = (cumulative / peak) - 1
    mdd        = drawdown.min()
    return {
        "Strategy": name,
        "AR (%)":   round(ar   * 100, 2),
        "RISK (%)": round(risk * 100, 2),
        "R/R":      round(rr,          2),
        "MDD (%)":  round(mdd  * 100, 2),
    }, cumulative


def calc_longshort(signal: pd.Series, next_returns: pd.Series, q: float = Q) -> float:
    valid = signal.dropna()
    if len(valid) == 0:
        return np.nan
    upper = valid.quantile(1 - q)
    lower = valid.quantile(q)
    long_stocks  = valid[valid >= upper].index
    short_stocks = valid[valid <= lower].index
    if len(long_stocks) == 0 or len(short_stocks) == 0:
        return np.nan
    return next_returns[long_stocks].mean() - next_returns[short_stocks].mean()


# ============================================================
# 5. PCA PLAIN 戦略の実行
# ============================================================

print("\n=== PCA PLAIN 戦略バックテスト ===")
print(f"ウィンドウ: {L}営業日, 主成分数: {K}, ロング/ショート: 上下{int(Q*100)}%")

pca_plain_returns = []
dates = []

for i in range(L, len(common_dates) - 1):
    today    = common_dates[i]
    tomorrow = common_dates[i + 1]

    # ── ウィンドウデータ ─────────────────────────────────
    window_us = us_cc.iloc[i - L : i]   # 過去L日の米国CC
    window_jp = jp_cc.iloc[i - L : i]   # 過去L日の日本CC

    # ── 当日の米国リターン（シグナル材料）────────────────
    z_us_today = us_cc.loc[today]

    # ── シグナル計算 ──────────────────────────────────────
    signal = calc_pca_plain_signal(window_us, window_jp, z_us_today)

    # ── 翌日の日本リターンで評価 ──────────────────────────
    next_ret = jp_oc.loc[tomorrow]

    ret = calc_longshort(signal, next_ret)
    pca_plain_returns.append(ret)
    dates.append(tomorrow)

    # 進捗表示
    if i % 500 == 0:
        print(f"  進捗: {i}/{len(common_dates)} 日完了...")

pca_plain_series = pd.Series(pca_plain_returns, index=dates, name="PCA_PLAIN")
pca_plain_series = pca_plain_series.dropna()

# ============================================================
# 6. 結果表示
# ============================================================

perf, cumret = calc_performance(pca_plain_series, "PCA_PLAIN")

print("\n=== PCA PLAIN パフォーマンス ===")
print(f"{'指標':<15} {'結果':>10}  {'論文値':>10}")
print("-" * 40)
print(f"{'年率リターン':<15} {perf['AR (%)']:>9.2f}%  {'6.24%':>10}")
print(f"{'年率リスク':<15} {perf['RISK (%)']:>9.2f}%  {'9.94%':>10}")
print(f"{'R/R':<15} {perf['R/R']:>10.2f}  {'0.62':>10}")
print(f"{'最大ドローダウン':<13} {perf['MDD (%)']:>9.2f}%  {'-23.65%':>10}")

# 保存
pca_plain_series.to_csv(DATA_DIR / "strategy_pca_plain.csv")
print(f"\n💾 data/strategy_pca_plain.csv に保存しました")
print("\n次のステップ: 05_backtest_pca_sub.py で論文のメイン手法を実装します")