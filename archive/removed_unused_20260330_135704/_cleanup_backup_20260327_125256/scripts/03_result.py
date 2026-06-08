"""
03_backtest.py
MOM（単純モメンタム）戦略のバックテスト

論文のベースライン戦略その1:
  過去60営業日の日本業種ETFの平均Close-to-Closeリターンをシグナルにする
  上位30% → ロング（買い）
  下位30% → ショート（空売り）
  翌日のOpen-to-Closeリターンで損益を評価する
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ============================================================
# パラメータ（論文と同じ値）
# ============================================================

L = 60       # ローリングウィンドウ: 60営業日
Q = 0.3      # 上位・下位 30% をロング・ショート

# ============================================================
# 1. データ読み込み
# ============================================================

print("=== データ読み込み ===")
jp_oc = pd.read_csv(DATA_DIR / "jp_oc_returns.csv", index_col=0, parse_dates=True)
jp_cc = pd.read_csv(DATA_DIR / "us_cc_returns.csv", index_col=0, parse_dates=True)
# ※ MOMは日本のClose-to-Closeリターンが必要なので別途計算

# 日本のClose-to-Closeリターンを作る
us_raw = pd.read_csv(DATA_DIR / "us_etf.csv",
                     header=[0, 1], index_col=0, parse_dates=True)
jp_raw = pd.read_csv(DATA_DIR / "jp_etf.csv",
                     header=[0, 1], index_col=0, parse_dates=True)

jp_close = jp_raw["Close"]
jp_cc_returns = jp_close.pct_change()   # 日本のClose-to-Close

# 共通営業日に絞る
common_dates = jp_oc.index.intersection(jp_cc_returns.index)
jp_oc = jp_oc.loc[common_dates]
jp_cc_returns = jp_cc_returns.loc[common_dates]

print(f"共通営業日数: {len(common_dates)}")
print(f"期間: {common_dates[0].date()} 〜 {common_dates[-1].date()}")

# ============================================================
# 2. パフォーマンス指標の計算関数
# ============================================================

def calc_performance(strategy_returns: pd.Series, name: str) -> dict:
    """
    年率リターン・リスク・R/R・最大ドローダウンを計算する

    引数:
        strategy_returns: 毎日の戦略リターン（小数。0.01 = 1%）
        name: 戦略名（表示用）
    """
    r = strategy_returns.dropna()

    # 年率リターン（AR）: 日次平均 × 252営業日
    ar = r.mean() * 252

    # 年率リスク（RISK）: 日次標準偏差 × √252
    risk = r.std() * np.sqrt(252)

    # リスク・リターン比（R/R）
    rr = ar / risk if risk > 0 else 0.0

    # 最大ドローダウン（MDD）: ピークからの最大下落率
    cumulative = (1 + r).cumprod()          # 累積リターン（1が元本）
    peak = cumulative.cummax()              # その時点までの最高値
    drawdown = (cumulative / peak) - 1      # 現在値 / ピーク - 1（マイナスが損失）
    mdd = drawdown.min()                    # 最も深い谷

    result = {
        "Strategy": name,
        "AR (%)":   round(ar   * 100, 2),
        "RISK (%)": round(risk * 100, 2),
        "R/R":      round(rr,          2),
        "MDD (%)":  round(mdd  * 100, 2),
    }

    return result, cumulative


# ============================================================
# 3. ロングショートポートフォリオの計算関数
# ============================================================

def calc_longshort(signal: pd.Series, next_returns: pd.Series, q: float = Q) -> float:
    """
    シグナルに基づいてロングショートポートフォリオのリターンを計算する

    引数:
        signal: 各業種のシグナル値（大きいほど有望）
        next_returns: 翌日の実際のリターン
        q: 上位・下位何%をロング・ショートにするか（デフォルト30%）

    返り値:
        その日の戦略リターン（小数）
    """
    valid = signal.dropna()
    if len(valid) == 0:
        return np.nan

    # 閾値を計算
    upper = valid.quantile(1 - q)   # 上位30% の閾値
    lower = valid.quantile(q)       # 下位30% の閾値

    # ロング・ショートの銘柄を選ぶ
    long_stocks  = valid[valid >= upper].index
    short_stocks = valid[valid <= lower].index

    if len(long_stocks) == 0 or len(short_stocks) == 0:
        return np.nan

    # 等ウェイト（均等に配分）でリターンを計算
    long_ret  = next_returns[long_stocks].mean()
    short_ret = next_returns[short_stocks].mean()

    # ロングのリターン - ショートのリターン = 戦略リターン
    return long_ret - short_ret


# ============================================================
# 4. MOM 戦略の実行
# ============================================================

print("\n=== MOM 戦略バックテスト ===")
print(f"ウィンドウ: {L}営業日, ロング/ショート比率: 上下{int(Q*100)}%")

mom_returns = []
dates = []

for i in range(L, len(common_dates) - 1):
    # 現在の日付
    today = common_dates[i]

    # ── シグナル計算 ──────────────────────────────────────
    # 過去L日間の日本業種ETFの平均Close-to-Closeリターン
    window = jp_cc_returns.iloc[i - L : i]   # 過去60日分
    signal = window.mean()                    # 各業種の平均リターン

    # ── 翌日のリターン ────────────────────────────────────
    tomorrow = common_dates[i + 1]
    next_ret = jp_oc.loc[tomorrow]            # 翌日のOpen-to-Close

    # ── ロングショートリターン計算 ────────────────────────
    ret = calc_longshort(signal, next_ret)
    mom_returns.append(ret)
    dates.append(tomorrow)

mom_series = pd.Series(mom_returns, index=dates, name="MOM")
mom_series = mom_series.dropna()

# ============================================================
# 5. 結果表示
# ============================================================

perf, cumret = calc_performance(mom_series, "MOM")

print("\n=== MOM パフォーマンス ===")
print(f"{'指標':<15} {'結果':>10}  {'論文値':>10}")
print("-" * 40)
print(f"{'年率リターン':<15} {perf['AR (%)']:>9.2f}%  {'5.63%':>10}")
print(f"{'年率リスク':<15} {perf['RISK (%)']:>9.2f}%  {'10.59%':>10}")
print(f"{'R/R':<15} {perf['R/R']:>10.2f}  {'0.53':>10}")
print(f"{'最大ドローダウン':<13} {perf['MDD (%)']:>9.2f}%  {'-16.97%':>10}")

print("\n論文値（表2）との差が小さければOK！")
print("（完全一致はしないが傾向が合ってれば再現性あり）")

# 結果をCSVに保存（後で他の戦略と比較するため）
mom_series.to_csv(DATA_DIR / "strategy_mom.csv")
print(f"\n💾 data/strategy_mom.csv に保存しました")
print("\n次のステップ: 04_pca_plain.py でPCA PLAINを実装します")