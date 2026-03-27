"""
13_analyze_market_env.py
B: 市場環境データ（日経平均・ボラティリティ・VIX）

分析内容:
  ① 日経平均のトレンド環境別勝率
     （20日移動平均との乖離率で分類）
  ② 日本市場のボラティリティ別勝率
     （前日の日本市場全体の値動きの大きさ）
  ③ VIX水準別勝率
     （米国恐怖指数）
  ④ 米国市場のボラティリティ別勝率
     （前日の米国全業種の値動きの分散）
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

HISTORY_DIR = Path(__file__).parent / "data" / "history"


def close_series(data: pd.DataFrame) -> pd.Series:
    """Return a single Close series regardless of yfinance column shape."""
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        # Single-ticker downloads can still come back as a 2D frame.
        return close.iloc[:, 0]
    return close

# ============================================================
# 1. 取引データ読み込み
# ============================================================

trades = pd.read_csv(HISTORY_DIR / "trades.csv",  index_col=0, parse_dates=True)
market = pd.read_csv(HISTORY_DIR / "market.csv",  index_col=0, parse_dates=True)
us_cols = [c for c in market.columns if c.startswith("us_cc_")]

start_date = trades.index[0].strftime("%Y-%m-%d")
end_date   = trades.index[-1].strftime("%Y-%m-%d")

# ============================================================
# 2. 追加データ取得
# ============================================================

print("📡 追加データ取得中...")

# 日経平均
nk225 = close_series(
    yf.download("^N225", start=start_date, end=end_date, auto_adjust=True, progress=False)
)

# VIX（米国恐怖指数）
vix = close_series(
    yf.download("^VIX", start=start_date, end=end_date, auto_adjust=True, progress=False)
)

# TOPIX（日本市場全体）
topix = close_series(
    yf.download("1306.T", start=start_date, end=end_date, auto_adjust=True, progress=False)
)

print(f"  日経平均: {len(nk225)}日分")
print(f"  VIX:      {len(vix)}日分")
print(f"  TOPIX:    {len(topix)}日分")

# ============================================================
# 3. 指標計算
# ============================================================

# 日経平均の20日移動平均乖離率
nk225_ma20  = nk225.rolling(20).mean()
nk225_dev   = (nk225 - nk225_ma20) / nk225_ma20* 100  # %乖離
nk225_ret   = nk225.pct_change()  # 日次リターン

# TOPIX前日リターン（日本市場の方向感）
topix_ret   = topix.pct_change()

# 日本市場ボラティリティ（20日間のリターン標準偏差）
topix_vol   = topix_ret.rolling(20).std() * 100

# 米国市場ボラティリティ（前日の全業種リターンの標準偏差）
us_vol_daily = market[us_cols].std(axis=1) * 100  # 業種間の分散

# 通信障害などで外部指標が取れない場合は、既存の米国業種データから近似指標を作る
if len(nk225) == 0 or len(vix) == 0 or len(topix) == 0:
    print("\n[WARN] 外部指数の取得に失敗したため、market.csv の近似指標で代替します")
    us_avg = market[us_cols].mean(axis=1).reindex(trades.index).fillna(0)

    proxy_level = (1 + us_avg).cumprod()
    proxy_ma20  = proxy_level.rolling(20, min_periods=5).mean()
    nk225_dev   = ((proxy_level - proxy_ma20) / proxy_ma20 * 100).replace([np.inf, -np.inf], np.nan)
    nk225_ret   = us_avg

    # VIX近似: 米国業種分散を平常域(15前後)スケールに変換
    vix = (15 + market[us_cols].std(axis=1).reindex(trades.index).fillna(0) * 120).clip(8, 60)

    topix_ret = us_avg
    topix_vol = topix_ret.rolling(20, min_periods=5).std() * 100

# ============================================================
# 4. tradesに結合
# ============================================================

df = trades.copy()
df = df.join(nk225_dev.rename("nk225_dev20"),   how="left")
df = df.join(nk225_ret.rename("nk225_ret"),      how="left")
df = df.join(topix_ret.rename("topix_ret"),      how="left")
df = df.join(topix_vol.rename("topix_vol20"),    how="left")
df = df.join(vix.rename("vix"),                  how="left")
df = df.join(us_vol_daily.rename("us_vol_daily"), how="left")

# 前日値にシフト（当日の戦略に使える情報は前日まで）
df["vix_prev"]       = df["vix"].shift(1)
df["topix_vol_prev"] = df["topix_vol20"].shift(1)
df["nk225_dev_prev"] = df["nk225_dev20"].shift(1)

df = df.dropna(subset=["vix_prev","topix_vol_prev","nk225_dev_prev"])

print(f"\n分析対象: {len(df)} 日")

print("\n" + "=" * 65)
print("  市場環境別パフォーマンスレポート（10年分）")
print("=" * 65)

# ============================================================
# ① 日経平均トレンド環境別勝率
# ============================================================

print("\n【① 日経平均トレンド環境別パフォーマンス】")
print("（20日移動平均からの乖離率で相場環境を分類）")

df["nk_trend"] = pd.cut(
    df["nk225_dev_prev"],
    bins=[-np.inf, -5, -2, 2, 5, np.inf],
    labels=["大幅下落圏(<-5%)", "下落圏(-5〜-2%)",
            "中立圏(±2%)", "上昇圏(+2〜+5%)", "大幅上昇圏(>+5%)"]
)

print(f"\n{'トレンド環境':<20} {'勝率':>8} {'件数':>6} {'平均R':>10} {'累積R':>10}")
print("-" * 60)

for label, group in df.groupby("nk_trend", observed=True):
    wr    = group["is_correct"].mean() * 100
    n     = len(group)
    avg_r = group["strategy_return"].mean() * 100
    cum_r = (1 + group["strategy_return"]).prod() - 1
    print(f"{str(label):<20} {wr:>7.1f}% {n:>6} "
          f"{avg_r:>+9.3f}%  {cum_r*100:>+9.2f}%")

# ============================================================
# ② VIX水準別勝率
# ============================================================

print("\n【② VIX水準別パフォーマンス】")
print("（VIX = 米国恐怖指数。高いほど市場が荒れている）")
print("（VIX 20以下=平静, 20-30=警戒, 30以上=パニック）")

df["vix_level"] = pd.cut(
    df["vix_prev"],
    bins=[0, 15, 20, 25, 30, np.inf],
    labels=["<15(超平静)", "15-20(平静)",
            "20-25(警戒)", "25-30(高警戒)", ">30(パニック)"]
)

print(f"\n{'VIX水準':<18} {'勝率':>8} {'件数':>6} {'平均R':>10} {'累積R':>10}")
print("-" * 58)

for label, group in df.groupby("vix_level", observed=True):
    wr    = group["is_correct"].mean() * 100
    n     = len(group)
    avg_r = group["strategy_return"].mean() * 100
    cum_r = (1 + group["strategy_return"]).prod() - 1
    print(f"{str(label):<18} {wr:>7.1f}% {n:>6} "
          f"{avg_r:>+9.3f}%  {cum_r*100:>+9.2f}%")

# ============================================================
# ③ 日本市場ボラティリティ別勝率
# ============================================================

print("\n【③ 日本市場ボラティリティ別パフォーマンス】")
print("（TOPIX の20日間ボラティリティ）")

df["jp_vol_level"] = pd.qcut(
    df["topix_vol_prev"], q=4,
    labels=["Q1(低ボラ)", "Q2", "Q3", "Q4(高ボラ)"]
)

print(f"\n{'ボラティリティ':<14} {'勝率':>8} {'件数':>6} {'平均R':>10} {'累積R':>10} {'ボラ範囲':>16}")
print("-" * 65)

for label, group in df.groupby("jp_vol_level", observed=True):
    wr     = group["is_correct"].mean() * 100
    n      = len(group)
    avg_r  = group["strategy_return"].mean() * 100
    cum_r  = (1 + group["strategy_return"]).prod() - 1
    v_min  = group["topix_vol_prev"].min()
    v_max  = group["topix_vol_prev"].max()
    print(f"{str(label):<14} {wr:>7.1f}% {n:>6} "
          f"{avg_r:>+9.3f}%  {cum_r*100:>+9.2f}%  "
          f"[{v_min:.2f}〜{v_max:.2f}%]")

# ============================================================
# ④ 米国業種間ボラティリティ別勝率
# ============================================================

print("\n【④ 米国業種間ボラティリティ別パフォーマンス】")
print("（前日の米国11業種リターンの標準偏差。業種間の分散が大きいほどリードラグが明確）")

df["us_vol_level"] = pd.qcut(
    df["us_vol_daily"], q=4,
    labels=["Q1(低分散)", "Q2", "Q3", "Q4(高分散)"]
)

print(f"\n{'米国業種分散':<14} {'勝率':>8} {'件数':>6} {'平均R':>10} {'累積R':>10} {'分散範囲':>16}")
print("-" * 65)

for label, group in df.groupby("us_vol_level", observed=True):
    wr    = group["is_correct"].mean() * 100
    n     = len(group)
    avg_r = group["strategy_return"].mean() * 100
    cum_r = (1 + group["strategy_return"]).prod() - 1
    v_min = group["us_vol_daily"].min()
    v_max = group["us_vol_daily"].max()
    print(f"{str(label):<14} {wr:>7.1f}% {n:>6} "
          f"{avg_r:>+9.3f}%  {cum_r*100:>+9.2f}%  "
          f"[{v_min:.2f}〜{v_max:.2f}%]")

# ============================================================
# ⑤ VIX × シグナル強度の組み合わせ
# ============================================================

print("\n【⑤ VIX × シグナル強度の組み合わせ】")
print("（最も重要な2変数の掛け合わせ）")

trades_sig = pd.read_csv(HISTORY_DIR / "signals.csv",
                         index_col=0, parse_dates=True)

if "signal_strength" not in df.columns:
    df = df.join(trades_sig[["signal_strength"]], how="left")
else:
    df["signal_strength"] = df["signal_strength"].fillna(
        trades_sig["signal_strength"].reindex(df.index)
    )

df["vix_high"]    = df["vix_prev"] >= 20
df["sig_strong"]  = df["signal_strength"] >= 0.152  # Q3以上

combo_labels = {
    (False, False): "低VIX×弱シグナル",
    (False, True):  "低VIX×強シグナル",
    (True,  False): "高VIX×弱シグナル",
    (True,  True):  "高VIX×強シグナル",
}

print(f"\n{'組み合わせ':<20} {'勝率':>8} {'件数':>6} {'平均R':>10} {'累積R':>10}")
print("-" * 60)

for (vix_h, sig_s), label in combo_labels.items():
    group = df[(df["vix_high"]==vix_h) & (df["sig_strong"]==sig_s)]
    if len(group) == 0:
        continue
    wr    = group["is_correct"].mean() * 100
    n     = len(group)
    avg_r = group["strategy_return"].mean() * 100
    cum_r = (1 + group["strategy_return"]).prod() - 1
    print(f"{label:<20} {wr:>7.1f}% {n:>6} "
          f"{avg_r:>+9.3f}%  {cum_r*100:>+9.2f}%")

print("\n" + "=" * 65)
print("以上をチャットに貼り付けてAI分析を依頼してください（B担当）")
print("=" * 65)