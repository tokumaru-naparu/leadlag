"""
11_analyze_patterns.py
692日分のデータから統計量を計算して
チャットに貼り付けられる形で出力する

出力内容:
  ① 曜日別勝率・平均リターン
  ② シグナル強度別勝率
  ③ 米国市場の方向別勝率
  ④ 月別勝率
  ⑤ 連敗パターン
  ⑥ ロング/ショート業種の頻度ランキング
  ⑦ 損益分布の要約
"""

import pandas as pd
import numpy as np
from pathlib import Path

HISTORY_DIR = Path(__file__).parent / "data" / "history"

# ============================================================
# データ読み込み
# ============================================================

trades  = pd.read_csv(HISTORY_DIR / "trades.csv",  index_col=0, parse_dates=True)
signals = pd.read_csv(HISTORY_DIR / "signals.csv", index_col=0, parse_dates=True)
market  = pd.read_csv(HISTORY_DIR / "market.csv",  index_col=0, parse_dates=True)
perf    = pd.read_csv(HISTORY_DIR / "performance.csv", index_col=0, parse_dates=True)

# 結合
df = trades.join(signals[["signal_strength", "signal_spread"]], how="left", rsuffix="_sig")
df = df.join(market, how="left")

print("=" * 60)
print("  PCA SUB 戦略 パターン分析レポート")
print(f"  期間: {df.index[0].date()} 〜 {df.index[-1].date()}")
print(f"  営業日数: {len(df)} 日")
print(f"  全体勝率: {df['is_correct'].mean()*100:.1f}%")
print(f"  全体平均リターン: {df['strategy_return'].mean()*100:+.3f}%/日")
print("=" * 60)

# ============================================================
# ① 曜日別勝率・平均リターン
# ============================================================

print("\n【① 曜日別パフォーマンス】")
print(f"{'曜日':<6} {'勝率':>8} {'勝/負':>8} {'平均リターン':>12} {'平均勝ちR':>12} {'平均負けR':>12}")
print("-" * 65)

wday_order = ["Mon","Tue","Wed","Thu","Fri"]
wday_ja    = {"Mon":"月曜","Tue":"火曜","Wed":"水曜","Thu":"木曜","Fri":"金曜"}

for w in wday_order:
    sub = df[df["weekday"] == w]
    if len(sub) == 0:
        continue
    wins   = sub["is_correct"].sum()
    total  = len(sub)
    wr     = wins / total * 100
    avg_r  = sub["strategy_return"].mean() * 100
    avg_w  = sub[sub["is_correct"]==1]["strategy_return"].mean() * 100
    avg_l  = sub[sub["is_correct"]==0]["strategy_return"].mean() * 100
    print(f"{wday_ja[w]:<6} {wr:>7.1f}% {wins:>3}/{total:<4} "
          f"{avg_r:>+11.3f}%  {avg_w:>+11.3f}%  {avg_l:>+11.3f}%")

# ============================================================
# ② シグナル強度別勝率
# ============================================================

print("\n【② シグナル強度別パフォーマンス】")
print("（signal_strength = シグナルの上位と下位の差。大きいほど確信度が高い）")

df["strength_bin"] = pd.qcut(df["signal_strength"], q=5,
                              labels=["Q1(弱)","Q2","Q3","Q4","Q5(強)"])

print(f"\n{'強度帯':<10} {'勝率':>8} {'件数':>6} {'平均リターン':>12} {'強度範囲':>20}")
print("-" * 60)

for label, group in df.groupby("strength_bin", observed=True):
    wr    = group["is_correct"].mean() * 100
    n     = len(group)
    avg_r = group["strategy_return"].mean() * 100
    s_min = group["signal_strength"].min()
    s_max = group["signal_strength"].max()
    print(f"{str(label):<10} {wr:>7.1f}% {n:>6} "
          f"{avg_r:>+11.3f}%  [{s_min:.3f} 〜 {s_max:.3f}]")

# ============================================================
# ③ 米国市場の方向別勝率
# ============================================================

print("\n【③ 米国市場の方向別パフォーマンス】")
print("（前日の米国市場全体の動きで分類）")

# 米国全業種の平均リターンで方向を判定
us_cols = [c for c in df.columns if c.startswith("us_cc_")]
if us_cols:
    df["us_market_avg"] = df[us_cols].mean(axis=1)
    df["us_direction"]  = pd.cut(
        df["us_market_avg"],
        bins=[-np.inf, -0.01, -0.003, 0.003, 0.01, np.inf],
        labels=["大幅下落(< -1%)", "小幅下落(-1〜-0.3%)",
                "横ばい(±0.3%)", "小幅上昇(+0.3〜+1%)", "大幅上昇(> +1%)"]
    )

    print(f"\n{'米国方向':<22} {'勝率':>8} {'件数':>6} {'平均リターン':>12}")
    print("-" * 55)

    for label, group in df.groupby("us_direction", observed=True):
        wr    = group["is_correct"].mean() * 100
        n     = len(group)
        avg_r = group["strategy_return"].mean() * 100
        print(f"{str(label):<22} {wr:>7.1f}% {n:>6} {avg_r:>+11.3f}%")

# ============================================================
# ④ 月別勝率
# ============================================================

print("\n【④ 月別パフォーマンス】")
month_ja = {1:"1月",2:"2月",3:"3月",4:"4月",5:"5月",6:"6月",
            7:"7月",8:"8月",9:"9月",10:"10月",11:"11月",12:"12月"}

print(f"\n{'月':<6} {'勝率':>8} {'件数':>6} {'平均リターン':>12} {'累積リターン':>12}")
print("-" * 50)

for month in sorted(df["month"].unique()):
    sub   = df[df["month"] == month]
    wr    = sub["is_correct"].mean() * 100
    n     = len(sub)
    avg_r = sub["strategy_return"].mean() * 100
    cum_r = (1 + sub["strategy_return"]).prod() - 1
    print(f"{month_ja[month]:<6} {wr:>7.1f}% {n:>6} "
          f"{avg_r:>+11.3f}%  {cum_r*100:>+11.2f}%")

# ============================================================
# ⑤ 連敗・連勝パターン
# ============================================================

print("\n【⑤ 連敗・連勝パターン】")

streak = 0
max_win_streak  = 0
max_loss_streak = 0
current_streak  = 0
prev_correct    = None
streaks = []

for _, row in df.iterrows():
    c = row["is_correct"]
    if c == prev_correct:
        current_streak += 1
    else:
        if prev_correct is not None:
            streaks.append((prev_correct, current_streak))
        current_streak = 1
    prev_correct = c

# 連敗後の翌日勝率
loss_streak_next = []
for i in range(len(df) - 1):
    streak_len = 0
    for j in range(i, -1, -1):
        if df.iloc[j]["is_correct"] == 0:
            streak_len += 1
        else:
            break
    if streak_len >= 2:
        loss_streak_next.append(df.iloc[i+1]["is_correct"])

print(f"  2連敗以上の翌日勝率: "
      f"{np.mean(loss_streak_next)*100:.1f}% ({len(loss_streak_next)}件)")

# 連勝後の翌日勝率
win_streak_next = []
for i in range(len(df) - 1):
    streak_len = 0
    for j in range(i, -1, -1):
        if df.iloc[j]["is_correct"] == 1:
            streak_len += 1
        else:
            break
    if streak_len >= 2:
        win_streak_next.append(df.iloc[i+1]["is_correct"])

print(f"  2連勝以上の翌日勝率: "
      f"{np.mean(win_streak_next)*100:.1f}% ({len(win_streak_next)}件)")

# ============================================================
# ⑥ ロング/ショート業種の頻度ランキング
# ============================================================

print("\n【⑥ ロング/ショート業種の頻度と勝率】")

long_cols  = ["long_1",  "long_2",  "long_3"]
short_cols = ["short_1", "short_2", "short_3"]

# ロング業種別勝率
long_stats = {}
for col in long_cols:
    for _, row in df.iterrows():
        sector = row[col]
        if sector == "" or pd.isna(sector):
            continue
        if sector not in long_stats:
            long_stats[sector] = {"wins": 0, "total": 0, "returns": []}
        long_stats[sector]["total"] += 1
        long_stats[sector]["wins"]  += row["is_correct"]
        long_stats[sector]["returns"].append(row["strategy_return"])

print(f"\nロング選択頻度TOP5（全{len(df)}日中）:")
print(f"{'業種':<15} {'選択回数':>8} {'勝率':>8} {'平均R':>10}")
print("-" * 45)

sorted_long = sorted(long_stats.items(),
                     key=lambda x: x[1]["total"], reverse=True)
for sector, stat in sorted_long[:5]:
    wr    = stat["wins"] / stat["total"] * 100
    avg_r = np.mean(stat["returns"]) * 100
    print(f"{sector:<15} {stat['total']:>8} {wr:>7.1f}% {avg_r:>+9.3f}%")

# ショート業種別
short_stats = {}
for col in short_cols:
    for _, row in df.iterrows():
        sector = row[col]
        if sector == "" or pd.isna(sector):
            continue
        if sector not in short_stats:
            short_stats[sector] = {"wins": 0, "total": 0, "returns": []}
        short_stats[sector]["total"] += 1
        short_stats[sector]["wins"]  += row["is_correct"]
        short_stats[sector]["returns"].append(row["strategy_return"])

print(f"\nショート選択頻度TOP5（全{len(df)}日中）:")
print(f"{'業種':<15} {'選択回数':>8} {'勝率':>8} {'平均R':>10}")
print("-" * 45)

sorted_short = sorted(short_stats.items(),
                      key=lambda x: x[1]["total"], reverse=True)
for sector, stat in sorted_short[:5]:
    wr    = stat["wins"] / stat["total"] * 100
    avg_r = np.mean(stat["returns"]) * 100
    print(f"{sector:<15} {stat['total']:>8} {wr:>7.1f}% {avg_r:>+9.3f}%")

# ============================================================
# ⑦ 損益分布の要約
# ============================================================

print("\n【⑦ 損益分布】")
r = df["strategy_return"] * 100

print(f"  最大利益日: {r.max():+.3f}%")
print(f"  最大損失日: {r.min():+.3f}%")
print(f"  平均利益日: {r[r>0].mean():+.3f}%  ({(r>0).sum()}日)")
print(f"  平均損失日: {r[r<0].mean():+.3f}%  ({(r<0).sum()}日)")
print(f"  損益比(勝ち/負け絶対値): {abs(r[r>0].mean()/r[r<0].mean()):.2f}")
print(f"  標準偏差: {r.std():.3f}%")

print(f"\n  リターン分布:")
bins = [-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf]
labels = ["<-2%", "-2〜-1%", "-1〜-0.5%", "-0.5〜0%",
          "0〜+0.5%", "+0.5〜+1%", "+1〜+2%", ">+2%"]
dist = pd.cut(r, bins=bins, labels=labels).value_counts().sort_index()
for label, count in dist.items():
    bar = "█" * (count // 3)
    print(f"  {label:>12}: {count:>4}日  {bar}")

print("\n" + "=" * 60)
print("以上のデータをチャットに貼り付けてAI分析を依頼してください")
print("=" * 60)