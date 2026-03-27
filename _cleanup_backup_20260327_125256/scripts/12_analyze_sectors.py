"""
12_analyze_sectors.py
A: 業種別詳細統計

分析内容:
  ① 業種別ロング勝率ランキング
  ② 業種別ショート勝率ランキング
  ③ 「この業種はロングに向かない・ショートに向かない」の特定
  ④ 業種別リターン分布
  ⑤ シグナル強度×業種の組み合わせ
"""

import pandas as pd
import numpy as np
from pathlib import Path

HISTORY_DIR = Path(__file__).parent / "data" / "history"

trades  = pd.read_csv(HISTORY_DIR / "trades.csv",  index_col=0, parse_dates=True)
signals = pd.read_csv(HISTORY_DIR / "signals.csv", index_col=0, parse_dates=True)
returns = pd.read_csv(HISTORY_DIR / "returns.csv", index_col=0, parse_dates=True)

# 業種キー一覧
JP_KEYS = [
    "food", "energy", "construction", "materials", "pharma",
    "auto", "steel", "machinery", "electronics", "it_services",
    "utilities", "transport", "trading", "retail", "banks",
    "finance", "realestate"
]

JP_NAMES_JP = {
    "food": "食品",           "energy": "エネルギー",
    "construction": "建設",   "materials": "素材・化学",
    "pharma": "医薬品",       "auto": "自動車",
    "steel": "鉄鋼・非鉄",    "machinery": "機械",
    "electronics": "電機・精密","it_services": "情報通信",
    "utilities": "電力・ガス", "transport": "運輸・物流",
    "trading": "商社・卸売",   "retail": "小売",
    "banks": "銀行",          "finance": "金融",
    "realestate": "不動産",
}

print("=" * 65)
print("  業種別詳細統計レポート（10年分・2341日）")
print("=" * 65)

# ============================================================
# ① 業種別ロング勝率
# ============================================================

print("\n【① 業種別ロング成績】")
print("（その業種をロングした日の勝率・平均リターン）")
print(f"\n{'業種':<12} {'選択回数':>8} {'割合':>6} {'勝率':>8} "
      f"{'平均R':>10} {'平均勝R':>10} {'平均負R':>10}")
print("-" * 70)

long_stats = {k: {"wins":0,"total":0,"rets":[]} for k in JP_KEYS}
for _, row in trades.iterrows():
    for col in ["long_1","long_2","long_3"]:
        s = row[col]
        if s and not pd.isna(s) and s in long_stats:
            long_stats[s]["total"] += 1
            long_stats[s]["wins"]  += row["is_correct"]
            long_stats[s]["rets"].append(row["strategy_return"])

total_days = len(trades)
sorted_long = sorted(long_stats.items(),
                     key=lambda x: x[1]["wins"]/x[1]["total"] if x[1]["total"]>0 else 0,
                     reverse=True)

for k, v in sorted_long:
    if v["total"] == 0:
        continue
    wr     = v["wins"] / v["total"] * 100
    ratio  = v["total"] / total_days * 100
    avg_r  = np.mean(v["rets"]) * 100
    wins_r = [r for r in v["rets"] if r > 0]
    loss_r = [r for r in v["rets"] if r <= 0]
    avg_w  = np.mean(wins_r) * 100 if wins_r else 0
    avg_l  = np.mean(loss_r) * 100 if loss_r else 0
    name   = JP_NAMES_JP.get(k, k)
    print(f"{name:<12} {v['total']:>8} {ratio:>5.1f}%  "
          f"{wr:>7.1f}%  {avg_r:>+9.3f}%  {avg_w:>+9.3f}%  {avg_l:>+9.3f}%")

# ============================================================
# ② 業種別ショート成績
# ============================================================

print("\n【② 業種別ショート成績】")
print("（その業種をショートした日の勝率・平均リターン）")
print(f"\n{'業種':<12} {'選択回数':>8} {'割合':>6} {'勝率':>8} "
      f"{'平均R':>10} {'平均勝R':>10} {'平均負R':>10}")
print("-" * 70)

short_stats = {k: {"wins":0,"total":0,"rets":[]} for k in JP_KEYS}
for _, row in trades.iterrows():
    for col in ["short_1","short_2","short_3"]:
        s = row[col]
        if s and not pd.isna(s) and s in short_stats:
            short_stats[s]["total"] += 1
            short_stats[s]["wins"]  += row["is_correct"]
            short_stats[s]["rets"].append(row["strategy_return"])

sorted_short = sorted(short_stats.items(),
                      key=lambda x: x[1]["wins"]/x[1]["total"] if x[1]["total"]>0 else 0,
                      reverse=True)

for k, v in sorted_short:
    if v["total"] == 0:
        continue
    wr     = v["wins"] / v["total"] * 100
    ratio  = v["total"] / total_days * 100
    avg_r  = np.mean(v["rets"]) * 100
    wins_r = [r for r in v["rets"] if r > 0]
    loss_r = [r for r in v["rets"] if r <= 0]
    avg_w  = np.mean(wins_r) * 100 if wins_r else 0
    avg_l  = np.mean(loss_r) * 100 if loss_r else 0
    name   = JP_NAMES_JP.get(k, k)
    print(f"{name:<12} {v['total']:>8} {ratio:>5.1f}%  "
          f"{wr:>7.1f}%  {avg_r:>+9.3f}%  {avg_w:>+9.3f}%  {avg_l:>+9.3f}%")

# ============================================================
# ③ 要注意業種の特定
# ============================================================

print("\n【③ 要注意業種まとめ】")
print("（ロング・ショートそれぞれ勝率55%以上 / 50%未満の業種）")

print("\n▶ ロングで強い業種（勝率55%以上）:")
for k, v in sorted_long:
    if v["total"] < 50: continue
    wr = v["wins"] / v["total"] * 100
    if wr >= 55:
        print(f"  {JP_NAMES_JP.get(k,k):<12} 勝率{wr:.1f}%  {v['total']}日")

print("\n▶ ロングで弱い業種（勝率50%未満）:")
for k, v in sorted(sorted_long, key=lambda x: x[1]["wins"]/x[1]["total"] if x[1]["total"]>0 else 1):
    if v["total"] < 50: continue
    wr = v["wins"] / v["total"] * 100
    if wr < 50:
        print(f"  {JP_NAMES_JP.get(k,k):<12} 勝率{wr:.1f}%  {v['total']}日")

print("\n▶ ショートで強い業種（勝率55%以上）:")
for k, v in sorted_short:
    if v["total"] < 50: continue
    wr = v["wins"] / v["total"] * 100
    if wr >= 55:
        print(f"  {JP_NAMES_JP.get(k,k):<12} 勝率{wr:.1f}%  {v['total']}日")

print("\n▶ ショートで弱い業種（勝率50%未満）:")
for k, v in sorted(sorted_short, key=lambda x: x[1]["wins"]/x[1]["total"] if x[1]["total"]>0 else 1):
    if v["total"] < 50: continue
    wr = v["wins"] / v["total"] * 100
    if wr < 50:
        print(f"  {JP_NAMES_JP.get(k,k):<12} 勝率{wr:.1f}%  {v['total']}日")

# ============================================================
# ④ 業種別シグナル強度×勝率
# ============================================================

print("\n【④ シグナル強度×業種（ロング）】")
print("（Q3以上のシグナルの日だけに絞った場合の業種別勝率）")
print(f"\n{'業種':<12} {'Q3以上勝率':>10} {'Q3以上件数':>10} {'全体勝率':>10}")
print("-" * 50)

sig_col = "signal_strength"
if sig_col not in trades.columns:
    trades = trades.join(signals[[sig_col]], how="left")

q3_thresh = signals[sig_col].quantile(0.4)  # Q3の下限

long_stats_q3 = {k: {"wins":0,"total":0} for k in JP_KEYS}
for _, row in trades.iterrows():
    if row.get(sig_col, 0) < q3_thresh:
        continue
    for col in ["long_1","long_2","long_3"]:
        s = row[col]
        if s and not pd.isna(s) and s in long_stats_q3:
            long_stats_q3[s]["total"] += 1
            long_stats_q3[s]["wins"]  += row["is_correct"]

for k in sorted(JP_KEYS, key=lambda x: long_stats_q3[x]["wins"]/long_stats_q3[x]["total"]
                if long_stats_q3[x]["total"]>0 else 0, reverse=True):
    v    = long_stats_q3[k]
    v_all = long_stats[k]
    if v["total"] < 20: continue
    wr_q3  = v["wins"] / v["total"] * 100
    wr_all = v_all["wins"] / v_all["total"] * 100 if v_all["total"] > 0 else 0
    name   = JP_NAMES_JP.get(k, k)
    diff   = wr_q3 - wr_all
    print(f"{name:<12} {wr_q3:>9.1f}%  {v['total']:>9}  "
          f"{wr_all:>9.1f}%  ({diff:+.1f}%)")

print("\n" + "=" * 65)
print("以上をチャットに貼り付けてAI分析を依頼してください（A担当）")
print("=" * 65)