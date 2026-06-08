"""
14_analyze_pairs.py
C: 業種ペア統計

分析内容:
  ① ロング×ショートの業種ペア別勝率
     （「銀行ロング×食品ショート」は当たりやすい等）
  ② 同一業種が両方に選ばれた日の特性
     （食品がロングにもショートにも入る日）
  ③ 勝率が高い業種ペアTOP10・低いペアTOP10
  ④ ペアの安定性（10年間で同じペアが繰り返されるか）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

HISTORY_DIR = Path(__file__).parent / "data" / "history"

trades = pd.read_csv(HISTORY_DIR / "trades.csv", index_col=0, parse_dates=True)

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
print("  業種ペア統計レポート（10年分・2341日）")
print("=" * 65)

# ============================================================
# ① ロング×ショートの業種ペア別勝率
# ============================================================

print("\n【① ロング×ショート業種ペア別勝率（登場10回以上）】")

pair_stats = {}

for _, row in trades.iterrows():
    longs  = [row[f"long_{i}"]  for i in [1,2,3] if row.get(f"long_{i}","")]
    shorts = [row[f"short_{i}"] for i in [1,2,3] if row.get(f"short_{i}","")]

    for l in longs:
        for s in shorts:
            if not l or not s or pd.isna(l) or pd.isna(s):
                continue
            key = (l, s)
            if key not in pair_stats:
                pair_stats[key] = {"wins": 0, "total": 0, "rets": []}
            pair_stats[key]["total"] += 1
            pair_stats[key]["wins"]  += row["is_correct"]
            pair_stats[key]["rets"].append(row["strategy_return"])

# 10回以上のペアだけ
filtered = {k: v for k, v in pair_stats.items() if v["total"] >= 10}

# 勝率順にソート
sorted_pairs = sorted(filtered.items(),
                      key=lambda x: x[1]["wins"]/x[1]["total"],
                      reverse=True)

print(f"\n▶ 勝率TOP15ペア（10回以上）:")
print(f"{'ロング':<12} {'ショート':<12} {'勝率':>8} {'件数':>6} {'平均R':>10} {'期待値':>10}")
print("-" * 62)

for (l, s), v in sorted_pairs[:15]:
    wr    = v["wins"] / v["total"] * 100
    avg_r = np.mean(v["rets"]) * 100
    ev    = (v["wins"]/v["total"] * np.mean([r for r in v["rets"] if r > 0] or [0])
             + (1-v["wins"]/v["total"]) * np.mean([r for r in v["rets"] if r <= 0] or [0])) * 100
    ln = JP_NAMES_JP.get(l, l)
    sn = JP_NAMES_JP.get(s, s)
    print(f"{ln:<12} {sn:<12} {wr:>7.1f}% {v['total']:>6} "
          f"{avg_r:>+9.3f}%  {ev:>+9.3f}%")

print(f"\n▶ 勝率BOTTOM15ペア（10回以上）:")
print(f"{'ロング':<12} {'ショート':<12} {'勝率':>8} {'件数':>6} {'平均R':>10} {'期待値':>10}")
print("-" * 62)

for (l, s), v in sorted_pairs[-15:]:
    wr    = v["wins"] / v["total"] * 100
    avg_r = np.mean(v["rets"]) * 100
    ev    = (v["wins"]/v["total"] * np.mean([r for r in v["rets"] if r > 0] or [0])
             + (1-v["wins"]/v["total"]) * np.mean([r for r in v["rets"] if r <= 0] or [0])) * 100
    ln = JP_NAMES_JP.get(l, l)
    sn = JP_NAMES_JP.get(s, s)
    print(f"{ln:<12} {sn:<12} {wr:>7.1f}% {v['total']:>6} "
          f"{avg_r:>+9.3f}%  {ev:>+9.3f}%")

# ============================================================
# ② 高頻度ペアの詳細（100回以上）
# ============================================================

print("\n【② 高頻度ペア詳細（100回以上登場）】")
print("（十分なサンプルがある信頼性の高いペア）")

high_freq = {k: v for k, v in pair_stats.items() if v["total"] >= 100}
sorted_hf  = sorted(high_freq.items(),
                    key=lambda x: x[1]["wins"]/x[1]["total"],
                    reverse=True)

print(f"\n{'ロング':<12} {'ショート':<12} {'勝率':>8} {'件数':>6} "
      f"{'平均R':>10} {'期待値':>10} {'損益比':>8}")
print("-" * 70)

for (l, s), v in sorted_hf:
    wr    = v["wins"] / v["total"] * 100
    avg_r = np.mean(v["rets"]) * 100
    wins_r = [r for r in v["rets"] if r > 0]
    loss_r = [r for r in v["rets"] if r <= 0]
    avg_w  = np.mean(wins_r) * 100 if wins_r else 0
    avg_l  = np.mean(loss_r) * 100 if loss_r else 0
    ev     = (v["wins"]/v["total"] * (avg_w/100)
              + (1-v["wins"]/v["total"]) * (avg_l/100)) * 100
    pl_ratio = abs(avg_w / avg_l) if avg_l != 0 else 0
    ln = JP_NAMES_JP.get(l, l)
    sn = JP_NAMES_JP.get(s, s)
    print(f"{ln:<12} {sn:<12} {wr:>7.1f}% {v['total']:>6} "
          f"{avg_r:>+9.3f}%  {ev:>+9.3f}%  {pl_ratio:>7.2f}x")

# ============================================================
# ③ 同一業種がロング・ショート両方に選ばれた日
# ============================================================

print("\n【③ 同一業種がロング・ショート両方に選ばれた日】")
print("（シグナルが混乱している可能性がある日）")

overlap_days = []
for _, row in trades.iterrows():
    longs  = set([row[f"long_{i}"]  for i in [1,2,3]
                  if row.get(f"long_{i}","") and not pd.isna(row.get(f"long_{i}",""))])
    shorts = set([row[f"short_{i}"] for i in [1,2,3]
                  if row.get(f"short_{i}","") and not pd.isna(row.get(f"short_{i}",""))])
    overlap = longs & shorts
    if overlap:
        overlap_days.append({
            "date": row.name,
            "overlap": list(overlap),
            "is_correct": row["is_correct"],
            "strategy_return": row["strategy_return"]
        })

overlap_df = pd.DataFrame(overlap_days)
if len(overlap_df) > 0:
    print(f"\n  重複発生日数: {len(overlap_df)}日 / {len(trades)}日中 "
          f"({len(overlap_df)/len(trades)*100:.1f}%)")
    print(f"  重複日の勝率: {overlap_df['is_correct'].mean()*100:.1f}%")
    print(f"  重複日の平均R: {overlap_df['strategy_return'].mean()*100:+.3f}%")
    print(f"  非重複日の勝率: "
          f"{trades[~trades.index.isin(overlap_df['date'])]['is_correct'].mean()*100:.1f}%")
else:
    print("  重複なし")

# ============================================================
# ④ ロング業種の組み合わせパターン
# ============================================================

print("\n【④ ロング3業種の組み合わせパターンTOP10】")
print("（毎日選ばれる3業種セットの頻度）")

combo_stats = {}
for _, row in trades.iterrows():
    longs = tuple(sorted([row[f"long_{i}"] for i in [1,2,3]
                          if row.get(f"long_{i}","") and not pd.isna(row.get(f"long_{i}",""))]))
    if len(longs) == 3:
        if longs not in combo_stats:
            combo_stats[longs] = {"wins": 0, "total": 0}
        combo_stats[longs]["total"] += 1
        combo_stats[longs]["wins"]  += row["is_correct"]

sorted_combos = sorted(combo_stats.items(),
                       key=lambda x: x[1]["total"], reverse=True)

print(f"\n{'ロング3業種セット':<40} {'件数':>6} {'勝率':>8}")
print("-" * 58)

for combo, v in sorted_combos[:10]:
    names = " + ".join([JP_NAMES_JP.get(c, c) for c in combo])
    wr    = v["wins"] / v["total"] * 100
    print(f"{names:<40} {v['total']:>6} {wr:>7.1f}%")

print("\n【⑤ ショート3業種の組み合わせパターンTOP10】")

combo_short = {}
for _, row in trades.iterrows():
    shorts = tuple(sorted([row[f"short_{i}"] for i in [1,2,3]
                           if row.get(f"short_{i}","") and not pd.isna(row.get(f"short_{i}",""))]))
    if len(shorts) == 3:
        if shorts not in combo_short:
            combo_short[shorts] = {"wins": 0, "total": 0}
        combo_short[shorts]["total"] += 1
        combo_short[shorts]["wins"]  += row["is_correct"]

sorted_short_combos = sorted(combo_short.items(),
                             key=lambda x: x[1]["total"], reverse=True)

print(f"\n{'ショート3業種セット':<40} {'件数':>6} {'勝率':>8}")
print("-" * 58)

for combo, v in sorted_short_combos[:10]:
    names = " + ".join([JP_NAMES_JP.get(c, c) for c in combo])
    wr    = v["wins"] / v["total"] * 100
    print(f"{names:<40} {v['total']:>6} {wr:>7.1f}%")

print("\n" + "=" * 65)
print("以上をチャットに貼り付けてAI分析を依頼してください（C担当）")
print("=" * 65)