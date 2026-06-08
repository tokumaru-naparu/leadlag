"""
15_analyze_signal_asymmetry.py
D: シグナルの非対称性

分析内容:
  ① ロング側シグナル強度 vs ショート側シグナル強度
     → どちらが強い日の方が勝率が高いか
  ② シグナルの偏り（全17業種のシグナルの分布）
     → 全体がプラス方向の日 vs マイナス方向の日
  ③ シグナルの自己相関
     → 昨日強かったシグナルは今日も強いか
  ④ ロング側のみ強い日・ショート側のみ強い日の特性
  ⑤ シグナルの安定性（シグナルが急変した日の勝率）
"""

import pandas as pd
import numpy as np
from pathlib import Path

HISTORY_DIR = Path(__file__).parent / "data" / "history"

trades  = pd.read_csv(HISTORY_DIR / "trades.csv",  index_col=0, parse_dates=True)
signals = pd.read_csv(HISTORY_DIR / "signals.csv", index_col=0, parse_dates=True)

JP_KEYS = [
    "food", "energy", "construction", "materials", "pharma",
    "auto", "steel", "machinery", "electronics", "it_services",
    "utilities", "transport", "trading", "retail", "banks",
    "finance", "realestate"
]

sig_cols = [f"signal_{k}" for k in JP_KEYS]

# tradesとsignalsを結合
signal_fields = sig_cols + ["signal_strength", "signal_spread"]
df = trades.copy()
for col in signal_fields:
    if col not in signals.columns:
        continue
    if col not in df.columns:
        df[col] = signals[col].reindex(df.index)
    else:
        df[col] = df[col].fillna(signals[col].reindex(df.index))

print("=" * 65)
print("  シグナル非対称性レポート（10年分・2341日）")
print("=" * 65)

# ============================================================
# ① ロング側 vs ショート側のシグナル強度
# ============================================================

print("\n【① ロング側・ショート側シグナル強度の比較】")

# ロング上位3業種のシグナル平均
def get_long_signal_avg(row):
    longs = [row[f"long_{i}"] for i in [1,2,3]
             if row.get(f"long_{i}","") and not pd.isna(row.get(f"long_{i}",""))]
    vals = [row.get(f"signal_{l}", np.nan) for l in longs]
    vals = [v for v in vals if not pd.isna(v)]
    return np.mean(vals) if vals else np.nan

def get_short_signal_avg(row):
    shorts = [row[f"short_{i}"] for i in [1,2,3]
              if row.get(f"short_{i}","") and not pd.isna(row.get(f"short_{i}",""))]
    vals = [row.get(f"signal_{s}", np.nan) for s in shorts]
    vals = [v for v in vals if not pd.isna(v)]
    return np.mean(vals) if vals else np.nan

df["long_sig_avg"]  = df.apply(get_long_signal_avg,  axis=1)
df["short_sig_avg"] = df.apply(get_short_signal_avg, axis=1)
df["long_short_diff"] = df["long_sig_avg"] - df["short_sig_avg"]  # ロング-ショートの差

# 差の大きさで分類
df["ls_diff_bin"] = pd.qcut(df["long_short_diff"].dropna(), q=5,
                              labels=["Q1(差小)", "Q2", "Q3", "Q4", "Q5(差大)"])

print(f"\n（ロング側シグナル平均 - ショート側シグナル平均）")
print(f"{'差の大きさ':<12} {'勝率':>8} {'件数':>6} {'平均R':>10} {'差の範囲':>20}")
print("-" * 60)

for label, group in df.groupby("ls_diff_bin", observed=True):
    wr    = group["is_correct"].mean() * 100
    n     = len(group)
    avg_r = group["strategy_return"].mean() * 100
    d_min = group["long_short_diff"].min()
    d_max = group["long_short_diff"].max()
    print(f"{str(label):<12} {wr:>7.1f}% {n:>6} "
          f"{avg_r:>+9.3f}%  [{d_min:.4f} 〜 {d_max:.4f}]")

# ============================================================
# ② シグナルの全体的な方向性（市場バイアス）
# ============================================================

print("\n【② シグナルの全体バイアス（17業種の平均シグナル）】")
print("（全業種が上向きの日 vs 下向きの日）")

df["signal_bias"] = df[sig_cols].mean(axis=1)  # 17業種シグナルの平均

df["bias_bin"] = pd.cut(
    df["signal_bias"],
    bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
    labels=["強い下向き(<-0.02)", "弱い下向き(-0.02〜-0.005)",
            "中立(±0.005)", "弱い上向き(+0.005〜+0.02)", "強い上向き(>+0.02)"]
)

print(f"\n{'市場バイアス':<24} {'勝率':>8} {'件数':>6} {'平均R':>10}")
print("-" * 55)

for label, group in df.groupby("bias_bin", observed=True):
    wr    = group["is_correct"].mean() * 100
    n     = len(group)
    avg_r = group["strategy_return"].mean() * 100
    print(f"{str(label):<24} {wr:>7.1f}% {n:>6} {avg_r:>+9.3f}%")

# ============================================================
# ③ シグナルの自己相関（昨日のシグナルと今日の勝敗）
# ============================================================

print("\n【③ 前日シグナル強度と本日パフォーマンスの関係】")
print("（昨日シグナルが強かった翌日はどうなるか）")

df["prev_strength"] = df["signal_strength"].shift(1)
df["prev_bin"] = pd.qcut(df["prev_strength"].dropna(), q=4,
                          labels=["Q1(前日弱)", "Q2", "Q3", "Q4(前日強)"])

print(f"\n{'前日シグナル強度':<16} {'勝率':>8} {'件数':>6} {'平均R':>10}")
print("-" * 45)

for label, group in df.groupby("prev_bin", observed=True):
    wr    = group["is_correct"].mean() * 100
    n     = len(group)
    avg_r = group["strategy_return"].mean() * 100
    print(f"{str(label):<16} {wr:>7.1f}% {n:>6} {avg_r:>+9.3f}%")

# ============================================================
# ④ シグナルの急変日（前日比で大きく変化した日）
# ============================================================

print("\n【④ シグナル強度の変化量と翌日パフォーマンス】")
print("（シグナルが急に強くなった日 / 急に弱くなった日）")

df["strength_change"] = df["signal_strength"].diff()

df["change_bin"] = pd.qcut(df["strength_change"].dropna(), q=5,
                            labels=["Q1(急低下)", "Q2", "Q3", "Q4", "Q5(急上昇)"])

print(f"\n{'シグナル変化':<14} {'勝率':>8} {'件数':>6} {'平均R':>10} {'変化範囲':>20}")
print("-" * 60)

for label, group in df.groupby("change_bin", observed=True):
    wr    = group["is_correct"].mean() * 100
    n     = len(group)
    avg_r = group["strategy_return"].mean() * 100
    c_min = group["strength_change"].min()
    c_max = group["strength_change"].max()
    print(f"{str(label):<14} {wr:>7.1f}% {n:>6} "
          f"{avg_r:>+9.3f}%  [{c_min:.4f} 〜 {c_max:.4f}]")

# ============================================================
# ⑤ ロング側・ショート側それぞれの強さで分類
# ============================================================

print("\n【⑤ ロング側シグナル強度 × ショート側シグナル強度】")
print("（2×2マトリクス）")

df["long_strong"]  = df["long_sig_avg"]  >= df["long_sig_avg"].median()
df["short_strong"] = df["short_sig_avg"].abs() >= df["short_sig_avg"].abs().median()

combos = {
    (True,  True):  "ロング強×ショート強",
    (True,  False): "ロング強×ショート弱",
    (False, True):  "ロング弱×ショート強",
    (False, False): "ロング弱×ショート弱",
}

print(f"\n{'組み合わせ':<22} {'勝率':>8} {'件数':>6} {'平均R':>10} {'累積R':>10}")
print("-" * 60)

for (ls, ss), label in combos.items():
    group = df[(df["long_strong"]==ls) & (df["short_strong"]==ss)]
    if len(group) == 0:
        continue
    wr    = group["is_correct"].mean() * 100
    n     = len(group)
    avg_r = group["strategy_return"].mean() * 100
    cum_r = (1 + group["strategy_return"]).prod() - 1
    print(f"{label:<22} {wr:>7.1f}% {n:>6} "
          f"{avg_r:>+9.3f}%  {cum_r*100:>+9.2f}%")

# ============================================================
# ⑥ シグナルの集中度（1業種に偏っているか分散しているか）
# ============================================================

print("\n【⑥ シグナルの集中度（上位3業種への集中具合）】")
print("（シグナルが1業種に集中 vs 均等に分散）")

# 上位3業種シグナルの合計 / 全17業種シグナル絶対値の合計
def calc_concentration(row):
    vals = [abs(row.get(f"signal_{k}", 0)) for k in JP_KEYS]
    vals = [v for v in vals if not pd.isna(v)]
    total = sum(vals)
    if total == 0:
        return np.nan
    top3 = sorted(vals, reverse=True)[:3]
    return sum(top3) / total

df["concentration"] = df.apply(calc_concentration, axis=1)

df["conc_bin"] = pd.qcut(df["concentration"].dropna(), q=4,
                          labels=["Q1(分散)", "Q2", "Q3", "Q4(集中)"])

print(f"\n{'集中度':<12} {'勝率':>8} {'件数':>6} {'平均R':>10} {'集中度範囲':>18}")
print("-" * 57)

for label, group in df.groupby("conc_bin", observed=True):
    wr    = group["is_correct"].mean() * 100
    n     = len(group)
    avg_r = group["strategy_return"].mean() * 100
    c_min = group["concentration"].min()
    c_max = group["concentration"].max()
    print(f"{str(label):<12} {wr:>7.1f}% {n:>6} "
          f"{avg_r:>+9.3f}%  [{c_min:.3f} 〜 {c_max:.3f}]")

print("\n" + "=" * 65)
print("以上をチャットに貼り付けてAI分析を依頼してください（D担当）")
print("=" * 65)