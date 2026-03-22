"""
07_summary.py
4戦略の比較表と累積リターングラフを作る

論文 図2・表2 の再現
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR   = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 日本語フォント設定（文字化け防止）
plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 各戦略のリターンを読み込む
# ============================================================

print("=== データ読み込み ===")

strategies = {
    "MOM":       pd.read_csv(DATA_DIR / "strategy_mom.csv",
                             index_col=0, parse_dates=True).squeeze(),
    "PCA PLAIN": pd.read_csv(DATA_DIR / "strategy_pca_plain.csv",
                             index_col=0, parse_dates=True).squeeze(),
    "PCA SUB":   pd.read_csv(DATA_DIR / "strategy_pca_sub.csv",
                             index_col=0, parse_dates=True).squeeze(),
    "DOUBLE":    pd.read_csv(DATA_DIR / "strategy_double.csv",
                             index_col=0, parse_dates=True).squeeze(),
}

for name, s in strategies.items():
    print(f"  {name}: {len(s)} 日分")

# ============================================================
# 2. パフォーマンス指標の計算
# ============================================================

def calc_performance(r: pd.Series, name: str) -> dict:
    r    = r.dropna()
    ar   = r.mean() * 252
    risk = r.std()  * np.sqrt(252)
    rr   = ar / risk if risk > 0 else 0.0
    cum  = (1 + r).cumprod()
    mdd  = ((cum / cum.cummax()) - 1).min()
    return {
        "戦略":         name,
        "年率リターン": f"{ar*100:.2f}%",
        "年率リスク":   f"{risk*100:.2f}%",
        "R/R":         f"{rr:.2f}",
        "最大DD":       f"{mdd*100:.2f}%",
    }

results = [calc_performance(s, name) for name, s in strategies.items()]
df_results = pd.DataFrame(results).set_index("戦略")

# 論文値
paper_values = pd.DataFrame({
    "戦略":         ["MOM", "PCA PLAIN", "PCA SUB", "DOUBLE"],
    "年率リターン": ["5.63%", "6.24%", "23.79%", "18.86%"],
    "年率リスク":   ["10.59%", "9.94%", "10.70%", "11.16%"],
    "R/R":         ["0.53", "0.62", "2.22", "1.69"],
    "最大DD":       ["-16.97%", "-23.65%", "-9.58%", "-12.10%"],
}).set_index("戦略")

print("\n=== 今回の結果 ===")
print(df_results.to_string())

print("\n=== 論文値（参考） ===")
print(paper_values.to_string())

# CSVに保存
df_results.to_csv(OUTPUT_DIR / "performance_summary.csv", encoding="utf-8-sig")
print(f"\n💾 output/performance_summary.csv に保存しました")

# ============================================================
# 3. 累積リターングラフ（論文 図2 の再現）
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# ── グラフ①: 累積リターン ────────────────────────────────
ax1 = axes[0]

colors = {
    "PCA SUB":   "#1f77b4",   # 青（論文と同じ色順）
    "DOUBLE":    "#ff7f0e",   # オレンジ
    "PCA PLAIN": "#2ca02c",   # 緑
    "MOM":       "#d62728",   # 赤
}
linestyles = {
    "PCA SUB":   "-",
    "DOUBLE":    "--",
    "PCA PLAIN": "-.",
    "MOM":       ":",
}

# 全戦略の共通開始日に揃える
all_series = pd.concat(strategies.values(), axis=1, keys=strategies.keys()).dropna()

for name in ["PCA SUB", "DOUBLE", "PCA PLAIN", "MOM"]:
    r   = all_series[name]
    cum = (1 + r).cumprod()
    ax1.plot(cum.index, cum.values,
             label=name,
             color=colors[name],
             linestyle=linestyles[name],
             linewidth=1.5)

ax1.set_title("各戦略の累積リターン（論文 図2 の再現）", fontsize=14)
ax1.set_ylabel("累積リターン（倍）", fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator(2))

# ── グラフ②: 年率リターンとR/Rの比較棒グラフ ─────────────
ax2 = axes[1]

strategy_names = ["MOM", "PCA PLAIN", "PCA SUB", "DOUBLE"]
x = np.arange(len(strategy_names))
width = 0.35

# 今回の値（数値に変換）
ar_values    = [float(df_results.loc[n, "年率リターン"].replace("%",""))
                for n in strategy_names]
ar_paper     = [float(paper_values.loc[n, "年率リターン"].replace("%",""))
                for n in strategy_names]

bars1 = ax2.bar(x - width/2, ar_values, width, label="今回",  color="#1f77b4", alpha=0.8)
bars2 = ax2.bar(x + width/2, ar_paper,  width, label="論文値", color="#ff7f0e", alpha=0.8)

# 数値ラベル
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

ax2.set_title("年率リターン比較（今回 vs 論文値）", fontsize=14)
ax2.set_ylabel("年率リターン（%）", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(strategy_names, fontsize=11)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout(pad=2.0)
plt.savefig(OUTPUT_DIR / "strategy_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 output/strategy_comparison.png に保存しました")

# ============================================================
# 4. 最終サマリー表示
# ============================================================

print("\n" + "="*55)
print("最終サマリー: 論文との比較")
print("="*55)
print(f"{'戦略':<12} {'R/R(今回)':>10} {'R/R(論文)':>10} {'順位一致':>8}")
print("-"*55)

order_now   = ["PCA SUB", "DOUBLE", "MOM", "PCA PLAIN"]
order_paper = ["PCA SUB", "DOUBLE", "PCA PLAIN", "MOM"]

rr_now = {n: float(df_results.loc[n,"R/R"]) for n in strategy_names}
rr_paper_dict = {"MOM": 0.53, "PCA PLAIN": 0.62, "PCA SUB": 2.22, "DOUBLE": 1.69}

for name in strategy_names:
    match = "✅" if (rr_now[name] > 1.0) == (rr_paper_dict[name] > 1.0) else "△"
    print(f"{name:<12} {rr_now[name]:>10.2f} {rr_paper_dict[name]:>10.2f} {match:>8}")

print("\n論文の主張「PCA SUB が最も優れている」→", end=" ")
if rr_now["PCA SUB"] == max(rr_now.values()):
    print("✅ 再現できています！")
else:
    print("△ 順位が異なります")
    