"""
09_walkforward_12weeks.py
過去12週間（約3ヶ月）のウォークフォワードテスト

毎週月曜〜金曜の5日間を1週として
12週分の週次収益と資産推移を表示する
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# パラメータ
# ============================================================

L              = 60
K              = 3
LAMBDA         = 0.9
TOP_N          = 3
INITIAL_CAPITAL = 1_000_000
N_WEEKS        = 52

US_TICKERS = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
JP_TICKERS = ["1617.T","1618.T","1619.T","1620.T","1621.T","1622.T","1623.T",
              "1624.T","1625.T","1626.T","1627.T","1628.T","1629.T","1630.T",
              "1631.T","1632.T","1633.T"]

JP_NAMES = {
    "1617.T": "食品",        "1618.T": "エネルギー",
    "1619.T": "建設・資材",  "1620.T": "素材・化学",
    "1621.T": "医薬品",      "1622.T": "自動車",
    "1623.T": "鉄鋼・非鉄",  "1624.T": "機械",
    "1625.T": "電機・精密",  "1626.T": "情報通信",
    "1627.T": "電力・ガス",  "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",  "1630.T": "小売",
    "1631.T": "銀行",        "1632.T": "金融",
    "1633.T": "不動産",
}

US_CYCLICAL  = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE = ["XLK", "XLP", "XLU", "XLV"]
JP_CYCLICAL  = ["1618.T", "1625.T", "1629.T", "1631.T"]
JP_DEFENSIVE = ["1617.T", "1621.T", "1627.T", "1630.T"]

# ============================================================
# PCA SUB 関数群
# ============================================================

def standardize(df):
    mu    = df.mean()
    sigma = df.std().replace(0, np.nan)
    return (df - mu) / sigma, mu, sigma

def build_prior_subspace(available_tickers):
    n      = len(available_tickers)
    is_us  = np.array([t in US_TICKERS for t in available_tickers], dtype=float)
    is_jp  = np.array([t in JP_TICKERS for t in available_tickers], dtype=float)
    v1     = np.ones(n) / np.sqrt(n)
    v2_raw = is_us - is_jp
    v2_raw -= np.dot(v2_raw, v1) * v1
    norm2  = np.linalg.norm(v2_raw)
    v2     = v2_raw / norm2 if norm2 > 1e-10 else np.zeros(n)
    v3_raw = np.zeros(n)
    for i, t in enumerate(available_tickers):
        if t in US_CYCLICAL or t in JP_CYCLICAL:
            v3_raw[i] = +1.0
        elif t in US_DEFENSIVE or t in JP_DEFENSIVE:
            v3_raw[i] = -1.0
    v3_raw -= np.dot(v3_raw, v1) * v1
    v3_raw -= np.dot(v3_raw, v2) * v2
    norm3  = np.linalg.norm(v3_raw)
    v3     = v3_raw / norm3 if norm3 > 1e-10 else np.zeros(n)
    return np.column_stack([v1, v2, v3])

def build_C0(V0, cfull):
    D0     = np.diag(np.diag(V0.T @ cfull @ V0))
    C0_raw = V0 @ D0 @ V0.T
    diag_v = np.where(np.diag(C0_raw) > 0, np.diag(C0_raw), 1.0)
    C0     = np.diag(1/np.sqrt(diag_v)) @ C0_raw @ np.diag(1/np.sqrt(diag_v))
    np.fill_diagonal(C0, 1.0)
    return C0

def calc_signal(window_us, window_jp, z_us_today, cfull_mat, cfull_tickers):
    combined = pd.concat([window_us, window_jp], axis=1)
    combined = combined.dropna(axis=1, how="all").fillna(0)
    if len(combined.columns) < K + 1:
        return pd.Series(np.nan, index=JP_TICKERS)
    available = combined.columns.tolist()
    us_avail  = [t for t in available if t in US_TICKERS]
    jp_avail  = [t for t in available if t in JP_TICKERS]
    if not us_avail or not jp_avail:
        return pd.Series(np.nan, index=JP_TICKERS)
    z_comb, _, _ = standardize(combined)
    Ct   = z_comb.values.T @ z_comb.values / len(z_comb)
    V0   = build_prior_subspace(available)
    cidx = [cfull_tickers.index(t) for t in available if t in cfull_tickers]
    C0   = build_C0(V0, cfull_mat[np.ix_(cidx, cidx)]) if len(cidx)==Ct.shape[0] else np.eye(len(available))
    C_reg = (1-LAMBDA)*Ct + LAMBDA*C0
    evals, evecs = np.linalg.eigh(C_reg)
    V    = evecs[:, np.argsort(evals)[::-1][:K]]
    ui   = [available.index(t) for t in us_avail]
    ji   = [available.index(t) for t in jp_avail]
    _, mu_us, sg = standardize(window_us.reindex(columns=us_avail).fillna(0))
    zt   = ((z_us_today.reindex(us_avail) - mu_us) / sg.replace(0, np.nan)).fillna(0)
    ft   = V[ui,:].T @ zt.values
    sig  = pd.Series(np.nan, index=JP_TICKERS)
    sig.loc[jp_avail] = V[ji,:] @ ft
    return sig

def calc_longshort(signal, next_ret):
    valid = signal.dropna()
    if len(valid) < 2:
        return np.nan
    long_t  = valid.nlargest(TOP_N).index
    short_t = valid.nsmallest(TOP_N).index
    return float(next_ret[long_t].mean() - next_ret[short_t].mean())

# ============================================================
# 1. データ取得（過去6ヶ月分）
# ============================================================

print("=== 過去12週間ウォークフォワードテスト ===")
print("📡 データ取得中...")

import yfinance as yf
from datetime import datetime

today      = datetime.today()
fetch_start = (today - timedelta(days=730)).strftime("%Y-%m-%d")
fetch_end   = today.strftime("%Y-%m-%d")

us_raw = yf.download(US_TICKERS, start=fetch_start, end=fetch_end,
                     auto_adjust=True, progress=False)
jp_raw = yf.download(JP_TICKERS, start=fetch_start, end=fetch_end,
                     auto_adjust=True, progress=False)

us_cc = us_raw["Close"].reindex(columns=US_TICKERS).pct_change()
jp_cc = jp_raw["Close"].reindex(columns=JP_TICKERS).pct_change()
jp_oc = ((jp_raw["Close"] - jp_raw["Open"]) / jp_raw["Open"]).reindex(columns=JP_TICKERS)

common = us_cc.index.intersection(jp_cc.index)
us_cc  = us_cc.loc[common]
jp_cc  = jp_cc.loc[common]
jp_oc  = jp_oc.loc[common]

print(f"取得完了: {len(common)} 営業日  "
      f"({common[0].date()} 〜 {common[-1].date()})")

# Cfull
cdf = pd.concat([us_cc, jp_cc], axis=1).dropna(axis=1, how="all").fillna(0)
z_c, _, _ = standardize(cdf)
cfull_mat     = z_c.values.T @ z_c.values / len(z_c)
cfull_tickers = cdf.columns.tolist()

# ============================================================
# 2. 12週間分の日次リターンを計算
# ============================================================

print("\n⚙️  シグナル計算中...")

daily_returns = []

for i in range(L, len(common) - 1):
    today_idx = common[i]
    next_idx  = common[i + 1]

    sig = calc_signal(
        us_cc.iloc[i-L:i],
        jp_cc.iloc[i-L:i],
        us_cc.iloc[i],
        cfull_mat, cfull_tickers
    )
    ret = calc_longshort(sig, jp_oc.loc[next_idx])
    daily_returns.append({
        "date":   next_idx,
        "return": ret,
    })

daily_df = pd.DataFrame(daily_returns).set_index("date")
daily_df = daily_df.dropna()

# 直近12週間（約60営業日）だけ使う
n_days   = N_WEEKS * 5
daily_df = daily_df.tail(n_days)

if daily_df.empty:
    raise ValueError("No valid daily returns were generated. Check data quality and signal settings.")

# 週次集計で使うため、indexの日付を列としても保持
daily_df["date"] = daily_df.index

print(f"対象期間: {daily_df.index[0].date()} 〜 {daily_df.index[-1].date()}")
print(f"営業日数: {len(daily_df)} 日")

# ============================================================
# 3. 週次集計
# ============================================================

# ISO週番号で集計
daily_df["week_num"]   = daily_df.index.isocalendar().week.values
daily_df["year"]       = daily_df.index.year
daily_df["week_label"] = daily_df.index.to_series().apply(
    lambda d: f"{d.year}-W{d.isocalendar()[1]:02d}"
)

weekly = daily_df.groupby("week_label").agg(
    week_start  = ("date", "first"),   # 週の最初の営業日
    week_end    = ("date", "last"),    # 週の最後の営業日
    days        = ("return", "count"),
    weekly_return = ("return", lambda x: (1 + x).prod() - 1),  # 複利計算
    win_days    = ("return", lambda x: (x > 0).sum()),
    avg_daily   = ("return", "mean"),
).reset_index()

# 週番号を1〜12に振り直す
weekly = weekly.sort_values("week_start").reset_index(drop=True)
weekly.index = range(1, len(weekly) + 1)
weekly.index.name = "週"

# 累積資産を計算
capital = INITIAL_CAPITAL
capitals = []
for ret in weekly["weekly_return"]:
    capital *= (1 + ret)
    capitals.append(round(capital))
weekly["cumulative_capital"] = capitals
weekly["cumulative_pnl_pct"] = (weekly["cumulative_capital"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

# ============================================================
# 4. 結果表示
# ============================================================

print("\n" + "="*75)
print("12週間ウォークフォワードテスト結果")
print("="*75)
print(f"{'週':>3}  {'期間':^22}  {'週次収益':>8}  {'勝日/日数':>8}  "
      f"{'累積資産':>12}  {'累積損益':>8}")
print("-"*75)

for w, row in weekly.iterrows():
    period   = f"{row['week_start'].date()} 〜 {row['week_end'].date()}"
    ret_str  = f"{row['weekly_return']*100:+.2f}%"
    win_str  = f"{int(row['win_days'])}/{int(row['days'])}日"
    cap_str  = f"{int(row['cumulative_capital']):>12,}円"
    pnl_str  = f"{row['cumulative_pnl_pct']:+.2f}%"
    mark     = "✅" if row["weekly_return"] > 0 else "❌"
    print(f"{w:>3}  {period:<22}  {ret_str:>8}  {win_str:>8}  "
          f"{cap_str}  {pnl_str:>8}  {mark}")

print("-"*75)

# サマリー
total_ret   = (weekly["cumulative_capital"].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
win_weeks   = (weekly["weekly_return"] > 0).sum()
total_weeks = len(weekly)
best_week   = weekly.loc[weekly["weekly_return"].idxmax()]
worst_week  = weekly.loc[weekly["weekly_return"].idxmin()]

print(f"\n📊 12週間サマリー")
print(f"  運用期間   : {weekly['week_start'].iloc[0].date()} 〜 {weekly['week_end'].iloc[-1].date()}")
print(f"  初期資産   : {INITIAL_CAPITAL:>12,} 円")
print(f"  最終資産   : {int(weekly['cumulative_capital'].iloc[-1]):>12,} 円")
print(f"  総損益     : {int(weekly['cumulative_capital'].iloc[-1]-INITIAL_CAPITAL):>+12,} 円  ({total_ret:+.2f}%)")
print(f"  週次勝率   : {win_weeks}/{total_weeks}週  ({win_weeks/total_weeks*100:.1f}%)")
print(f"  最良週     : W{best_week.name}  {best_week['weekly_return']*100:+.2f}%")
print(f"  最悪週     : W{worst_week.name}  {worst_week['weekly_return']*100:+.2f}%")
print(f"  平均週次   : {weekly['weekly_return'].mean()*100:+.2f}%")

# 年率換算（12週 → 52週換算）
annualized = (1 + weekly["weekly_return"].mean()) ** 52 - 1
print(f"  年率換算   : {annualized*100:+.2f}%  （週次平均を52週換算）")

# ============================================================
# 5. グラフ
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(13, 9))

# ── グラフ①: 累積資産推移 ────────────────────────────────
ax1 = axes[0]
ax1.plot(weekly.index, weekly["cumulative_capital"] / 10000,
         color="#1f77b4", marker="o", linewidth=2, markersize=6)
ax1.axhline(y=INITIAL_CAPITAL/10000, color="gray",
            linestyle="--", linewidth=1, label="初期資産 100万円")
ax1.fill_between(
    weekly.index,
    INITIAL_CAPITAL / 10000,
    weekly["cumulative_capital"] / 10000,
    where=weekly["cumulative_capital"] >= INITIAL_CAPITAL,
    color="#1f77b4", alpha=0.15, label="利益"
)
ax1.fill_between(
    weekly.index,
    INITIAL_CAPITAL / 10000,
    weekly["cumulative_capital"] / 10000,
    where=weekly["cumulative_capital"] < INITIAL_CAPITAL,
    color="#d62728", alpha=0.15, label="損失"
)
ax1.set_title(f"累積資産推移（100万円スタート）  総損益: {total_ret:+.2f}%",
              fontsize=13)
ax1.set_ylabel("資産（万円）", fontsize=11)
ax1.set_xlabel("週", fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(weekly.index)
ax1.set_xticklabels([f"W{w}" for w in weekly.index], fontsize=9)

# ── グラフ②: 週次リターン棒グラフ ────────────────────────
ax2 = axes[1]
colors = ["#1f77b4" if r > 0 else "#d62728"
          for r in weekly["weekly_return"]]
bars = ax2.bar(weekly.index, weekly["weekly_return"] * 100,
               color=colors, alpha=0.8, edgecolor="white")
ax2.axhline(y=0, color="black", linewidth=0.8)

# 数値ラベル
for bar, ret in zip(bars, weekly["weekly_return"]):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2,
             h + (0.05 if h >= 0 else -0.15),
             f"{ret*100:+.2f}%",
             ha="center", va="bottom" if h >= 0 else "top",
             fontsize=8)

ax2.set_title(f"週次リターン  勝率: {win_weeks}/{total_weeks}週 ({win_weeks/total_weeks*100:.0f}%)",
              fontsize=13)
ax2.set_ylabel("週次リターン（%）", fontsize=11)
ax2.set_xlabel("週", fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_xticks(weekly.index)
ax2.set_xticklabels([f"W{w}" for w in weekly.index], fontsize=9)

plt.tight_layout(pad=2.0)
plt.savefig(OUTPUT_DIR / "walkforward_12weeks.png", dpi=150, bbox_inches="tight")
plt.show()

# CSV保存
weekly.to_csv(OUTPUT_DIR / "walkforward_12weeks.csv", encoding="utf-8-sig")
print(f"\n💾 output/walkforward_12weeks.png")
print(f"💾 output/walkforward_12weeks.csv")