"""
16_step3_holdout_final.py
Step3: ホールドアウト最終検証（方式E）

要件:
- 学習期間: 2016-05-16 ～ 2024-09-30
- ホールドアウト: 2024-10-01 ～ 2026-03
- 確定パラメータで固定検証
- 学習/テスト成績、年別・月別リターン、合格判定、比較表を出力
- output/charts と output/results へ保存
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ================================================================
# config読み込み
# ================================================================
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    PROCESSED_DIR,
    LEGACY_SCRIPTS_HISTORY_DIR,
    LEGACY_HISTORY_DIR,
    CHARTS_DIR,
    RESULTS_DIR,
    pick_largest_csv,
)


# ================================================================
# 固定パラメータ
# ================================================================
TRAIN_END = "2024-09-30"
HOLDOUT_START = "2024-10-01"

SIG_COEF = 2.0
SIZE_CAP = 3.0
SIZE_MIN = 0.1
SHORT_AMP = 2.0
TARGET_VOL = 0.015

DD_T1 = -0.06
DD_S1 = 0.7
DD_T2 = -0.18
DD_S2 = 0.2

# 条件3で使う固定参照値（Round3）
TRAIN_WORST_ROUND_MAXDD = -0.122


# ================================================================
# データ読み込み
# ================================================================
print("=" * 70)
print("Step3 ホールドアウト最終検証（方式E）")
print("=" * 70)
print("\n[1] データ読み込み...")


def load_csv(filename: str) -> pd.DataFrame:
    """scripts/data/history を優先し、存在する候補の中で最大行数を採用する。"""
    candidates = [
        LEGACY_SCRIPTS_HISTORY_DIR / filename,
        PROCESSED_DIR / filename,
        LEGACY_HISTORY_DIR / filename,
    ]
    path = pick_largest_csv(candidates)
    if path is None:
        raise FileNotFoundError(f"{filename} が見つかりません")
    df = pd.read_csv(path, parse_dates=["date"])  # history系はdate列
    df.set_index("date", inplace=True)
    print(f"  ✅ {filename}: {len(df)}行 ({path})")
    return df


signals = load_csv("signals.csv")
trades = load_csv("trades.csv")

df = pd.DataFrame(index=signals.index)
df["signal_strength"] = signals["signal_strength"]
df["long_return"] = trades["long_return"]
df["short_return"] = trades["short_return"]
df["is_correct"] = trades["is_correct"]
df = df.dropna(subset=["signal_strength", "long_return", "short_return"])


# シグナル分位: 全期間ランク基準
df["sig_rank"] = df["signal_strength"].rank(pct=True)


def to_quintile(r: float) -> int:
    if pd.isna(r):
        return 3
    if r <= 0.20:
        return 1
    if r <= 0.40:
        return 2
    if r <= 0.60:
        return 3
    if r <= 0.80:
        return 4
    return 5


df["sig_quintile"] = df["sig_rank"].apply(to_quintile).astype(int)

# 先読み防止: shift(1)
base_ret = df["long_return"] - df["short_return"]
df["recent_vol_20"] = base_ret.rolling(20).std().shift(1)
df["recent_vol_20"] = df["recent_vol_20"].fillna(base_ret.std())

print(
    f"  使用データ: {len(df)}日 / {df.index.min().date()} ～ {df.index.max().date()}"
)


# ================================================================
# 方式E
# ================================================================
PARAMS = {
    "sig_coef": SIG_COEF,
    "size_cap": SIZE_CAP,
    "size_min": SIZE_MIN,
    "short_amp": SHORT_AMP,
    "target_vol": TARGET_VOL,
    "dd_t1": DD_T1,
    "dd_s1": DD_S1,
    "dd_t2": DD_T2,
    "dd_s2": DD_S2,
}


def run_E(df_sub: pd.DataFrame, params: dict) -> pd.Series:
    """方式E + DD動的縮小で日次リターン系列を返す。"""
    sig = df_sub["signal_strength"].values
    lr = df_sub["long_return"].values
    sr = df_sub["short_return"].values
    q = df_sub["sig_quintile"].values
    vol = df_sub["recent_vol_20"].values

    # 1) ボラターゲット調整
    vol_adj = np.where(vol > 0, params["target_vol"] / vol, 1.0)

    # 2) ポジションサイズ
    size = np.clip(sig * params["sig_coef"], params["size_min"], params["size_cap"])
    size = np.clip(size * vol_adj, params["size_min"], params["size_cap"])

    # 3) リターン計算
    base_r = np.where(
        q >= 4,
        0.7 * lr - params["short_amp"] * sr,
        lr - sr,
    )
    raw = size * base_r

    # 4) DD動的縮小
    out = np.zeros(len(raw))
    capital = 1.0
    peak = 1.0
    for i, r in enumerate(raw):
        dd_now = (capital - peak) / peak
        if dd_now <= params["dd_t2"]:
            scale = params["dd_s2"]
        elif dd_now <= params["dd_t1"]:
            scale = params["dd_s1"]
        else:
            scale = 1.0

        out[i] = r * scale
        capital *= 1 + out[i]
        if capital > peak:
            peak = capital

    return pd.Series(out, index=df_sub.index, name="strategy_return")


# ================================================================
# 評価関数
# ================================================================
def evaluate(returns: pd.Series) -> dict:
    r = returns.dropna()
    if len(r) == 0:
        return {
            "annual_r": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "final_asset": 100.0,
            "cum": pd.Series(dtype=float),
            "dd": pd.Series(dtype=float),
        }

    cum = (1 + r).cumprod()
    n_years = len(r) / 252
    total_r = cum.iloc[-1] - 1
    annual_r = (1 + total_r) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    vol = r.std() * np.sqrt(252)
    sharpe = annual_r / vol if vol > 0 else 0.0
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = annual_r / abs(max_dd) if max_dd != 0 else 0.0
    win_rate = (r > 0).mean()

    return {
        "annual_r": annual_r,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "final_asset": cum.iloc[-1] * 100.0,
        "cum": cum,
        "dd": dd,
    }


def yearly_returns(returns: pd.Series) -> pd.DataFrame:
    rows = []
    for y in sorted(returns.index.year.unique()):
        sy = returns[returns.index.year == y]
        if len(sy) == 0:
            continue
        yr = (1 + sy).prod() - 1
        rows.append({"year": int(y), "return": yr})
    return pd.DataFrame(rows)


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    m = (1 + returns).resample("ME").prod() - 1
    out = m.reset_index()
    out.columns = ["month", "return"]
    out["month"] = out["month"].dt.strftime("%Y-%m")
    return out


def fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def fmt_num(x: float) -> str:
    return f"{x:.2f}"


# ================================================================
# 実行
# ================================================================
print("\n[2] 方式Eバックテスト...")
ret_all = run_E(df, PARAMS)

train_mask = ret_all.index <= pd.Timestamp(TRAIN_END)
holdout_mask = ret_all.index >= pd.Timestamp(HOLDOUT_START)

ret_train = ret_all[train_mask]
ret_holdout = ret_all[holdout_mask]

ev_train = evaluate(ret_train)
ev_holdout = evaluate(ret_holdout)

train_yearly = yearly_returns(ret_train)
holdout_monthly = monthly_returns(ret_holdout)


# ================================================================
# 合格判定
# ================================================================
print("\n[3] 合格判定...")
cond1_threshold = ev_train["sharpe"] * 0.70
cond3_threshold = TRAIN_WORST_ROUND_MAXDD * 1.5

cond1 = ev_holdout["sharpe"] >= cond1_threshold
cond2 = ev_holdout["win_rate"] > 0.50
cond3 = ev_holdout["max_dd"] >= cond3_threshold

overall_pass = cond1 and cond2 and cond3


# ================================================================
# 出力表示
# ================================================================
print("\n" + "=" * 70)
print("① 学習期間の成績（参照値）")
print("=" * 70)
print(f"  期間: {ret_train.index.min().date()} ～ {ret_train.index.max().date()}")
print(f"  年率:      {fmt_pct(ev_train['annual_r'])}")
print(f"  Sharpe:    {fmt_num(ev_train['sharpe'])}")
print(f"  MaxDD:     {fmt_pct(ev_train['max_dd'])}")
print(f"  Calmar:    {fmt_num(ev_train['calmar'])}")
print(f"  勝率:      {fmt_pct(ev_train['win_rate'])}")
print(f"  最終資産:  {ev_train['final_asset']:.0f}万円")

print("\n  年別リターン（2016〜2024）")
for _, row in train_yearly.iterrows():
    print(f"    {int(row['year'])}: {fmt_pct(row['return'])}")

print("\n" + "=" * 70)
print("② ホールドアウト期間の成績")
print("=" * 70)
print(f"  期間: {ret_holdout.index.min().date()} ～ {ret_holdout.index.max().date()}")
print(f"  年率:      {fmt_pct(ev_holdout['annual_r'])}")
print(f"  Sharpe:    {fmt_num(ev_holdout['sharpe'])}")
print(f"  MaxDD:     {fmt_pct(ev_holdout['max_dd'])}")
print(f"  Calmar:    {fmt_num(ev_holdout['calmar'])}")
print(f"  勝率:      {fmt_pct(ev_holdout['win_rate'])}")
print(f"  最終資産:  {ev_holdout['final_asset']:.0f}万円")

print("\n  月別リターン（2024-10〜2026-03）")
for _, row in holdout_monthly.iterrows():
    print(f"    {row['month']}: {fmt_pct(row['return'])}")

print("\n" + "=" * 70)
print("③ 合格判定")
print("=" * 70)
print(
    f"  条件1: テストSharpe >= 学習Sharpeの70%  "
    f"({ev_holdout['sharpe']:.2f} >= {cond1_threshold:.2f})  -> {'PASS' if cond1 else 'FAIL'}"
)
print(
    f"  条件2: テスト勝率 > 50%  "
    f"({ev_holdout['win_rate']:.1%} > 50.0%)  -> {'PASS' if cond2 else 'FAIL'}"
)
print(
    f"  条件3: テストMaxDD >= 学習最悪ラウンドMaxDD × 1.5  "
    f"({ev_holdout['max_dd']:.1%} >= {cond3_threshold:.1%})  -> {'PASS' if cond3 else 'FAIL'}"
)
print(f"\n  総合判定: {'✅ 合格' if overall_pass else '❌ 不合格'}")

print("\n" + "=" * 70)
print("④ 学習 vs ホールドアウト 比較表")
print("=" * 70)
cmp_df = pd.DataFrame(
    [
        {
            "期間": "学習",
            "年率": fmt_pct(ev_train["annual_r"]),
            "Sharpe": fmt_num(ev_train["sharpe"]),
            "MaxDD": fmt_pct(ev_train["max_dd"]),
            "Calmar": fmt_num(ev_train["calmar"]),
            "勝率": fmt_pct(ev_train["win_rate"]),
            "最終資産(万)": f"{ev_train['final_asset']:.0f}",
        },
        {
            "期間": "ホールドアウト",
            "年率": fmt_pct(ev_holdout["annual_r"]),
            "Sharpe": fmt_num(ev_holdout["sharpe"]),
            "MaxDD": fmt_pct(ev_holdout["max_dd"]),
            "Calmar": fmt_num(ev_holdout["calmar"]),
            "勝率": fmt_pct(ev_holdout["win_rate"]),
            "最終資産(万)": f"{ev_holdout['final_asset']:.0f}",
        },
    ]
)
print(cmp_df.to_string(index=False))


# ================================================================
# グラフ
# ================================================================
print("\n[4] グラフ出力...")
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams["font.family"] = "MS Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    full_eval = evaluate(ret_all)
    cum_all = full_eval["cum"] * 100.0  # 万円
    dd_all = full_eval["dd"] * 100.0

    train_part = cum_all[cum_all.index <= pd.Timestamp(TRAIN_END)]
    holdout_part = cum_all[cum_all.index >= pd.Timestamp(HOLDOUT_START)]

    train_dd = dd_all[dd_all.index <= pd.Timestamp(TRAIN_END)]
    holdout_dd = dd_all[dd_all.index >= pd.Timestamp(HOLDOUT_START)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 上段: 累積資産
    axes[0].plot(train_part.index, train_part.values, color="black", linewidth=2, label="学習")
    axes[0].plot(holdout_part.index, holdout_part.values, color="red", linewidth=2, label="ホールドアウト")
    axes[0].axvline(pd.Timestamp(HOLDOUT_START), color="gray", linestyle="--", linewidth=1.2)
    axes[0].set_title("Step3 Holdout Final — 累積資産推移（100万円スタート）")
    axes[0].set_ylabel("資産（万円）")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # 下段: ドローダウン
    axes[1].plot(train_dd.index, train_dd.values, color="black", linewidth=1.8, label="学習")
    axes[1].plot(holdout_dd.index, holdout_dd.values, color="red", linewidth=1.8, label="ホールドアウト")
    axes[1].axvline(pd.Timestamp(HOLDOUT_START), color="gray", linestyle="--", linewidth=1.2)
    axes[1].axhline(-15, color="#f39c12", linestyle=":", linewidth=1.5, label="-15%")
    axes[1].axhline(-20, color="#e74c3c", linestyle=":", linewidth=1.5, label="-20%")
    axes[1].set_title("ドローダウン推移")
    axes[1].set_ylabel("ドローダウン（%）")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    out_png = CHARTS_DIR / "step3_holdout_final.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out_png}")
except Exception as e:
    print(f"  ⚠ グラフ出力失敗: {e}")


# ================================================================
# CSV保存
# ================================================================
print("\n[5] CSV出力...")

summary_df = pd.DataFrame(
    [
        {
            "period": "train",
            "start": str(ret_train.index.min().date()),
            "end": str(ret_train.index.max().date()),
            "annual_r": ev_train["annual_r"],
            "sharpe": ev_train["sharpe"],
            "max_dd": ev_train["max_dd"],
            "calmar": ev_train["calmar"],
            "win_rate": ev_train["win_rate"],
            "final_asset_10k": ev_train["final_asset"],
        },
        {
            "period": "holdout",
            "start": str(ret_holdout.index.min().date()),
            "end": str(ret_holdout.index.max().date()),
            "annual_r": ev_holdout["annual_r"],
            "sharpe": ev_holdout["sharpe"],
            "max_dd": ev_holdout["max_dd"],
            "calmar": ev_holdout["calmar"],
            "win_rate": ev_holdout["win_rate"],
            "final_asset_10k": ev_holdout["final_asset"],
        },
    ]
)

judge_df = pd.DataFrame(
    [
        {
            "condition": "test_sharpe >= train_sharpe*0.70",
            "lhs": ev_holdout["sharpe"],
            "rhs": cond1_threshold,
            "pass": cond1,
        },
        {
            "condition": "test_win_rate > 0.50",
            "lhs": ev_holdout["win_rate"],
            "rhs": 0.50,
            "pass": cond2,
        },
        {
            "condition": "test_maxdd >= train_worst_round_maxdd*1.5",
            "lhs": ev_holdout["max_dd"],
            "rhs": cond3_threshold,
            "pass": cond3,
        },
        {
            "condition": "overall_pass",
            "lhs": int(overall_pass),
            "rhs": 1,
            "pass": overall_pass,
        },
    ]
)

curve_df = pd.DataFrame(
    {
        "date": ret_all.index,
        "daily_return": ret_all.values,
        "cum_asset_10k": (1 + ret_all).cumprod().values * 100.0,
    }
)

train_yearly_out = train_yearly.copy()
train_yearly_out["return_pct"] = train_yearly_out["return"] * 100.0

holdout_monthly_out = holdout_monthly.copy()
holdout_monthly_out["return_pct"] = holdout_monthly_out["return"] * 100.0

out_summary = RESULTS_DIR / "step3_holdout_summary.csv"
out_yearly = RESULTS_DIR / "step3_holdout_train_yearly.csv"
out_monthly = RESULTS_DIR / "step3_holdout_test_monthly.csv"
out_judge = RESULTS_DIR / "step3_holdout_judgement.csv"
out_curve = RESULTS_DIR / "step3_holdout_curve.csv"

summary_df.to_csv(out_summary, index=False, encoding="utf-8-sig")
train_yearly_out.to_csv(out_yearly, index=False, encoding="utf-8-sig")
holdout_monthly_out.to_csv(out_monthly, index=False, encoding="utf-8-sig")
judge_df.to_csv(out_judge, index=False, encoding="utf-8-sig")
curve_df.to_csv(out_curve, index=False, encoding="utf-8-sig")

print(f"  ✅ {out_summary}")
print(f"  ✅ {out_yearly}")
print(f"  ✅ {out_monthly}")
print(f"  ✅ {out_judge}")
print(f"  ✅ {out_curve}")

print("\n" + "=" * 70)
print("完了")
print("=" * 70)