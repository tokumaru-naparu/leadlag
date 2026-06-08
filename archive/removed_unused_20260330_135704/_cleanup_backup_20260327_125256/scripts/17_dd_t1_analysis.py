"""
17_dd_t1_analysis.py
DD動的 dd_t1 発動回数の確認と、t1候補の比較。

出力:
  1) dd_t1=-6% の発動回数（全期間・年別）と発動日一覧
  2) t1=-6/-8/-10/-12 の比較表
  3) 年平均発動回数に基づく自動判定コメント
  4) output/results/dd_t1_analysis.csv
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
    RESULTS_DIR,
    pick_largest_csv,
)


# ================================================================
# 確定パラメータ（共通）
# ================================================================
SIG_COEF = 2.0
SIZE_CAP = 3.0
SIZE_MIN = 0.1
SHORT_AMP = 2.0
TARGET_VOL = 0.015

DD_S1 = 0.7
DD_T2 = -0.18
DD_S2 = 0.2

T1_PATTERNS = [
    ("現行", -0.06),
    ("候補", -0.08),
    ("参考1", -0.10),
    ("参考2", -0.12),
]


# ================================================================
# データ読み込み
# ================================================================
print("=" * 70)
print("DD動的 dd_t1 分析")
print("=" * 70)
print("\n[1] データ読み込み...")


def load_csv(filename: str) -> pd.DataFrame:
    candidates = [
        LEGACY_SCRIPTS_HISTORY_DIR / filename,
        PROCESSED_DIR / filename,
        LEGACY_HISTORY_DIR / filename,
    ]
    path = pick_largest_csv(candidates)
    if path is None:
        raise FileNotFoundError(f"{filename} が見つかりません")

    df = pd.read_csv(path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    print(f"  ✅ {filename}: {len(df)}行 ({path})")
    return df


signals = load_csv("signals.csv")
trades = load_csv("trades.csv")

df = pd.DataFrame(index=signals.index)
df["signal_strength"] = signals["signal_strength"]
df["long_return"] = trades["long_return"]
df["short_return"] = trades["short_return"]
df = df.dropna(subset=["signal_strength", "long_return", "short_return"])


# シグナル分位（全期間ランク基準）
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

# 先読み防止
base_ret = df["long_return"] - df["short_return"]
df["recent_vol_20"] = base_ret.rolling(20).std().shift(1)
df["recent_vol_20"] = df["recent_vol_20"].fillna(base_ret.std())

print(
    f"  使用期間: {df.index.min().date()} ～ {df.index.max().date()} / {len(df)}日"
)


# ================================================================
# 方式E（DDトリガー記録付き）
# ================================================================
def run_E_with_triggers(
    df_sub: pd.DataFrame,
    dd_t1: float,
    dd_s1: float,
    dd_t2: float,
    dd_s2: float,
) -> tuple[pd.Series, pd.DataFrame]:
    sig = df_sub["signal_strength"].values
    lr = df_sub["long_return"].values
    sr = df_sub["short_return"].values
    q = df_sub["sig_quintile"].values
    vol = df_sub["recent_vol_20"].values
    idx = df_sub.index

    # 1) ボラターゲット
    vol_adj = np.where(vol > 0, TARGET_VOL / vol, 1.0)

    # 2) サイズ
    size = np.clip(sig * SIG_COEF, SIZE_MIN, SIZE_CAP)
    size = np.clip(size * vol_adj, SIZE_MIN, SIZE_CAP)

    # 3) ベースリターン
    base_r = np.where(q >= 4, 0.7 * lr - SHORT_AMP * sr, lr - sr)
    raw = size * base_r

    # 4) DD動的縮小 + 発動記録
    out = np.zeros(len(raw))
    records = []
    capital = 1.0
    peak = 1.0

    for i, r in enumerate(raw):
        dd_now = (capital - peak) / peak
        triggered_t1 = dd_now <= dd_t1

        if dd_now <= dd_t2:
            scale = dd_s2
            regime = "t2"
        elif dd_now <= dd_t1:
            scale = dd_s1
            regime = "t1"
        else:
            scale = 1.0
            regime = "normal"

        if triggered_t1:
            records.append(
                {
                    "date": idx[i],
                    "dd": dd_now,
                    "regime": regime,
                    "scale": scale,
                }
            )

        out[i] = r * scale
        capital *= 1 + out[i]
        if capital > peak:
            peak = capital

    return pd.Series(out, index=idx, name="strategy_return"), pd.DataFrame(records)


def evaluate(returns: pd.Series) -> dict:
    r = returns.dropna()
    if len(r) == 0:
        return {
            "annual_r": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "calmar": 0.0,
            "ret_2025": 0.0,
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

    r2025 = r[r.index.year == 2025]
    ret_2025 = (1 + r2025).prod() - 1 if len(r2025) > 0 else 0.0

    return {
        "annual_r": annual_r,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "ret_2025": ret_2025,
    }


def trigger_comment(avg_count: float) -> str:
    if avg_count <= 2:
        return "適切"
    if 3 <= avg_count <= 4:
        return "やや多い・-8%を検討"
    return "多すぎ・-8%以上に変更推奨"


# ================================================================
# ① dd_t1=-6% の発動回数
# ================================================================
print("\n[2] dd_t1=-6% 発動回数の集計...")

ret_current, trig_current = run_E_with_triggers(
    df,
    dd_t1=-0.06,
    dd_s1=DD_S1,
    dd_t2=DD_T2,
    dd_s2=DD_S2,
)

if len(trig_current) > 0:
    trig_current["year"] = trig_current["date"].dt.year
    yearly_counts = trig_current.groupby("year").size().rename("count")
else:
    yearly_counts = pd.Series(dtype=int)

print("\n" + "=" * 70)
print("① dd_t1=-6% 発動回数（全期間・年別）")
print("=" * 70)

years = sorted(df.index.year.unique())
for y in years:
    c = int(yearly_counts.get(y, 0))
    print(f"  {y}: {c}回")

print(f"\n  合計発動回数: {len(trig_current)}回")

print("\n  発動日とDD水準（dd_t1=-6%）")
if len(trig_current) == 0:
    print("    発動なし")
else:
    for _, row in trig_current.iterrows():
        print(
            f"    {row['date'].date()}  DD={row['dd']:.2%}  "
            f"regime={row['regime']}  scale={row['scale']:.1f}"
        )


# ================================================================
# ② t1比較
# ================================================================
print("\n[3] t1パターン比較...")

rows = []
for name, t1 in T1_PATTERNS:
    ret, trig = run_E_with_triggers(
        df,
        dd_t1=t1,
        dd_s1=DD_S1,
        dd_t2=DD_T2,
        dd_s2=DD_S2,
    )
    ev = evaluate(ret)

    n_years = len(df.index.year.unique())
    total_trigger = len(trig)
    avg_trigger = total_trigger / n_years if n_years > 0 else 0.0
    comment = trigger_comment(avg_trigger)

    rows.append(
        {
            "パターン": name,
            "dd_t1": f"{t1:.2f}",
            "年率": ev["annual_r"],
            "Sharpe": ev["sharpe"],
            "MaxDD": ev["max_dd"],
            "Calmar": ev["calmar"],
            "2025年リターン": ev["ret_2025"],
            "発動回数_合計": total_trigger,
            "発動回数_年平均": avg_trigger,
            "判定コメント": comment,
        }
    )

cmp = pd.DataFrame(rows)

print("\n" + "=" * 70)
print("② t1=-6% vs -8% vs -10% vs -12% 比較表")
print("=" * 70)

show = cmp.copy()
for c in ["年率", "MaxDD", "2025年リターン"]:
    show[c] = show[c].map(lambda x: f"{x:.1%}")
for c in ["Sharpe", "Calmar", "発動回数_年平均"]:
    show[c] = show[c].map(lambda x: f"{x:.2f}")

print(
    show[
        [
            "パターン",
            "dd_t1",
            "年率",
            "Sharpe",
            "MaxDD",
            "Calmar",
            "2025年リターン",
            "発動回数_合計",
            "発動回数_年平均",
            "判定コメント",
        ]
    ].to_string(index=False)
)


# ================================================================
# ③ 判定コメント
# ================================================================
print("\n" + "=" * 70)
print("③ 自動判定コメント")
print("=" * 70)
for _, row in cmp.iterrows():
    print(
        f"  {row['パターン']} (dd_t1={row['dd_t1']}): "
        f"年平均 {row['発動回数_年平均']:.2f}回 -> {row['判定コメント']}"
    )


# ================================================================
# CSV出力
# ================================================================
print("\n[4] CSV出力...")
out_csv = RESULTS_DIR / "dd_t1_analysis.csv"
cmp.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"  ✅ {out_csv}")

print("\n" + "=" * 70)
print("完了")
print("=" * 70)