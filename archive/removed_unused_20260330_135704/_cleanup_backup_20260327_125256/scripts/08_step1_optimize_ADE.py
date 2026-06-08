# 15_dd_param_table.py
# DDパラメータを変えたときの指標一覧表を出力
# 全データ（2016〜2026）で比較

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    PROCESSED_DIR, LEGACY_SCRIPTS_HISTORY_DIR, LEGACY_HISTORY_DIR,
    RESULTS_DIR, pick_largest_csv
)

# ================================================================
# 確定サイジングパラメータ
# ================================================================
BEST_SIZING = {
    'sig_coef':   2.0,
    'size_min':   0.1,
    'short_amp':  2.0,
    'target_vol': 0.015,
}
SIZE_CAP = 3.0

# ================================================================
# データ読み込み
# ================================================================
print("データ読み込み中...")

def load_csv(filename):
    candidates = [
        PROCESSED_DIR / filename,
        LEGACY_SCRIPTS_HISTORY_DIR / filename,
        LEGACY_HISTORY_DIR / filename,
    ]
    path = pick_largest_csv(candidates)
    if path is None:
        raise FileNotFoundError(f"{filename} が見つかりません")
    df = pd.read_csv(path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

signals = load_csv('signals.csv')
trades  = load_csv('trades.csv')

df = pd.DataFrame(index=signals.index)
df['signal_strength'] = signals['signal_strength']
df['long_return']     = trades['long_return']
df['short_return']    = trades['short_return']
df = df.dropna(subset=['signal_strength', 'long_return', 'short_return'])

df['sig_rank']     = df['signal_strength'].rank(pct=True)
df['sig_quintile'] = df['sig_rank'].apply(
    lambda r: 1 if r<=.2 else 2 if r<=.4 else 3 if r<=.6 else 4 if r<=.8 else 5
).astype(int)
base_r = df['long_return'] - df['short_return']
df['recent_vol_20'] = base_r.rolling(20).std().shift(1).fillna(base_r.std())

years_arr = df.index.year.values
print(f"✅ {len(df)}日 ({df.index.min().date()} 〜 {df.index.max().date()})")

# ================================================================
# バックテスト関数（回復条件付き）
# ================================================================
def run_E(df_sub, dd_t1, dd_s1, dd_t2, dd_s2, recovery=-0.05):
    """
    recovery: DDがこの水準まで回復したらフルサイズに戻る
    例: recovery=-0.05 → ピークから-5%まで回復したら再開
    """
    sig  = df_sub['signal_strength'].values
    lr   = df_sub['long_return'].values
    sr   = df_sub['short_return'].values
    q    = df_sub['sig_quintile'].values
    vol  = df_sub['recent_vol_20'].values

    vol_adj   = np.where(vol > 0, BEST_SIZING['target_vol'] / vol, 1.0)
    base_size = np.clip(sig * BEST_SIZING['sig_coef'], BEST_SIZING['size_min'], SIZE_CAP)
    sizes     = np.clip(base_size * vol_adj, BEST_SIZING['size_min'], SIZE_CAP)
    base_r    = np.where(q >= 4,
                         0.7 * lr - BEST_SIZING['short_amp'] * sr,
                         lr - sr)
    raw = sizes * base_r

    result  = np.zeros(len(raw))
    capital = 1.0
    peak    = 1.0
    for i, r in enumerate(raw):
        dd_now = (capital - peak) / peak
        # 回復条件：DDが recovery より浅くなったらフルサイズ
        if dd_now >= recovery:
            scale = 1.0
        elif dd_now <= dd_t2:
            scale = dd_s2
        elif dd_now <= dd_t1:
            scale = dd_s1
        else:
            scale = 1.0
        result[i] = r * scale
        capital   *= (1 + result[i])
        if capital > peak:
            peak = capital
    return result

# ================================================================
# 評価関数
# ================================================================
def evaluate(returns_arr):
    cum     = np.cumprod(1 + returns_arr)
    n_years = len(returns_arr) / 252
    ann_r   = cum[-1] ** (1/n_years) - 1
    std     = np.std(returns_arr) * np.sqrt(252)
    sharpe  = ann_r / std if std > 0 else 0
    peak    = np.maximum.accumulate(cum)
    dd      = (cum - peak) / peak
    max_dd  = dd.min()
    calmar  = ann_r / abs(max_dd) if max_dd != 0 else 0
    win_rate = np.mean(returns_arr > 0)

    yearly_calmar = []
    for year in np.unique(years_arr):
        mask = years_arr == year
        if mask.sum() < 40: continue
        yc   = np.cumprod(1 + returns_arr[mask])
        yn   = mask.sum() / 252
        ya   = yc[-1] ** (1/yn) - 1
        ypk  = np.maximum.accumulate(yc)
        ymdd = ((yc - ypk) / ypk).min()
        yearly_calmar.append(ya / abs(ymdd) if ymdd != 0 else 0)

    # 2025年の個別成績（暴落年）
    mask25   = years_arr == 2025
    ret25    = returns_arr[mask25]
    cum25    = np.cumprod(1 + ret25) if len(ret25) > 0 else np.array([1.0])
    ann25    = cum25[-1] ** (1 / (len(ret25)/252)) - 1 if len(ret25) > 50 else np.nan

    return {
        'annual_r':      ann_r,
        'sharpe':        sharpe,
        'max_dd':        max_dd,
        'calmar':        calmar,
        'calmar_median': np.median(yearly_calmar) if yearly_calmar else 0,
        'win_rate':      win_rate,
        'final_万':      cum[-1] * 100,
        'annual_r_2025': ann25,
    }

# ================================================================
# テストするDDパラメータ組み合わせ
# ================================================================
scenarios = [
    # ラベル,              dd_t1,  dd_s1, dd_t2,  dd_s2, recovery
    ('DDなし（基準）',      None,   None,  None,   None,  None),
    # --- 第1閾値を変えたパターン ---
    ('t1=-6% s1=0.7',     -0.06,  0.7,  -0.15,   0.3,  -0.05),
    ('t1=-8% s1=0.7',     -0.08,  0.7,  -0.15,   0.3,  -0.05),
    ('t1=-10% s1=0.7',    -0.10,  0.7,  -0.15,   0.3,  -0.05),
    ('t1=-12% s1=0.7',    -0.12,  0.7,  -0.15,   0.3,  -0.05),
    ('t1=-15% s1=0.7',    -0.15,  0.7,  -0.20,   0.3,  -0.05),
    # --- 第1スケールを変えたパターン ---
    ('t1=-6% s1=0.5',     -0.06,  0.5,  -0.15,   0.2,  -0.05),
    ('t1=-6% s1=0.7',     -0.06,  0.7,  -0.15,   0.2,  -0.05),
    ('t1=-6% s1=0.9',     -0.06,  0.9,  -0.15,   0.2,  -0.05),
    # --- 第2閾値を変えたパターン ---
    ('t2=-12% s2=0.2',    -0.06,  0.7,  -0.12,   0.2,  -0.05),
    ('t2=-15% s2=0.2',    -0.06,  0.7,  -0.15,   0.2,  -0.05),
    ('t2=-18% s2=0.2',    -0.06,  0.7,  -0.18,   0.2,  -0.05),
    ('t2=-20% s2=0.2',    -0.06,  0.7,  -0.20,   0.2,  -0.05),
    ('t2=-25% s2=0.2',    -0.06,  0.7,  -0.25,   0.2,  -0.05),
    # --- 第2スケールを変えたパターン（回復条件付き）---
    ('t2=-15% s2=0.0 回復-5%',  -0.06,  0.7, -0.15,  0.0,  -0.05),
    ('t2=-15% s2=0.0 回復-8%',  -0.06,  0.7, -0.15,  0.0,  -0.08),
    ('t2=-15% s2=0.1',          -0.06,  0.7, -0.15,  0.1,  -0.05),
    ('t2=-15% s2=0.2',          -0.06,  0.7, -0.15,  0.2,  -0.05),
    ('t2=-15% s2=0.3',          -0.06,  0.7, -0.15,  0.3,  -0.05),
    # --- 前回v2の確定値 ---
    ('v2確定値',                 -0.12,  0.7, -0.18,  0.1,  -0.05),
    # --- v3の確定値 ---
    ('v3確定値',                 -0.06,  0.7, -0.25,  0.3,  -0.05),
]

# ================================================================
# 全シナリオ実行
# ================================================================
print("\nバックテスト実行中...")
results = []

for label, dd_t1, dd_s1, dd_t2, dd_s2, recovery in scenarios:
    if dd_t1 is None:
        # DDなし
        vol_adj   = np.where(df['recent_vol_20'].values > 0,
                             BEST_SIZING['target_vol'] / df['recent_vol_20'].values, 1.0)
        base_size = np.clip(df['signal_strength'].values * BEST_SIZING['sig_coef'],
                            BEST_SIZING['size_min'], SIZE_CAP)
        sizes     = np.clip(base_size * vol_adj, BEST_SIZING['size_min'], SIZE_CAP)
        base_r    = np.where(df['sig_quintile'].values >= 4,
                             0.7 * df['long_return'].values - BEST_SIZING['short_amp'] * df['short_return'].values,
                             df['long_return'].values - df['short_return'].values)
        ret = sizes * base_r
    else:
        ret = run_E(df, dd_t1, dd_s1, dd_t2, dd_s2, recovery)

    ev = evaluate(ret)
    results.append({
        'シナリオ':       label,
        'dd_t1':         f"{dd_t1*100:.0f}%" if dd_t1 else "―",
        'dd_s1':         f"{dd_s1:.1f}" if dd_s1 else "―",
        'dd_t2':         f"{dd_t2*100:.0f}%" if dd_t2 else "―",
        'dd_s2':         f"{dd_s2:.1f}" if dd_s2 is not None and dd_t1 else "―",
        '年率':           f"{ev['annual_r']:.1%}",
        'Sharpe':        f"{ev['sharpe']:.2f}",
        'MaxDD':         f"{ev['max_dd']:.1%}",
        'Calmar':        f"{ev['calmar']:.2f}",
        'Calmar中央値':  f"{ev['calmar_median']:.2f}",
        '勝率':           f"{ev['win_rate']:.1%}",
        '最終資産(万)':   f"{ev['final_万']:.0f}",
        '2025年リターン': f"{ev['annual_r_2025']:.1%}" if not np.isnan(ev['annual_r_2025']) else "―",
    })

# ================================================================
# 表示
# ================================================================
print("\n" + "=" * 110)
print("【DDパラメータ別 指標一覧表】（全データ 2016〜2026）")
print("=" * 110)

cols = ['シナリオ', 'dd_t1', 'dd_s1', 'dd_t2', 'dd_s2',
        '年率', 'Sharpe', 'MaxDD', 'Calmar', 'Calmar中央値', '勝率', '最終資産(万)', '2025年リターン']
widths = [26, 7, 7, 7, 7, 8, 8, 8, 8, 12, 7, 14, 14]

header = "".join([f"{c:>{w}}" for c, w in zip(cols, widths)])
print(header)
print("-" * sum(widths))

for res in results:
    row = "".join([f"{str(res.get(c,'―')):>{w}}" for c, w in zip(cols, widths)])
    # MaxDD-20%以内かつ年率80%以上にマーク
    max_dd_val = float(res['MaxDD'].replace('%','')) / 100
    ann_r_val  = float(res['年率'].replace('%','')) / 100
    if res['シナリオ'] != 'DDなし（基準）' and max_dd_val >= -0.20 and ann_r_val >= 0.80:
        row += "  ★"
    print(row)

print("\n★ = MaxDD-20%以内 かつ 年率80%以上")

# ================================================================
# CSV出力
# ================================================================
out_csv = RESULTS_DIR / 'dd_param_table.csv'
pd.DataFrame(results).to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n✅ CSV出力: {out_csv}")
print("完了！")