# 07_step1_method_comparison.py
# Step 1: 方式対決（A/D/E × 3設定 = 9パターン）
# ホールドアウト期間（2024.10〜2026.3）を除いた8年データで比較
# 評価指標: 年別Calmarの中央値・最悪ケースCalmar・前半/後半比較

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# config読み込み
# ================================================================
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    PROCESSED_DIR, LEGACY_SCRIPTS_HISTORY_DIR, LEGACY_HISTORY_DIR,
    CHARTS_DIR, RESULTS_DIR, pick_largest_csv
)

# ================================================================
# 期間設定
# ================================================================
TRAIN_END    = '2024-09-30'   # 学習期間の終わり
HOLDOUT_START = '2024-10-01'  # ホールドアウト開始（今回は使わない）
FRONT_END    = '2020-09-30'   # 前半/後半の境目（約4年ずつ）

# ================================================================
# データ読み込み
# ================================================================
print("=" * 60)
print("データ読み込み中...")
print("=" * 60)

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
    print(f"  ✅ {filename}: {len(df)}行")
    return df

signals = load_csv('signals.csv')
trades  = load_csv('trades.csv')

# 全データ結合
df_all = pd.DataFrame(index=signals.index)
df_all['signal_strength'] = signals['signal_strength']
df_all['long_return']     = trades['long_return']
df_all['short_return']    = trades['short_return']
df_all['is_correct']      = trades['is_correct']
df_all = df_all.dropna(subset=['signal_strength', 'long_return', 'short_return'])

# LS独立用：ロング・ショート個別シグナル計算
print("  LS独立用シグナル計算中...")
signal_cols = [c for c in signals.columns
               if c.startswith('signal_') and c not in ('signal_strength', 'signal_spread')]
sig_map = {c.replace('signal_', ''): c for c in signal_cols}

long_sig_list, short_sig_list = [], []
for date in df_all.index:
    if date not in trades.index or date not in signals.index:
        long_sig_list.append(0.05)
        short_sig_list.append(0.05)
        continue
    t_row = trades.loc[date]
    s_row = signals.loc[date]
    lvals, svals = [], []
    for col in ['long_1', 'long_2', 'long_3']:
        sc = sig_map.get(str(t_row.get(col, '')).strip())
        if sc and sc in s_row:
            lvals.append(abs(s_row[sc]))
    for col in ['short_1', 'short_2', 'short_3']:
        sc = sig_map.get(str(t_row.get(col, '')).strip())
        if sc and sc in s_row:
            svals.append(abs(s_row[sc]))
    long_sig_list.append(np.mean(lvals) if lvals else 0.05)
    short_sig_list.append(np.mean(svals) if svals else 0.05)

df_all['long_signal']  = long_sig_list
df_all['short_signal'] = short_sig_list

# ホールドアウト除外（学習用データのみ）
df = df_all[:TRAIN_END].copy()

# シグナル分位（ランク基準）
df['sig_rank'] = df['signal_strength'].rank(pct=True)
def to_quintile(r):
    if pd.isna(r): return 3
    if r <= 0.20:  return 1
    if r <= 0.40:  return 2
    if r <= 0.60:  return 3
    if r <= 0.80:  return 4
    return 5
df['sig_quintile'] = df['sig_rank'].apply(to_quintile).astype(int)

# ボラターゲット用：直近20日ボラ（先読み防止）
base_r = df['long_return'] - df['short_return']
df['recent_vol_20'] = base_r.rolling(20).std().shift(1).fillna(base_r.std())

# numpy配列化
sig_arr        = df['signal_strength'].values
long_r_arr     = df['long_return'].values
short_r_arr    = df['short_return'].values
quintile_arr   = df['sig_quintile'].values
recent_vol_arr = df['recent_vol_20'].values
long_sig_arr   = df['long_signal'].values
short_sig_arr  = df['short_signal'].values
dates_arr      = df.index

print(f"\n  学習データ: {len(df)}日 ({df.index.min().date()} 〜 {df.index.max().date()})")
print(f"  ホールドアウト: {HOLDOUT_START} 〜 2026-03-19（今回は未使用）")

# ================================================================
# バックテスト関数
# ================================================================
def run_backtest(returns_arr, dd_dynamic=True,
                 dd_t1=-0.10, dd_s1=0.5,
                 dd_t2=-0.15, dd_s2=0.3):
    """DD動的縮小を適用してリターン配列を返す"""
    if not dd_dynamic:
        return returns_arr.copy()

    result = np.zeros(len(returns_arr))
    capital = 1.0
    peak    = 1.0
    for i, r in enumerate(returns_arr):
        dd = (capital - peak) / peak
        if dd <= dd_t2:
            scale = dd_s2
        elif dd <= dd_t1:
            scale = dd_s1
        else:
            scale = 1.0
        result[i] = r * scale
        capital *= (1 + result[i])
        if capital > peak:
            peak = capital
    return result

# ================================================================
# 評価関数
# ================================================================
def evaluate_full(returns_arr, dates):
    """全体・前半・後半・年別Calmarを計算"""
    s = pd.Series(returns_arr, index=dates)
    if len(s) == 0:
        return {}

    def _metrics(r):
        if len(r) == 0:
            return {'annual_r': 0, 'max_dd': 0, 'calmar': 0, 'sharpe': 0}
        cum     = (1 + r).cumprod()
        n_years = len(r) / 252
        total_r = cum.iloc[-1] - 1
        ann_r   = (1 + total_r) ** (1 / n_years) - 1 if n_years > 0 else 0
        std     = r.std() * np.sqrt(252)
        sharpe  = ann_r / std if std > 0 else 0
        peak    = cum.cummax()
        dd      = (cum - peak) / peak
        max_dd  = dd.min()
        calmar  = ann_r / abs(max_dd) if max_dd != 0 else 0
        return {'annual_r': ann_r, 'max_dd': max_dd,
                'calmar': calmar, 'sharpe': sharpe}

    # 全体
    m_all   = _metrics(s)
    # 前半
    m_front = _metrics(s[:FRONT_END])
    # 後半
    m_back  = _metrics(s[FRONT_END:])

    # 年別Calmar
    yearly_calmar = []
    yearly_r      = []
    for year in sorted(s.index.year.unique()):
        sy = s[s.index.year == year]
        if len(sy) < 50:  # 50日未満の年はスキップ
            continue
        my = _metrics(sy)
        yearly_calmar.append(my['calmar'])
        yearly_r.append(my['annual_r'])

    return {
        'annual_r':      m_all['annual_r'],
        'sharpe':        m_all['sharpe'],
        'max_dd':        m_all['max_dd'],
        'calmar':        m_all['calmar'],
        'calmar_median': np.median(yearly_calmar) if yearly_calmar else 0,
        'calmar_min':    min(yearly_calmar) if yearly_calmar else 0,
        'calmar_front':  m_front['calmar'],
        'calmar_back':   m_back['calmar'],
        'annual_r_min':  min(yearly_r) if yearly_r else 0,
        'win_rate':      (s > 0).mean(),
        'yearly_calmar': yearly_calmar,
        'yearly_r':      yearly_r,
    }

# ================================================================
# 9パターン定義（Opusが提示した設定）
# ================================================================

# --- 方式A ---
def base_A(sig_coef, size_cap, size_min, short_amp):
    sizes = np.clip(sig_arr * sig_coef, size_min, size_cap)
    q_high = (quintile_arr >= 4)
    base_r = np.where(
        q_high,
        0.7 * long_r_arr - short_amp * short_r_arr,
        long_r_arr - short_r_arr
    )
    return sizes * base_r

# --- 方式D ---
def base_D(long_coef, short_coef, size_cap, size_min):
    ls = np.clip(long_sig_arr  * long_coef,  size_min, size_cap)
    ss = np.clip(short_sig_arr * short_coef, size_min, size_cap)
    return ls * long_r_arr - ss * short_r_arr

# --- 方式E ---
def base_E(sig_coef, size_cap, size_min, short_amp, target_vol):
    vol_adj   = np.where(recent_vol_arr > 0, target_vol / recent_vol_arr, 1.0)
    base_size = np.clip(sig_arr * sig_coef, size_min, size_cap)
    sizes     = np.clip(base_size * vol_adj, size_min, size_cap)
    q_high    = (quintile_arr >= 4)
    base_r    = np.where(
        q_high,
        0.7 * long_r_arr - short_amp * short_r_arr,
        long_r_arr - short_r_arr
    )
    return sizes * base_r

# 9パターン（控えめ・標準・攻撃的）
patterns = {
    # 方式A
    'A_控えめ': {'method': 'A', 'params': dict(sig_coef=3, size_cap=2.0, size_min=0.2, short_amp=1.0)},
    'A_標準':   {'method': 'A', 'params': dict(sig_coef=4, size_cap=3.0, size_min=0.3, short_amp=1.15)},
    'A_攻撃的': {'method': 'A', 'params': dict(sig_coef=5, size_cap=4.0, size_min=0.3, short_amp=1.3)},
    # 方式D
    'D_控えめ': {'method': 'D', 'params': dict(long_coef=3, short_coef=5,  size_cap=3.0, size_min=0.2)},
    'D_標準':   {'method': 'D', 'params': dict(long_coef=5, short_coef=8,  size_cap=4.0, size_min=0.2)},
    'D_攻撃的': {'method': 'D', 'params': dict(long_coef=5, short_coef=10, size_cap=4.0, size_min=0.1)},
    # 方式E
    'E_控えめ': {'method': 'E', 'params': dict(sig_coef=2, size_cap=3.0, size_min=0.2, short_amp=1.0,  target_vol=0.010)},
    'E_標準':   {'method': 'E', 'params': dict(sig_coef=2, size_cap=4.0, size_min=0.2, short_amp=1.1,  target_vol=0.015)},
    'E_攻撃的': {'method': 'E', 'params': dict(sig_coef=3, size_cap=4.0, size_min=0.2, short_amp=1.3,  target_vol=0.020)},
}

# ================================================================
# バックテスト実行
# ================================================================
print("\n" + "=" * 60)
print("バックテスト実行中...")
print("=" * 60)

results = {}
for name, cfg in patterns.items():
    m = cfg['method']
    p = cfg['params']
    if m == 'A':
        base = base_A(**p)
    elif m == 'D':
        base = base_D(**p)
    else:
        base = base_E(**p)

    # DD動的なし
    r_no_dd = base
    # DD動的あり（デフォルト）
    r_dd    = run_backtest(base, dd_dynamic=True)

    results[name] = {
        'no_dd': evaluate_full(r_no_dd, dates_arr),
        'dd':    evaluate_full(r_dd,    dates_arr),
        'returns_dd': r_dd,
    }
    print(f"  ✅ {name}")

# ================================================================
# 結果表示
# ================================================================
cols_main = ['annual_r', 'sharpe', 'max_dd', 'calmar',
             'calmar_median', 'calmar_min', 'calmar_front', 'calmar_back', 'annual_r_min']
col_labels = ['年率', 'Sharpe', 'MaxDD', 'Calmar全体',
              'Calmar中央値★', 'Calmar最悪', 'Calmar前半', 'Calmar後半', '最悪年リターン']

def fmt(v, key):
    if key in ('annual_r', 'max_dd', 'annual_r_min'):
        return f"{v:.1%}"
    elif key in ('sharpe', 'calmar', 'calmar_median', 'calmar_min', 'calmar_front', 'calmar_back'):
        return f"{v:.2f}"
    return str(v)

for dd_label, dd_key in [('DD動的なし（参考）', 'no_dd'), ('DD動的あり（判断基準）', 'dd')]:
    print(f"\n{'=' * 80}")
    print(f"【{dd_label}】")
    print(f"{'=' * 80}")

    header = f"{'パターン':<12}" + "".join([f"{l:>12}" for l in col_labels])
    print(header)
    print("-" * (12 + 12 * len(col_labels)))

    for name, res in results.items():
        r = res[dd_key]
        row = f"{name:<12}"
        for key in cols_main:
            row += f"{fmt(r.get(key, 0), key):>12}"
        # Calmar中央値が3.0以上かつMaxDD-20%以内にマーク
        if r.get('calmar_median', 0) >= 3.0 and r.get('max_dd', -1) >= -0.20:
            row += "  ★"
        print(row)

print("\n★ = Calmar中央値3.0以上 かつ MaxDD -20%以内")

# ================================================================
# 方式別「最悪ケースCalmar」比較（Opusの判断基準）
# ================================================================
print(f"\n{'=' * 60}")
print("【方式別 最悪ケース比較（DD動的あり）】")
print("Opusの指示: 各方式の最悪設定のCalmarで方式を選ぶ")
print(f"{'=' * 60}")

for method in ['A', 'D', 'E']:
    method_patterns = {k: v for k, v in results.items() if k.startswith(method)}
    worst_calmar  = min(v['dd']['calmar_median'] for v in method_patterns.values())
    best_calmar   = max(v['dd']['calmar_median'] for v in method_patterns.values())
    worst_pattern = min(method_patterns, key=lambda k: method_patterns[k]['dd']['calmar_median'])
    print(f"\n  方式{method}:")
    print(f"    最悪設定のCalmar中央値: {worst_calmar:.2f} ({worst_pattern})")
    print(f"    最良設定のCalmar中央値: {best_calmar:.2f}")
    for name, res in method_patterns.items():
        r = res['dd']
        print(f"    {name:<12} Calmar中央値:{r['calmar_median']:.2f}  "
              f"最悪年:{r['annual_r_min']:.1%}  MaxDD:{r['max_dd']:.1%}")

# ================================================================
# 年別Calmar詳細
# ================================================================
print(f"\n{'=' * 60}")
print("【年別Calmar詳細（DD動的あり）】")
print(f"{'=' * 60}")

# 年のリスト取得
years = sorted(df.index.year.unique())
header = f"{'パターン':<12}" + "".join([f"{y:>8}" for y in years]) + f"{'中央値':>8}"
print(header)
print("-" * (12 + 8 * (len(years) + 1)))

for name, res in results.items():
    r = res['dd']
    s = pd.Series(res['returns_dd'], index=dates_arr)
    row = f"{name:<12}"
    yc_list = []
    for year in years:
        sy = s[s.index.year == year]
        if len(sy) < 50:
            row += f"{'―':>8}"
            continue
        cum  = (1 + sy).cumprod()
        n_y  = len(sy) / 252
        ann  = (1 + cum.iloc[-1] - 1) ** (1 / n_y) - 1
        pk   = cum.cummax()
        dd   = (cum - pk) / pk
        mdd  = dd.min()
        cal  = ann / abs(mdd) if mdd != 0 else 0
        yc_list.append(cal)
        row += f"{cal:>8.1f}"
    row += f"{np.median(yc_list):>8.2f}"
    print(row)

# ================================================================
# CSV出力
# ================================================================
rows = []
for name, res in results.items():
    for dd_key, dd_label in [('no_dd', 'DD動的なし'), ('dd', 'DD動的あり')]:
        r = res[dd_key]
        row = {'パターン': name, 'DD動的': dd_label}
        for key, label in zip(cols_main, col_labels):
            row[label] = fmt(r.get(key, 0), key)
        rows.append(row)

out_csv = RESULTS_DIR / 'step1_method_comparison.csv'
pd.DataFrame(rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n✅ CSV出力: {out_csv}")

# ================================================================
# グラフ出力
# ================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams['font.family'] = 'MS Gothic'

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Step1 方式対決 — 累積資産推移（DD動的あり）', fontsize=14)

    colors = {'A': '#333333', 'D': '#2ecc71', 'E': '#f39c12'}
    styles = {'控えめ': ':', '標準': '-', '攻撃的': '--'}

    for idx, (name, res) in enumerate(results.items()):
        ax  = axes[idx // 3][idx % 3]
        method = name.split('_')[0]
        level  = name.split('_')[1]
        s   = pd.Series(res['returns_dd'], index=dates_arr)
        cum = (1 + s).cumprod() * 100

        ax.plot(cum.index, cum.values,
                color=colors[method], linestyle=styles[level], linewidth=2)
        ax.axvline(pd.Timestamp(FRONT_END), color='gray',
                   linestyle=':', alpha=0.5, label='前半/後半境目')

        r = res['dd']
        ax.set_title(f"{name}\n年率{r['annual_r']:.1%} MaxDD{r['max_dd']:.1%} "
                     f"Calmar中央値{r['calmar_median']:.2f}", fontsize=9)
        ax.set_ylabel('資産（万円）')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = CHARTS_DIR / 'step1_method_comparison.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ グラフ出力: {out_png}")

except Exception as e:
    print(f"⚠ グラフ出力失敗: {e}")

print(f"\n{'=' * 60}")
print("完了！次のステップ: 結果を見て勝者方式を決定 → Step2へ")
print(f"{'=' * 60}")