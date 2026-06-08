# 06_grid_search_ADE.py
# 方式A（H+）・方式D（LS独立）・方式E（H+×ボラ合体）の一晩グリッドサーチ
# 合計約6,000パターンを10年バックテストで比較

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import itertools
import time
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

df = pd.DataFrame(index=signals.index)
df['signal_strength'] = signals['signal_strength']
df['long_return']     = trades['long_return']
df['short_return']    = trades['short_return']
df = df.dropna(subset=['signal_strength', 'long_return', 'short_return'])

# LS独立サイジング用：ロング・ショート個別シグナル強度を事前計算
print("  LS独立用シグナル計算中...")
signal_cols = [c for c in signals.columns
               if c.startswith('signal_') and c not in ('signal_strength', 'signal_spread')]
sig_map = {c.replace('signal_', ''): c for c in signal_cols}

long_sig_list  = []
short_sig_list = []
for date in df.index:
    if date not in trades.index or date not in signals.index:
        long_sig_list.append(0.05)
        short_sig_list.append(0.05)
        continue
    t_row = trades.loc[date]
    s_row = signals.loc[date]
    lvals, svals = [], []
    for side, lst in [(['long_1','long_2','long_3'], lvals),
                      (['short_1','short_2','short_3'], svals)]:
        for col in side:
            sector = str(t_row.get(col, '')).strip()
            sc = sig_map.get(sector)
            if sc and sc in s_row:
                lst.append(abs(s_row[sc]))
    long_sig_list.append(np.mean(lvals) if lvals else 0.05)
    short_sig_list.append(np.mean(svals) if svals else 0.05)

df['long_signal']  = long_sig_list
df['short_signal'] = short_sig_list

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

# ボラターゲット用：直近20日ボラを事前計算（先読み防止でshift(1)）
base_r = df['long_return'] - df['short_return']
df['recent_vol_20'] = base_r.rolling(20).std().shift(1).fillna(base_r.std())

print(f"\n  使用データ: {len(df)}日 ({df.index.min().date()} 〜 {df.index.max().date()})")

# ================================================================
# numpy配列化（高速化）
# ================================================================
sig_arr      = df['signal_strength'].values
long_r_arr   = df['long_return'].values
short_r_arr  = df['short_return'].values
quintile_arr = df['sig_quintile'].values
recent_vol_arr = df['recent_vol_20'].values
long_sig_arr   = df['long_signal'].values
short_sig_arr  = df['short_signal'].values
n = len(df)

# ================================================================
# 評価関数（numpy高速版）
# ================================================================
def evaluate_fast(returns_arr):
    """numpy配列で高速評価"""
    if len(returns_arr) == 0:
        return None
    cum      = np.cumprod(1 + returns_arr)
    total_r  = cum[-1] - 1
    n_years  = len(returns_arr) / 252
    annual_r = (1 + total_r) ** (1 / n_years) - 1
    std      = np.std(returns_arr) * np.sqrt(252)
    sharpe   = annual_r / std if std > 0 else 0
    peak     = np.maximum.accumulate(cum)
    dd       = (cum - peak) / peak
    max_dd   = dd.min()
    calmar   = annual_r / abs(max_dd) if max_dd != 0 else 0
    win_rate = np.mean(returns_arr > 0)
    return {
        'annual_r': annual_r,
        'sharpe':   sharpe,
        'max_dd':   max_dd,
        'calmar':   calmar,
        'win_rate': win_rate,
        'final':    cum[-1] * 100,
    }

# ================================================================
# 方式A: H+ バックテスト（numpy高速版）
# ================================================================
def run_A(sig_coef, size_cap, size_min, short_amp):
    sizes = np.clip(sig_arr * sig_coef, size_min, size_cap)
    # Q4/Q5: ロング0.7倍・ショートshort_amp倍
    q_high = (quintile_arr >= 4)
    base_r = np.where(
        q_high,
        0.7 * long_r_arr - short_amp * short_r_arr,
        long_r_arr - short_r_arr
    )
    return sizes * base_r

# ================================================================
# 方式E: H+×ボラ合体 バックテスト（numpy高速版）
# ================================================================
def run_E(sig_coef, size_cap, size_min, short_amp, target_vol, vol_window=20):
    # ボラ調整サイズ
    vol_adj  = np.where(recent_vol_arr > 0, target_vol / recent_vol_arr, 1.0)
    base_size = np.clip(sig_arr * sig_coef, size_min, size_cap)
    sizes    = np.clip(base_size * vol_adj, size_min, size_cap)
    # Q4/Q5: ショート増幅
    q_high = (quintile_arr >= 4)
    base_r = np.where(
        q_high,
        0.7 * long_r_arr - short_amp * short_r_arr,
        long_r_arr - short_r_arr
    )
    return sizes * base_r

# ================================================================
# 方式D: LS独立サイジング（numpy高速版）
# ================================================================
def run_D(long_coef, short_coef, size_cap, size_min):
    """
    LS独立: ロング側・ショート側のシグナル強度を個別にサイズ決定
    ロングが強い日はロングを大きく、ショートが強い日はショートを大きく
    """
    long_sizes  = np.clip(long_sig_arr  * long_coef,  size_min, size_cap)
    short_sizes = np.clip(short_sig_arr * short_coef, size_min, size_cap)
    return long_sizes * long_r_arr - short_sizes * short_r_arr

# ================================================================
# グリッドパラメータ定義
# ================================================================

# --- 方式A ---
grid_A = {
    'sig_coef':  [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],           # 7
    'size_cap':  [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],            # 7
    'size_min':  [0.1, 0.2, 0.3, 0.5],                             # 4
    'short_amp': [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5], # 9
}
# 7×7×4×9 = 1,764パターン

# --- 方式E ---
grid_E = {
    'sig_coef':   [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],           # 7
    'size_cap':   [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],            # 7
    'size_min':   [0.1, 0.2, 0.3, 0.5],                             # 4
    'short_amp':  [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5], # 9
    'target_vol': [0.003, 0.005, 0.007, 0.008, 0.01, 0.012, 0.015, 0.02], # 8
}
# 7×7×4×9×8 = 14,112パターン ← 多すぎるので絞る

# target_volとsig_coefの組み合わせを絞る（夜通し用に約3,000パターン）
grid_E_reduced = {
    'sig_coef':   [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],           # 7
    'size_cap':   [2.0, 2.5, 3.0, 3.5, 4.0],                       # 5
    'size_min':   [0.1, 0.2, 0.3],                                  # 3
    'short_amp':  [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],                 # 6
    'target_vol': [0.005, 0.007, 0.01, 0.012, 0.015],              # 5
}
# 7×5×3×6×5 = 3,150パターン

# --- 方式D ---
grid_D = {
    'long_coef':  [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],   # 8
    'short_coef': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],   # 8
    'size_cap':   [2.0, 2.5, 3.0, 3.5, 4.0],                     # 5
    'size_min':   [0.1, 0.2, 0.3],                                # 3
}
# 8×8×5×3 = 960パターン

total_A = (len(grid_A['sig_coef']) * len(grid_A['size_cap']) *
           len(grid_A['size_min']) * len(grid_A['short_amp']))
total_D = (len(grid_D['long_coef']) * len(grid_D['short_coef']) *
           len(grid_D['size_cap'])  * len(grid_D['size_min']))
total_E = (len(grid_E_reduced['sig_coef']) * len(grid_E_reduced['size_cap']) *
           len(grid_E_reduced['size_min']) * len(grid_E_reduced['short_amp']) *
           len(grid_E_reduced['target_vol']))

print(f"\n方式A: {total_A}パターン")
print(f"方式D: {total_D}パターン")
print(f"方式E: {total_E}パターン")
print(f"合計:  {total_A + total_D + total_E}パターン")

# ================================================================
# グリッドサーチ実行
# ================================================================
def run_grid(method_name, grid_params, run_func, param_keys):
    print(f"\n{'=' * 60}")
    print(f"【{method_name}】グリッドサーチ開始...")
    print(f"{'=' * 60}")

    combos   = list(itertools.product(*[grid_params[k] for k in param_keys]))
    total    = len(combos)
    results  = []
    start_t  = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(param_keys, combo))
        ret    = run_func(**params)
        res    = evaluate_fast(ret)
        if res is None:
            continue
        row = {**params, **res}
        results.append(row)

        # 進捗表示（100件ごと）
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - start_t
            eta     = elapsed / (i + 1) * (total - i - 1)
            print(f"  {i+1:>5}/{total}  経過:{elapsed/60:.1f}分  残り:{eta/60:.1f}分", end='\r')

    print(f"\n  完了！ ({time.time()-start_t:.1f}秒)")
    return pd.DataFrame(results)

# 方式A実行
df_A = run_grid('方式A: H+', grid_A, run_A,
                ['sig_coef', 'size_cap', 'size_min', 'short_amp'])

# 方式D実行
df_D = run_grid('方式D: LS独立', grid_D, run_D,
                ['long_coef', 'short_coef', 'size_cap', 'size_min'])

# 方式E実行
df_E = run_grid('方式E: H+×ボラ合体', grid_E_reduced, run_E,
                ['sig_coef', 'size_cap', 'size_min', 'short_amp', 'target_vol'])

# ================================================================
# 結果集計・出力
# ================================================================
def summarize(df_res, method_name, top_n=20):
    print(f"\n{'=' * 60}")
    print(f"【{method_name}】上位{top_n}パターン（Calmar順）")
    print(f"{'=' * 60}")

    # Calmar順でソート
    df_sorted = df_res.sort_values('calmar', ascending=False).head(top_n)

    # 表示列
    param_cols  = [c for c in df_res.columns if c in
                   ['sig_coef','size_cap','size_min','short_amp','target_vol']]
    metric_cols = ['annual_r','sharpe','max_dd','calmar','win_rate','final']

    header = "  " + "".join([f"{c:>12}" for c in param_cols + metric_cols])
    print(header)
    print("  " + "-" * (12 * (len(param_cols) + len(metric_cols))))

    for _, row in df_sorted.iterrows():
        line = "  "
        for c in param_cols:
            line += f"{row[c]:>12.3f}"
        line += f"{row['annual_r']:>11.1%}"
        line += f"{row['sharpe']:>12.2f}"
        line += f"{row['max_dd']:>11.1%}"
        line += f"{row['calmar']:>12.2f}"
        line += f"{row['win_rate']:>11.1%}"
        line += f"{row['final']:>12.0f}"
        # 目標達成（年率50%+・MaxDD-20%以内）にマーク
        if row['annual_r'] >= 0.50 and row['max_dd'] >= -0.20:
            line += "  ★"
        print(line)

    # 統計サマリー
    print(f"\n  全{len(df_res)}パターンの統計:")
    print(f"    年率    中央値:{df_res['annual_r'].median():.1%}  最大:{df_res['annual_r'].max():.1%}")
    print(f"    MaxDD   中央値:{df_res['max_dd'].median():.1%}  最小:{df_res['max_dd'].min():.1%}")
    print(f"    Calmar  中央値:{df_res['calmar'].median():.2f}  最大:{df_res['calmar'].max():.2f}")
    print(f"    ★条件達成(年率50%+・DD-20%以内): {((df_res['annual_r']>=0.50)&(df_res['max_dd']>=-0.20)).sum()}パターン")

summarize(df_A, '方式A: H+')
summarize(df_D, '方式D: LS独立')
summarize(df_E, '方式E: H+×ボラ合体')

# ================================================================
# パレートフロンティア（年率 vs MaxDD のトレードオフ上位）
# ================================================================
def pareto_frontier(df_res, method_name):
    """年率を最大化しMaxDDを最小化するパレート最適パターンを抽出"""
    df_s = df_res.sort_values('annual_r', ascending=False).copy()
    pareto = []
    best_dd = -np.inf
    for _, row in df_s.iterrows():
        if row['max_dd'] > best_dd:  # max_ddは負なので大きい=小さいDD
            pareto.append(row)
            best_dd = row['max_dd']
    return pd.DataFrame(pareto)

print(f"\n{'=' * 60}")
print("【パレートフロンティア比較】")
print("年率を犠牲にせずMaxDDを改善するパターン")
print(f"{'=' * 60}")

pf_A = pareto_frontier(df_A, '方式A')
pf_D = pareto_frontier(df_D, '方式D')
pf_E = pareto_frontier(df_E, '方式E')

for pf, label, p_cols in [
    (pf_A, '方式A: H+',        ['sig_coef','size_cap','size_min','short_amp']),
    (pf_D, '方式D: LS独立',    ['long_coef','short_coef','size_cap','size_min']),
    (pf_E, '方式E: H+×ボラ合体', ['sig_coef','size_cap','size_min','short_amp','target_vol']),
]:
    print(f"\n{label} パレート上位:")
    print(f"  {'annual_r':>10} {'max_dd':>10} {'calmar':>10} {'sharpe':>10} {'params'}")
    for _, row in pf.head(10).iterrows():
        cols = [c for c in p_cols if c in row.index]
        params_str = " ".join([f"{c}={row[c]:.3f}" for c in cols])
        print(f"  {row['annual_r']:>9.1%} {row['max_dd']:>10.1%} {row['calmar']:>10.2f} {row['sharpe']:>10.2f}  {params_str}")

# ================================================================
# CSV出力
# ================================================================
df_A['method'] = 'A'
df_D['method'] = 'D'
df_E['method'] = 'E'
df_all = pd.concat([df_A, df_D, df_E], ignore_index=True)

out_all    = RESULTS_DIR / 'grid_search_ADE_full.csv'
out_pareto = RESULTS_DIR / 'grid_search_ADE_pareto.csv'

df_all.to_csv(out_all, index=False, encoding='utf-8-sig')

pf_all = pd.concat([
    pf_A.assign(method='A'),
    pf_D.assign(method='D'),
    pf_E.assign(method='E')
], ignore_index=True)
pf_all.to_csv(out_pareto, index=False, encoding='utf-8-sig')

print(f"\n✅ 全結果CSV:      {out_all}")
print(f"✅ パレートCSV:    {out_pareto}")

# ================================================================
# グラフ: パレートフロンティア比較
# ================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'MS Gothic'
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, df_res, pf, title, color in [
        (axes[0], df_A, pf_A, '方式A: H+',        '#333333'),
        (axes[1], df_D, pf_D, '方式D: LS独立',     '#2ecc71'),
        (axes[2], df_E, pf_E, '方式E: H+×ボラ合体', '#f39c12'),
    ]:
        # 全パターンを散布図
        ax.scatter(df_res['max_dd'] * 100, df_res['annual_r'] * 100,
                   alpha=0.15, s=8, color=color, label='全パターン')

        # パレートフロンティアを強調
        ax.scatter(pf['max_dd'] * 100, pf['annual_r'] * 100,
                   s=60, color='red', zorder=5, label='パレート最適')
        ax.plot(pf.sort_values('max_dd')['max_dd'] * 100,
                pf.sort_values('max_dd')['annual_r'] * 100,
                color='red', linewidth=1.5, zorder=4)

        # デフォルト設定の位置
        default = df_res.iloc[(df_res['calmar'] - df_res['calmar'].max()).abs().argsort()[:1]]
        ax.scatter(default['max_dd'] * 100, default['annual_r'] * 100,
                   s=200, marker='★', color='blue', zorder=6, label='Calmar最高')

        ax.axvline(x=-20, color='gray', linestyle='--', alpha=0.5, label='MaxDD -20%')
        ax.axhline(y=50,  color='gray', linestyle=':', alpha=0.5, label='年率50%')
        ax.set_xlabel('MaxDD (%)')
        ax.set_ylabel('年率 (%)')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('グリッドサーチ結果: パレートフロンティア（年率 vs MaxDD）', fontsize=14)
    plt.tight_layout()
    out_png = CHARTS_DIR / 'grid_search_ADE_pareto.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ グラフ出力: {out_png}")

except Exception as e:
    print(f"⚠ グラフ出力失敗: {e}")

print(f"\n{'=' * 60}")
print("完了！")
print(f"{'=' * 60}")