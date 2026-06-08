# 05_sizing_comparison.py
# サイジング方式対決（A:H+ / B:ボラターゲット / C:ケリー / D:LS独立 / E:合体）
# デフォルト設定で10年バックテストし、4スパンで比較する

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# config.py を読み込む
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

def load_csv(filename: str) -> pd.DataFrame:
    """新構成→旧構成の順で探して読み込む"""
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
    print(f"  ✅ {filename}: {len(df)}行 ({path})")
    return df

signals = load_csv('signals.csv')
trades  = load_csv('trades.csv')

# 必要な列を1つのDataFrameにまとめる
df = pd.DataFrame(index=signals.index)
df['signal_strength'] = signals['signal_strength']
df['long_return']     = trades['long_return']
df['short_return']    = trades['short_return']
df['is_correct']      = trades['is_correct']

# LS独立サイジング用：ロング・ショート個別シグナル
# signal_banks等の列名からロング・ショートそれぞれの平均を計算
signal_cols = [c for c in signals.columns if c.startswith('signal_') and c != 'signal_strength' and c != 'signal_spread']
long_cols   = ['long_1', 'long_2', 'long_3']
short_cols  = ['short_1', 'short_2', 'short_3']

# ロング3業種・ショート3業種のシグナル値を取得
sig_map = {c.replace('signal_', ''): c for c in signal_cols}

def get_side_signal(row, side_cols, sig_map, signals_row):
    """その日のロング(またはショート)3業種の平均シグナル絶対値"""
    vals = []
    for col in side_cols:
        sector = str(row.get(col, '')).strip()
        sig_col = sig_map.get(sector)
        if sig_col and sig_col in signals_row:
            vals.append(abs(signals_row[sig_col]))
    return np.mean(vals) if vals else 0.0

print("  LS独立用シグナル計算中...")
long_sig_list  = []
short_sig_list = []
for date in df.index:
    if date not in trades.index or date not in signals.index:
        long_sig_list.append(np.nan)
        short_sig_list.append(np.nan)
        continue
    t_row = trades.loc[date]
    s_row = signals.loc[date]
    long_sig_list.append(get_side_signal(t_row, long_cols, sig_map, s_row))
    short_sig_list.append(get_side_signal(t_row, short_cols, sig_map, s_row))

df['long_signal']  = long_sig_list
df['short_signal'] = short_sig_list
df = df.dropna(subset=['signal_strength', 'long_return', 'short_return'])

print(f"\n  使用データ: {len(df)}日 ({df.index.min().date()} 〜 {df.index.max().date()})")

# ================================================================
# シグナル分位（ランク基準）
# ================================================================
df['sig_rank'] = df['signal_strength'].rank(pct=True)
def rank_to_quintile(r):
    if pd.isna(r): return 3
    if r <= 0.20:  return 1
    if r <= 0.40:  return 2
    if r <= 0.60:  return 3
    if r <= 0.80:  return 4
    return 5
df['sig_quintile'] = df['sig_rank'].apply(rank_to_quintile).astype(int)

# ================================================================
# 5つのサイジング方式
# ================================================================

# --- 方式A: H+（現行）---
def method_A(df):
    """H+: シグナル比例サイズ + Q4/Q5でショート1.3倍"""
    returns = []
    for _, row in df.iterrows():
        size = min(max(row['signal_strength'] * 4, 0.3), 3.0)
        if row['sig_quintile'] >= 4:
            r = 0.7 * row['long_return'] - 1.3 * row['short_return']
        else:
            r = row['long_return'] - row['short_return']
        returns.append(size * r)
    return pd.Series(returns, index=df.index, name='A_Hplus')

# --- 方式B: ボラティリティターゲット ---
def method_B(df, target_vol=0.01, window=20, size_min=0.3, size_max=3.0):
    """
    ボラターゲット: 日次リスクを毎日target_vol（デフォルト1%）に保つ
    recent_vol = 直近20日の戦略リターンの標準偏差
    size = target_vol / recent_vol
    """
    base_returns = df['long_return'] - df['short_return']
    recent_vol   = base_returns.rolling(window).std().shift(1)  # 先読み防止でshift(1)
    size = (target_vol / recent_vol).clip(size_min, size_max).fillna(1.0)
    returns = size * base_returns
    return pd.Series(returns.values, index=df.index, name='B_VolTarget')

# --- 方式C: ケリー基準 ---
def method_C(df, window=60, safety=0.5, size_min=0.3, size_max=3.0):
    """
    ケリー基準: 勝率と損益比から数学的に最適サイズを計算
    kelly = win_rate - (1 - win_rate) / profit_ratio
    size  = kelly * safety（ハーフケリーが標準）
    """
    returns = []
    base = df['long_return'] - df['short_return']

    for i, (date, row) in enumerate(df.iterrows()):
        if i < window:
            returns.append(base.iloc[i])  # ウォームアップ期間は等サイズ
            continue

        past = base.iloc[i-window:i]
        wins   = past[past > 0]
        losses = past[past < 0]

        win_rate = (past > 0).mean()
        if len(losses) == 0 or wins.mean() is np.nan:
            size = 1.0
        else:
            profit_ratio = wins.mean() / abs(losses.mean())
            kelly = win_rate - (1 - win_rate) / profit_ratio
            size  = max(kelly * safety, 0)  # マイナスケリーは0に

        size = np.clip(size, size_min, size_max)
        returns.append(size * base.iloc[i])

    return pd.Series(returns, index=df.index, name='C_Kelly')

# --- 方式D: LS独立サイジング ---
def method_D(df, coef=6.0, size_min=0.3, size_max=3.0):
    """
    LS独立: ロング側・ショート側のシグナル強度を個別に見てサイズ決定
    ロングが強い日はロングを大きく、ショートが強い日はショートを大きく
    """
    returns = []
    for _, row in df.iterrows():
        long_size  = np.clip(row['long_signal']  * coef, size_min, size_max)
        short_size = np.clip(row['short_signal'] * coef, size_min, size_max)
        r = long_size * row['long_return'] - short_size * row['short_return']
        returns.append(r)
    return pd.Series(returns, index=df.index, name='D_LSIndep')

# --- 方式E: H+ × ボラターゲット合体 ---
def method_E(df, target_vol=0.01, window=20, size_min=0.3, size_max=3.0):
    """
    合体: シグナル自信度 × リスク調整
    base_size = signal_strength * 4（H+のサイズ）
    vol_adj   = target_vol / recent_vol
    size      = base_size * vol_adj
    """
    base_returns = df['long_return'] - df['short_return']
    recent_vol   = base_returns.rolling(window).std().shift(1)
    recent_vol   = recent_vol.fillna(recent_vol.mean())

    returns = []
    for i, (_, row) in enumerate(df.iterrows()):
        base_size = min(max(row['signal_strength'] * 4, 0.3), 3.0)
        vol_adj   = target_vol / recent_vol.iloc[i] if recent_vol.iloc[i] > 0 else 1.0
        size      = np.clip(base_size * vol_adj, size_min, size_max)

        if row['sig_quintile'] >= 4:
            r = 0.7 * row['long_return'] - 1.3 * row['short_return']
        else:
            r = row['long_return'] - row['short_return']
        returns.append(size * r)

    return pd.Series(returns, index=df.index, name='E_Combined')

# ================================================================
# 評価関数
# ================================================================
def evaluate(returns_series):
    s = returns_series.dropna()
    if len(s) == 0:
        return {}
    cum      = (1 + s).cumprod()
    n_years  = len(s) / 252
    total_r  = cum.iloc[-1] - 1
    annual_r = (1 + total_r) ** (1 / n_years) - 1
    std      = s.std() * np.sqrt(252)
    sharpe   = annual_r / std if std > 0 else 0
    peak     = cum.cummax()
    dd       = (cum - peak) / peak
    max_dd   = dd.min()
    calmar   = annual_r / abs(max_dd) if max_dd != 0 else np.inf
    win_rate = (s > 0).mean()
    return {
        '年率':         f"{annual_r:.1%}",
        'Sharpe':       f"{sharpe:.2f}",
        'MaxDD':        f"{max_dd:.1%}",
        'Calmar':       f"{calmar:.2f}",
        '勝率':         f"{win_rate:.1%}",
        '最終資産(万)':  f"{cum.iloc[-1] * 100:.0f}",
        '_annual_r':    annual_r,
        '_max_dd':      max_dd,
        '_sharpe':      sharpe,
        '_calmar':      calmar,
    }

# ================================================================
# バックテスト実行
# ================================================================
print("\n" + "=" * 60)
print("バックテスト実行中...")
print("=" * 60)

methods = {
    'A: H+（現行）':        method_A(df),
    'B: ボラターゲット':     method_B(df),
    'C: ケリー基準':        method_C(df),
    'D: LS独立':            method_D(df),
    'E: H+×ボラ合体':      method_E(df),
}
for name in methods:
    print(f"  ✅ {name}")

# ================================================================
# スパン定義
# ================================================================
spans = {
    '10Y': slice(None, None),
    '5Y':  slice('2021-04-20', None),
    '3Y':  slice('2023-04-11', None),
    '1Y':  slice('2025-03-28', None),
}

# ================================================================
# 10年比較表
# ================================================================
print("\n" + "=" * 60)
print("【10年 方式比較表】")
print("=" * 60)

cols = ['年率', 'Sharpe', 'MaxDD', 'Calmar', '勝率', '最終資産(万)']
header = f"{'方式':<22}" + "".join([f"{c:>12}" for c in cols])
print(header)
print("-" * (22 + 12 * len(cols)))

results_10y = {}
for name, ret in methods.items():
    res = evaluate(ret)
    results_10y[name] = res
    row_str = f"{name:<22}" + "".join([f"{res.get(c,'―'):>12}" for c in cols])
    # 優秀な行にマーク
    if res.get('_annual_r', 0) >= 0.50 and res.get('_max_dd', -1) >= -0.20:
        row_str += "  ★"
    print(row_str)
print("\n★ = 年率50%以上 かつ MaxDD -20%以内")

# ================================================================
# スパン別比較
# ================================================================
print("\n" + "=" * 60)
print("【スパン別 年率 比較】")
print("=" * 60)

span_header = f"{'方式':<22}" + "".join([f"{s:>10}" for s in spans.keys()])
print(span_header)
print("-" * (22 + 10 * len(spans)))

for name, ret in methods.items():
    row_str = f"{name:<22}"
    for span_name, sl in spans.items():
        sub = ret[sl]
        if len(sub) == 0:
            row_str += f"{'―':>10}"
            continue
        res = evaluate(sub)
        row_str += f"{res.get('年率','―'):>10}"
    print(row_str)

# ================================================================
# スパン別 MaxDD
# ================================================================
print("\n" + "=" * 60)
print("【スパン別 MaxDD 比較】")
print("=" * 60)

print(span_header)
print("-" * (22 + 10 * len(spans)))

for name, ret in methods.items():
    row_str = f"{name:<22}"
    for span_name, sl in spans.items():
        sub = ret[sl]
        if len(sub) == 0:
            row_str += f"{'―':>10}"
            continue
        res = evaluate(sub)
        row_str += f"{res.get('MaxDD','―'):>10}"
    print(row_str)

# ================================================================
# 年別リターン
# ================================================================
print("\n" + "=" * 60)
print("【年別リターン】")
print("=" * 60)

years = sorted(df.index.year.unique())
year_header = f"{'年':<6}" + "".join([f"{n[:8]:>12}" for n in methods.keys()])
print(year_header)
print("-" * (6 + 12 * len(methods)))

for year in years:
    row_str = f"{year:<6}"
    for name, ret in methods.items():
        r_year = ret[ret.index.year == year]
        if len(r_year) == 0:
            row_str += f"{'―':>12}"
            continue
        total = (1 + r_year).prod() - 1
        mark  = "+" if total >= 0 else ""
        row_str += f"{mark}{total:.1%}".rjust(12)
    print(row_str)

# ================================================================
# CSV出力
# ================================================================
result_df = pd.DataFrame(results_10y).T[cols].reset_index()
result_df.columns = ['方式'] + cols
out_csv = RESULTS_DIR / 'sizing_comparison.csv'
result_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    colors    = ['#333333', '#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    styles    = ['-', '--', '-.', ':', '-']
    widths    = [2.5, 1.8, 1.8, 1.8, 1.8]

    for i, (name, ret) in enumerate(methods.items()):
        cum = (1 + ret).cumprod() * 100
        ax1.plot(cum.index, cum.values,
                 label=name, color=colors[i],
                 linestyle=styles[i], linewidth=widths[i])

    ax1.set_title('サイジング方式比較 — 累積資産推移（100万円スタート）', fontsize=13)
    ax1.set_ylabel('資産（万円）')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_yscale('log')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True, alpha=0.3)

    for i, (name, ret) in enumerate(methods.items()):
        cum  = (1 + ret).cumprod()
        peak = cum.cummax()
        dd   = (cum - peak) / peak * 100
        ax2.plot(dd.index, dd.values,
                 label=name, color=colors[i],
                 linestyle=styles[i], linewidth=widths[i])

    ax2.axhline(y=-20, color='red', linestyle=':', alpha=0.5)
    ax2.set_title('ドローダウン推移（赤点線=-20%）', fontsize=13)
    ax2.set_ylabel('ドローダウン（%）')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = CHARTS_DIR / 'sizing_comparison.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ グラフ出力: {out_png}")

except Exception as e:
    print(f"⚠ グラフ出力失敗: {e}")

print("\n" + "=" * 60)
print("完了！")
print("=" * 60)