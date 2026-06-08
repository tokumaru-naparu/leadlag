# 20_filter_backtest.py
# H+戦略フィルター比較バックテスト
# 全シナリオを個別・組み合わせで検証し比較表を出力

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# パス設定
# ================================================================
BASE_DIR    = r"C:\Users\hg317\Desktop\projects\leadlag"
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
DATA_10Y    = os.path.join(SCRIPTS_DIR, 'data', 'history')
DATA_MAIN   = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# データ読み込み
# ================================================================
print("=" * 60)
print("データ読み込み中...")
print("=" * 60)

# analysis_data_for_opus.csv（10年・64列）を優先して探す
candidates = [
    os.path.join(SCRIPTS_DIR, 'data', 'analysis_data_for_opus.csv'),
    os.path.join(DATA_MAIN, 'analysis_data_for_opus.csv'),
]
csv_path = None
for c in candidates:
    if os.path.exists(c):
        tmp = pd.read_csv(c, nrows=5)
        if len(pd.read_csv(c)) >= 2000:
            csv_path = c
            break
if csv_path is None:
    # 行数が少なくてもとりあえず使う
    for c in candidates:
        if os.path.exists(c):
            csv_path = c
            break

if csv_path is None:
    raise FileNotFoundError("analysis_data_for_opus.csv が見つかりません")

df = pd.read_csv(csv_path, parse_dates=['date'])
df.set_index('date', inplace=True)
print(f"  ✅ {csv_path}")
print(f"     行数: {len(df)} 日 / 期間: {df.index.min().date()} 〜 {df.index.max().date()}")

# ================================================================
# 追加データ取得（VIX・ドル円）
# ================================================================
print("\n追加データ取得中（VIX・ドル円）...")

try:
    import yfinance as yf
    start = df.index.min().strftime('%Y-%m-%d')
    end   = df.index.max().strftime('%Y-%m-%d')

    vix_raw    = yf.download('^VIX',  start=start, end=end, progress=False)['Close']
    usdjpy_raw = yf.download('JPY=X', start=start, end=end, progress=False)['Close']
    sp500_raw  = yf.download('^GSPC', start=start, end=end, progress=False)['Close']

    # インデックスをtz-naiveに統一
    for s in [vix_raw, usdjpy_raw, sp500_raw]:
        if hasattr(s.index, 'tz') and s.index.tz is not None:
            s.index = s.index.tz_localize(None)

    df['vix']             = vix_raw.reindex(df.index).ffill()
    df['vix_prev']        = df['vix'].shift(1)
    df['vix_spike']       = (df['vix'] - df['vix_prev']) >= 5

    df['usdjpy']          = usdjpy_raw.reindex(df.index).ffill()
    df['usdjpy_change']   = df['usdjpy'].diff().abs()

    df['sp500_change_pct']= sp500_raw.pct_change().reindex(df.index).ffill() * 100

    print(f"  ✅ VIX取得: {df['vix'].notna().sum()}件")
    print(f"  ✅ ドル円取得: {df['usdjpy'].notna().sum()}件")
    print(f"  ✅ S&P500取得: {df['sp500_change_pct'].notna().sum()}件")
    HAS_VIX = True
    HAS_FX  = True

except Exception as e:
    print(f"  ⚠ 追加データ取得失敗（{e}）")
    print("  → B2, C3, C4 はスキップします")
    df['vix']             = np.nan
    df['vix_prev']        = np.nan
    df['vix_spike']       = False
    df['usdjpy_change']   = np.nan
    df['sp500_change_pct']= np.nan
    HAS_VIX = False
    HAS_FX  = False

# S&P500の代替: us_return_avg（既存データ）があれば使う
if 'us_return_avg' in df.columns and df['sp500_change_pct'].isna().all():
    df['sp500_change_pct'] = df['us_return_avg'] * 100
    print("  → S&P500代替: us_return_avg を使用（C2フィルター有効）")
    HAS_SP500_PROXY = True
else:
    HAS_SP500_PROXY = False

# ================================================================
# 前処理
# ================================================================
print("\nデータ検証・補完中...")

# signal_strength が全NaNの場合 → signals.csv から直接補完
sig_null_count = df['signal_strength'].isna().sum() if 'signal_strength' in df.columns else len(df)
print(f"  signal_strength NaN数: {sig_null_count} / {len(df)}")

if sig_null_count == len(df):
    print("  ⚠ signal_strength が全NaN → signals.csv から直接読み込みます")
    for sc in [os.path.join(DATA_10Y, 'signals.csv'),
               os.path.join(DATA_MAIN, 'history', 'signals.csv')]:
        if os.path.exists(sc):
            sig_df = pd.read_csv(sc, parse_dates=['date']).set_index('date')
            df['signal_strength'] = sig_df['signal_strength'].reindex(df.index)
            print(f"  ✅ signals.csv から補完: {df['signal_strength'].notna().sum()}件 ({sc})")
            break

# long_return / short_return が無い場合 → trades.csv から補完
if 'long_return' not in df.columns or df['long_return'].isna().all():
    print("  ⚠ long_return/short_return が無い → trades.csv から補完します")
    for tc in [os.path.join(DATA_10Y, 'trades.csv'),
               os.path.join(DATA_MAIN, 'history', 'trades.csv')]:
        if os.path.exists(tc):
            trd_df = pd.read_csv(tc, parse_dates=['date']).set_index('date')
            for c in ['long_return', 'short_return', 'strategy_return', 'is_correct']:
                if c in trd_df.columns:
                    df[c] = trd_df[c].reindex(df.index)
            print(f"  ✅ trades.csv から補完完了 ({tc})")
            break

# 最終確認
for col in ['signal_strength', 'long_return', 'short_return']:
    n = df[col].notna().sum() if col in df.columns else 0
    print(f"  {col}: {n}件有効")
    if n == 0:
        raise ValueError(f"❌ {col} が全てNaNです。データを確認してください。")

# シグナル分位（ランク基準で安定計算）
df['sig_rank'] = df['signal_strength'].rank(pct=True, na_option='keep')
def rank_to_quintile(r):
    if pd.isna(r): return 3
    if r <= 0.20: return 1
    if r <= 0.40: return 2
    if r <= 0.60: return 3
    if r <= 0.80: return 4
    return 5
df['sig_quintile'] = df['sig_rank'].apply(rank_to_quintile).astype(int)
print(f"  sig_quintile 分布: {df['sig_quintile'].value_counts().sort_index().to_dict()}")

# VIXスパイク後3日フラグ
df['days_since_vix_spike'] = 999
if HAS_VIX:
    spike_dates = df.index[df['vix_spike']]
    for sd in spike_dates:
        for d in range(1, 4):
            try:
                target = df.index[df.index.get_loc(sd) + d]
                df.loc[target, 'days_since_vix_spike'] = d
            except:
                pass

# 雇用統計（毎月第1金曜）のイベント日リスト
import calendar as cal_mod
event_dates = set()
for year in range(2016, 2027):
    for month in range(1, 13):
        for day in range(1, 8):
            try:
                d = pd.Timestamp(year, month, day)
                if d.dayofweek == 4:  # 金曜
                    event_dates.add(d)
                    break
            except:
                pass

# ================================================================
# H+フィルター統合関数
# ================================================================
def calc_hplus_filtered(df, params):
    """
    params辞書でフィルターのON/OFFとパラメータを制御する
    """
    returns = []
    peak_capital = 1.0
    capital = 1.0

    for date, row in df.iterrows():
        sig = row['signal_strength']
        q   = int(row['sig_quintile'])

        # --- ベースサイズ（A1: キャップ変更）---
        cap       = params.get('size_cap', 3.0)
        base_size = min(max(sig * 4, 0.3), cap)

        # --- ダンプナー（掛け算で重ねる）---
        dampener = 1.0

        # B1: 最弱シグナル帯
        if params.get('b1', False):
            if sig < 0.08:
                dampener *= 0.2

        # B2: 高VIX × 弱シグナル
        if params.get('b2', False) and HAS_VIX:
            vix = row.get('vix', np.nan)
            if not np.isnan(vix) and vix >= 20 and sig < 0.15:
                dampener *= 0.3

        # B3: LS弱（Q1/Q2）ダンプナー
        if params.get('b3', False):
            if q <= 2:
                dampener *= 0.5

        # C1: 米国イベント日（雇用統計等）
        if params.get('c1', False):
            if date in event_dates:
                dampener *= 0.5

        # C2: 夜間先物（S&P500前日変動で代替）
        if params.get('c2', False):
            sp_chg = row.get('sp500_change_pct', 0)
            if not np.isnan(sp_chg) and abs(sp_chg) > 2.0:
                dampener *= 0.5

        # C3: VIXスパイク後
        if params.get('c3', False) and HAS_VIX:
            if row.get('days_since_vix_spike', 999) <= 3:
                dampener *= 0.5

        # C4: ドル円急変動
        if params.get('c4', False) and HAS_FX:
            fx_chg = row.get('usdjpy_change', 0)
            if not np.isnan(fx_chg) and fx_chg > 1.5:
                dampener *= 0.5

        # DD動的縮小（おまけ）
        if params.get('dd_dynamic', False):
            dd_now = (capital - peak_capital) / peak_capital
            if dd_now <= -0.15:
                dampener *= 0.3
            elif dd_now <= -0.10:
                dampener *= 0.5

        final_size = base_size * dampener

        # --- リターン計算 ---
        amp    = params.get('short_amp', 1.3)
        ls_weak = (q <= 2)

        # B4: LS弱の日はQ4/Q5でもショート増幅しない
        if params.get('b4', False) and ls_weak:
            daily_r = row['long_return'] - row['short_return']
        elif q >= 4:
            daily_r = 0.7 * row['long_return'] - amp * row['short_return']
        else:
            daily_r = row['long_return'] - row['short_return']

        r = final_size * daily_r
        returns.append(r)

        # 資産追跡（DD動的縮小用）
        capital *= (1 + r)
        if capital > peak_capital:
            peak_capital = capital

    return pd.Series(returns, index=df.index)

# ================================================================
# 評価関数
# ================================================================
def evaluate(returns_series, label=''):
    s = returns_series.dropna()
    if len(s) == 0:
        return {}

    cumulative = (1 + s).cumprod()
    total_r    = cumulative.iloc[-1] - 1
    n_years    = len(s) / 252
    annual_r   = (1 + total_r) ** (1 / n_years) - 1
    annual_std = s.std() * np.sqrt(252)
    sharpe     = annual_r / annual_std if annual_std > 0 else 0

    peak   = cumulative.cummax()
    dd     = (cumulative - peak) / peak
    max_dd = dd.min()

    dd_end    = dd.idxmin()
    peak_date = cumulative[:dd_end].idxmax()
    try:
        dd_days = (dd_end - peak_date).days
    except:
        dd_days = 0

    win_rate = (s > 0).mean()
    wins   = s[s > 0]
    losses = s[s < 0]
    pf = wins.mean() / abs(losses.mean()) if len(losses) > 0 else np.inf
    calmar = annual_r / abs(max_dd) if max_dd != 0 else np.inf

    return {
        '年率':        f"{annual_r:.1%}",
        'Sharpe':      f"{sharpe:.2f}",
        'MaxDD':       f"{max_dd:.1%}",
        'DD日数':      dd_days,
        '勝率':        f"{win_rate:.1%}",
        '損益比':      f"{pf:.2f}",
        'Calmar':      f"{calmar:.2f}",
        '最終資産(万)': f"{cumulative.iloc[-1] * 100:.0f}",
        '_annual_r':   annual_r,
        '_max_dd':     max_dd,
        '_sharpe':     sharpe,
    }

# ================================================================
# シナリオ定義
# ================================================================
scenarios = {
    # ベースライン
    'H+ (現行)':           {'size_cap': 3.0, 'short_amp': 1.3},

    # A: リスク管理
    'A1: cap=2.0':         {'size_cap': 2.0, 'short_amp': 1.3},
    'A2: amp=1.15':        {'size_cap': 3.0, 'short_amp': 1.15},
    'A1+A2':               {'size_cap': 2.0, 'short_amp': 1.15},

    # B: データフィルター
    'B1: 弱sig×0.2':      {'size_cap': 3.0, 'short_amp': 1.3, 'b1': True},
    'B2: 高VIX弱sig×0.3': {'size_cap': 3.0, 'short_amp': 1.3, 'b2': True},
    'B3: LS弱×0.5':       {'size_cap': 3.0, 'short_amp': 1.3, 'b3': True},
    'B4: LS弱増幅無効':    {'size_cap': 3.0, 'short_amp': 1.3, 'b4': True},

    # C: 構造フィルター
    'C1: イベント日×0.5': {'size_cap': 3.0, 'short_amp': 1.3, 'c1': True},
    'C2: 夜間先物×0.5':   {'size_cap': 3.0, 'short_amp': 1.3, 'c2': True},
    'C3: VIX急騰×0.5':    {'size_cap': 3.0, 'short_amp': 1.3, 'c3': True},
    'C4: 為替急変×0.5':   {'size_cap': 3.0, 'short_amp': 1.3, 'c4': True},

    # 組み合わせ
    'A1+A2+B全部':         {'size_cap': 2.0, 'short_amp': 1.15,
                            'b1': True, 'b2': True, 'b3': True, 'b4': True},
    'A1+A2+C全部':         {'size_cap': 2.0, 'short_amp': 1.15,
                            'c1': True, 'c2': True, 'c3': True, 'c4': True},
    'A+B+C全部':           {'size_cap': 2.0, 'short_amp': 1.15,
                            'b1': True, 'b2': True, 'b3': True, 'b4': True,
                            'c1': True, 'c2': True, 'c3': True, 'c4': True},

    # おまけ: DD動的縮小
    'A1+A2+DD動的':        {'size_cap': 2.0, 'short_amp': 1.15, 'dd_dynamic': True},
    'A+B+C+DD動的':        {'size_cap': 2.0, 'short_amp': 1.15,
                            'b1': True, 'b2': True, 'b3': True, 'b4': True,
                            'c1': True, 'c2': True, 'c3': True, 'c4': True,
                            'dd_dynamic': True},
}

# ================================================================
# スパン定義
# ================================================================
spans = {
    '10Y': (None, None),
    '5Y':  ('2021-04-20', None),
    '3Y':  ('2023-04-11', None),
    '1Y':  ('2025-03-28', None),
}

# ================================================================
# バックテスト実行
# ================================================================
print("\n" + "=" * 60)
print("バックテスト実行中...")
print("=" * 60)

# 全スパン結果格納
results_10y = {}
results_span = {k: {} for k in spans}
all_returns  = {}  # 累積グラフ用

for name, params in scenarios.items():
    print(f"  {name}...")

    # 10年全体
    ret = calc_hplus_filtered(df, params)
    results_10y[name] = evaluate(ret, name)
    all_returns[name] = ret

    # 各スパン
    for span_name, (start, end) in spans.items():
        sub = df[start:end] if start else df
        r   = calc_hplus_filtered(sub, params)
        results_span[span_name][name] = evaluate(r)

print("  ✅ 完了")

# ================================================================
# 比較表出力（10年）
# ================================================================
print("\n" + "=" * 60)
print("【10年 全シナリオ比較表】")
print("=" * 60)

cols = ['年率', 'Sharpe', 'MaxDD', 'DD日数', '勝率', '損益比', 'Calmar', '最終資産(万)']
header = f"{'シナリオ':<22}" + "".join([f"{c:>12}" for c in cols])
print(header)
print("-" * (22 + 12 * len(cols)))

for name, res in results_10y.items():
    if not res:
        continue
    row_str = f"{name:<22}" + "".join([f"{res.get(c,'―'):>12}" for c in cols])

    # 優秀な行にマーク
    try:
        ar = res['_annual_r']
        md = res['_max_dd']
        if ar >= 0.45 and md >= -0.20:
            row_str += "  ★"
    except:
        pass
    print(row_str)

print("\n★ = 年率45%以上 かつ MaxDD -20%以内")

# ================================================================
# スパン別MaxDD比較
# ================================================================
print("\n" + "=" * 60)
print("【スパン別 MaxDD 比較】")
print("=" * 60)

span_header = f"{'シナリオ':<22}" + "".join([f"{s:>10}" for s in spans.keys()])
print(span_header)
print("-" * (22 + 10 * len(spans)))

for name in scenarios.keys():
    row_str = f"{name:<22}"
    for span_name in spans.keys():
        val = results_span[span_name].get(name, {}).get('MaxDD', '―')
        row_str += f"{val:>10}"
    print(row_str)

# ================================================================
# 年別リターン（H+ vs 推奨上位）
# ================================================================
print("\n" + "=" * 60)
print("【年別リターン比較】")
print("=" * 60)

compare_scenarios = ['H+ (現行)', 'A1+A2', 'A1+A2+B全部', 'A+B+C全部', 'A1+A2+DD動的']
years = sorted(df.index.year.unique())

year_header = f"{'年':<6}" + "".join([f"{s[:12]:>14}" for s in compare_scenarios])
print(year_header)
print("-" * (6 + 14 * len(compare_scenarios)))

for year in years:
    row_str = f"{year:<6}"
    mask = df.index.year == year
    for name in compare_scenarios:
        if name not in all_returns:
            row_str += f"{'―':>14}"
            continue
        r_year = all_returns[name][mask]
        total  = (1 + r_year).prod() - 1
        mark   = "+" if total >= 0 else ""
        row_str += f"{mark}{total:.1%}".rjust(14)
    print(row_str)

# ================================================================
# MaxDD発生時期
# ================================================================
print("\n" + "=" * 60)
print("【MaxDD 発生時期】")
print("=" * 60)

for name in compare_scenarios:
    if name not in all_returns:
        continue
    r = all_returns[name]
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    dd_end = dd.idxmin()
    peak_date = cum[:dd_end].idxmax()
    max_dd_val = dd.min()
    print(f"  {name:<22} MaxDD: {max_dd_val:.1%}  "
          f"（{peak_date.date()} → {dd_end.date()}）")

# ================================================================
# CSV出力
# ================================================================
result_df = pd.DataFrame(results_10y).T[cols].reset_index()
result_df.columns = ['シナリオ'] + cols
out_path = os.path.join(OUTPUT_DIR, 'filter_backtest_results.csv')
result_df.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n✅ CSV出力: {out_path}")

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

    colors = ['#333333', '#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    graph_scenarios = ['H+ (現行)', 'A1+A2', 'A1+A2+B全部', 'A+B+C全部', 'A1+A2+DD動的']

    # 上段: 累積資産推移
    for i, name in enumerate(graph_scenarios):
        if name not in all_returns:
            continue
        cum = (1 + all_returns[name]).cumprod() * 100
        ax1.plot(cum.index, cum.values, label=name,
                 color=colors[i % len(colors)],
                 linewidth=1.5 if i > 0 else 2.5,
                 linestyle='-' if i == 0 else ['--', '-.', ':', '-', '--'][i % 5])

    ax1.set_title('H+フィルター比較 — 累積資産推移（100万円スタート）', fontsize=13)
    ax1.set_ylabel('資産（万円）')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 下段: ドローダウン
    for i, name in enumerate(graph_scenarios):
        if name not in all_returns:
            continue
        cum = (1 + all_returns[name]).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak * 100
        ax2.plot(dd.index, dd.values, label=name,
                 color=colors[i % len(colors)],
                 linewidth=1.5 if i > 0 else 2.5,
                 linestyle='-' if i == 0 else ['--', '-.', ':', '-', '--'][i % 5])

    ax2.set_title('ドローダウン推移', fontsize=13)
    ax2.set_ylabel('ドローダウン（%）')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=-20, color='red', linestyle=':', alpha=0.5, label='-20%ライン')

    plt.tight_layout()
    graph_path = os.path.join(OUTPUT_DIR, 'filter_backtest_chart.png')
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ グラフ出力: {graph_path}")

except Exception as e:
    print(f"⚠ グラフ出力失敗: {e}")

print("\n" + "=" * 60)
print("完了！")
print("=" * 60)
print(f"  比較表CSV: {os.path.join(OUTPUT_DIR, 'filter_backtest_results.csv')}")
print(f"  グラフ:    {os.path.join(OUTPUT_DIR, 'filter_backtest_chart.png')}")