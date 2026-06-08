#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21_filter_grid_backtest.py
H+戦略 フィルターグリッドサーチ

cap / amp / DD動的 / イベント日 / Bフィルターを細かく振って
年率とMaxDDのトレードオフを可視化する。

使い方:
  cd C:/Users/hg317/Desktop/projects/leadlag
  python scripts/21_filter_grid_backtest.py
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import itertools
import calendar as cal_mod
from datetime import datetime

warnings.filterwarnings('ignore')

# ================================================================
# パス設定（プロジェクトルートから自動解決）
# ================================================================
# スクリプトの場所から親ディレクトリを遡ってルートを推定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def detect_project_root(start_dir):
    cur = start_dir
    while True:
        # このリポジトリ固有ファイルがあれば最優先で採用
        if os.path.exists(os.path.join(cur, 'leadlag_CLAUDE.md')):
            return cur

        # 汎用フォールバック: scripts と data が揃っていればルート候補
        if (os.path.isdir(os.path.join(cur, 'scripts')) and
                os.path.isdir(os.path.join(cur, 'data'))):
            return cur

        parent = os.path.dirname(cur)
        if parent == cur:
            # 見つからなければ開始位置を返す
            return start_dir
        cur = parent


ROOT = detect_project_root(SCRIPT_DIR)

# 各パス
DATA_DIR        = os.path.join(ROOT, 'data')
DATA_RAW_DIR    = os.path.join(DATA_DIR, 'raw')
DATA_PROC_DIR   = os.path.join(DATA_DIR, 'processed')
DATA_EXT_DIR    = os.path.join(DATA_DIR, 'external')
HISTORY_DIR     = os.path.join(DATA_DIR, 'history')
SCRIPTS_DATA    = os.path.join(ROOT, 'scripts', 'data')
HISTORY_10Y     = os.path.join(SCRIPTS_DATA, 'history')
OUTPUT_DIR      = os.path.join(ROOT, 'output')

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROC_DIR, exist_ok=True)
os.makedirs(DATA_EXT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pick_best_csv(paths):
    """存在する候補のうち行数が最大のCSVを返す。"""
    best_path = None
    best_rows = -1
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            rows = len(pd.read_csv(p))
        except Exception:
            rows = -1
        if rows > best_rows:
            best_rows = rows
            best_path = p
    return best_path

print("=" * 70)
print("H+ フィルター グリッドサーチ バックテスト")
print("=" * 70)
print(f"  ROOT: {ROOT}")

# ================================================================
# 1. メインデータ読み込み
# ================================================================
print("\n[1] データ読み込み...")

# analysis_data_for_opus.csv / analysis_data.csv を探す
ANALYSIS_CANDIDATES = [
    os.path.join(DATA_PROC_DIR, 'analysis_data.csv'),
    os.path.join(DATA_PROC_DIR, 'analysis_data_for_opus.csv'),
    os.path.join(DATA_DIR, 'analysis_data_for_opus.csv'),
    os.path.join(SCRIPTS_DATA, 'analysis_data_for_opus.csv'),
]
csv_path = pick_best_csv(ANALYSIS_CANDIDATES)

if csv_path is None:
    print("  ❌ analysis_data_for_opus.csv / analysis_data.csv が見つかりません")
    print("     探した場所:")
    for c in ANALYSIS_CANDIDATES:
        print(f"       {c}")
    sys.exit(1)

df = pd.read_csv(csv_path, parse_dates=['date'])
df.set_index('date', inplace=True)
print(f"  ✅ {csv_path}")
print(f"     {len(df)}日 / {df.index.min().date()} 〜 {df.index.max().date()}")

# ================================================================
# 2. signal_strength / long_return / short_return の補完
# ================================================================
print("\n[2] データ補完...")

# --- signal_strength ---
if 'signal_strength' not in df.columns or df['signal_strength'].isna().all():
    print("  signal_strength が無い → signals.csv から補完")
    for sp in [os.path.join(DATA_PROC_DIR, 'signals.csv'),
               os.path.join(HISTORY_10Y, 'signals.csv'),
               os.path.join(HISTORY_DIR, 'signals.csv')]:
        if os.path.exists(sp):
            sig = pd.read_csv(sp, parse_dates=['date']).set_index('date')
            df['signal_strength'] = sig['signal_strength'].reindex(df.index)
            print(f"  ✅ {sp} → {df['signal_strength'].notna().sum()}件")
            break

# --- long_return / short_return ---
if 'long_return' not in df.columns or df['long_return'].isna().all():
    print("  long_return が無い → trades.csv から補完")
    for tp in [os.path.join(DATA_PROC_DIR, 'trades.csv'),
               os.path.join(HISTORY_10Y, 'trades.csv'),
               os.path.join(HISTORY_DIR, 'trades.csv')]:
        if os.path.exists(tp):
            trd = pd.read_csv(tp, parse_dates=['date']).set_index('date')
            for col in ['long_return', 'short_return', 'strategy_return']:
                if col in trd.columns:
                    df[col] = trd[col].reindex(df.index)
            print(f"  ✅ {tp}")
            break

# 検証
for col in ['signal_strength', 'long_return', 'short_return']:
    n = df[col].notna().sum() if col in df.columns else 0
    print(f"  {col}: {n}件")
    if n == 0:
        print(f"  ❌ {col} が全NaN。中断します。")
        sys.exit(1)

# NaN行を除外
df = df.dropna(subset=['signal_strength', 'long_return', 'short_return'])
print(f"  有効行数: {len(df)}")

# sig_quintile（ランクベース）
df['sig_quintile'] = pd.qcut(df['signal_strength'], 5, labels=[1,2,3,4,5]).astype(int)
print(f"  sig_quintile: {df['sig_quintile'].value_counts().sort_index().to_dict()}")

# ================================================================
# 3. 外部データ取得（VIX / ドル円 / S&P500）
# ================================================================
print("\n[3] 外部データ取得...")

HAS_VIX = False
HAS_FX  = False
HAS_SP  = False

# --- 方法1: ローカルCSVから読み込み ---
local_files = {
    'vix':    [os.path.join(DATA_EXT_DIR, 'vix.csv'),
               os.path.join(SCRIPTS_DATA, '^VIX.csv'),
               os.path.join(DATA_DIR, '^VIX.csv'),
               os.path.join(ROOT, '^VIX.csv')],
    'usdjpy': [os.path.join(DATA_EXT_DIR, 'usdjpy.csv'),
               os.path.join(SCRIPTS_DATA, 'JPY=X.csv'),
               os.path.join(DATA_DIR, 'JPY=X.csv'),
               os.path.join(ROOT, 'JPY=X.csv')],
    'sp500':  [os.path.join(DATA_EXT_DIR, 'sp500.csv'),
               os.path.join(SCRIPTS_DATA, '^GSPC.csv'),
               os.path.join(DATA_DIR, '^GSPC.csv'),
               os.path.join(ROOT, '^GSPC.csv')],
}

def load_local_csv(name, paths):
    for p in paths:
        if os.path.exists(p):
            tmp = pd.read_csv(p, parse_dates=['Date'])
            if 'Close' in tmp.columns:
                tmp.set_index('Date', inplace=True)
                # MultiIndex対応（yfinanceの新形式）
                if isinstance(tmp.columns, pd.MultiIndex):
                    tmp.columns = tmp.columns.get_level_values(0)
                print(f"  ✅ {name}: ローカル {p} ({len(tmp)}件)")
                return tmp['Close']
    return None

vix_raw    = load_local_csv('VIX', local_files['vix'])
usdjpy_raw = load_local_csv('USDJPY', local_files['usdjpy'])
sp500_raw  = load_local_csv('SP500', local_files['sp500'])

# --- 方法2: yfinanceで取得（ローカルに無い場合）---
if vix_raw is None or usdjpy_raw is None or sp500_raw is None:
    print("  ローカルCSVが不足 → yfinance で取得を試みます...")
    try:
        import yfinance as yf
        start = df.index.min().strftime('%Y-%m-%d')
        end   = (df.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        if vix_raw is None:
            try:
                tmp = yf.download('^VIX', start=start, end=end, progress=False)
                if len(tmp) > 0:
                    if isinstance(tmp.columns, pd.MultiIndex):
                        tmp.columns = tmp.columns.get_level_values(0)
                    vix_raw = tmp['Close']
                    if hasattr(vix_raw.index, 'tz') and vix_raw.index.tz:
                        vix_raw.index = vix_raw.index.tz_localize(None)
                    print(f"  ✅ VIX: yfinance ({len(vix_raw)}件)")
                    # ローカルにも保存
                    tmp.to_csv(os.path.join(DATA_EXT_DIR, 'vix.csv'))
            except Exception as e:
                print(f"  ⚠ VIX取得失敗: {e}")

        if usdjpy_raw is None:
            try:
                tmp = yf.download('JPY=X', start=start, end=end, progress=False)
                if len(tmp) > 0:
                    if isinstance(tmp.columns, pd.MultiIndex):
                        tmp.columns = tmp.columns.get_level_values(0)
                    usdjpy_raw = tmp['Close']
                    if hasattr(usdjpy_raw.index, 'tz') and usdjpy_raw.index.tz:
                        usdjpy_raw.index = usdjpy_raw.index.tz_localize(None)
                    print(f"  ✅ USDJPY: yfinance ({len(usdjpy_raw)}件)")
                    tmp.to_csv(os.path.join(DATA_EXT_DIR, 'usdjpy.csv'))
            except Exception as e:
                print(f"  ⚠ USDJPY取得失敗: {e}")

        if sp500_raw is None:
            try:
                tmp = yf.download('^GSPC', start=start, end=end, progress=False)
                if len(tmp) > 0:
                    if isinstance(tmp.columns, pd.MultiIndex):
                        tmp.columns = tmp.columns.get_level_values(0)
                    sp500_raw = tmp['Close']
                    if hasattr(sp500_raw.index, 'tz') and sp500_raw.index.tz:
                        sp500_raw.index = sp500_raw.index.tz_localize(None)
                    print(f"  ✅ SP500: yfinance ({len(sp500_raw)}件)")
                    tmp.to_csv(os.path.join(DATA_EXT_DIR, 'sp500.csv'))
            except Exception as e:
                print(f"  ⚠ SP500取得失敗: {e}")

    except ImportError:
        print("  ⚠ yfinance がインストールされていません")
        print("    pip install yfinance で入れてください")

# --- dfに結合 ---
if vix_raw is not None and len(vix_raw) > 100:
    df['vix']       = vix_raw.reindex(df.index).ffill()
    df['vix_prev']  = df['vix'].shift(1)
    df['vix_spike'] = (df['vix'] - df['vix_prev']) >= 5
    HAS_VIX = True
    print(f"  VIX有効: {df['vix'].notna().sum()}件")
else:
    df['vix']       = np.nan
    df['vix_prev']  = np.nan
    df['vix_spike'] = False
    print("  ⚠ VIXデータなし → B2, C3 スキップ")

if usdjpy_raw is not None and len(usdjpy_raw) > 100:
    df['usdjpy']        = usdjpy_raw.reindex(df.index).ffill()
    df['usdjpy_change'] = df['usdjpy'].diff().abs()
    HAS_FX = True
    print(f"  ドル円有効: {df['usdjpy'].notna().sum()}件")
else:
    df['usdjpy_change'] = np.nan
    print("  ⚠ ドル円データなし → C4 スキップ")

if sp500_raw is not None and len(sp500_raw) > 100:
    df['sp500_chg_pct'] = sp500_raw.pct_change().reindex(df.index).ffill() * 100
    HAS_SP = True
    print(f"  S&P500有効: {df['sp500_chg_pct'].notna().sum()}件")
else:
    # 代替: us_return_avg があれば使う
    if 'us_return_avg' in df.columns:
        df['sp500_chg_pct'] = df['us_return_avg'] * 100
        HAS_SP = True
        print("  S&P500代替: us_return_avg 使用")
    else:
        df['sp500_chg_pct'] = np.nan
        print("  ⚠ S&P500データなし → C2 スキップ")

# VIXスパイク後3日フラグ
df['days_since_spike'] = 999
if HAS_VIX:
    spike_idx = df.index[df['vix_spike'].fillna(False)]
    for sd in spike_idx:
        loc = df.index.get_loc(sd)
        for d in range(1, 4):
            if loc + d < len(df):
                df.iloc[loc + d, df.columns.get_loc('days_since_spike')] = d

# ================================================================
# 4. イベントカレンダー（雇用統計: 毎月第1金曜）
# ================================================================
print("\n[4] イベントカレンダー生成...")

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
print(f"  雇用統計日: {len(event_dates)}日")

# ================================================================
# 5. バックテストエンジン
# ================================================================

def run_backtest(df, cap, amp, b1=False, b3=False, c1=False, c2=False,
                 b2=False, c3=False, c4=False, dd_dynamic=False,
                 dd_thresh1=-0.10, dd_scale1=0.5,
                 dd_thresh2=-0.15, dd_scale2=0.3):
    """
    H+戦略にフィルターを適用してリターン系列を返す。
    
    パラメータ:
      cap: サイズ上限 (2.0〜3.0)
      amp: ショート増幅率 (1.0〜1.3)
      b1:  弱シグナル(< 0.08)サイズ×0.2
      b2:  高VIX×弱シグナル サイズ×0.3
      b3:  sig_quintile<=2 サイズ×0.5
      c1:  雇用統計日 サイズ×0.5
      c2:  S&P500前日±2%超 サイズ×0.5
      c3:  VIXスパイク後3日 サイズ×0.5
      c4:  ドル円前日±1.5円超 サイズ×0.5
      dd_dynamic: DD動的縮小ON/OFF
      dd_thresh1/2, dd_scale1/2: DD縮小パラメータ
    """
    n = len(df)
    returns = np.empty(n)
    capital = 1.0
    peak    = 1.0

    sig_arr   = df['signal_strength'].values
    q_arr     = df['sig_quintile'].values
    lr_arr    = df['long_return'].values
    sr_arr    = df['short_return'].values
    vix_arr   = df['vix'].values if HAS_VIX else np.full(n, np.nan)
    sp_arr    = df['sp500_chg_pct'].values if HAS_SP else np.full(n, 0.0)
    fx_arr    = df['usdjpy_change'].values if HAS_FX else np.full(n, 0.0)
    spike_arr = df['days_since_spike'].values
    dates     = df.index

    for i in range(n):
        sig = sig_arr[i]
        q   = q_arr[i]

        # ベースサイズ
        base_size = min(max(sig * 4, 0.3), cap)

        # ダンプナー
        damp = 1.0

        if b1 and sig < 0.08:
            damp *= 0.2

        if b2 and HAS_VIX:
            v = vix_arr[i]
            if not np.isnan(v) and v >= 20 and sig < 0.15:
                damp *= 0.3

        if b3 and q <= 2:
            damp *= 0.5

        if c1 and dates[i] in event_dates:
            damp *= 0.5

        if c2 and HAS_SP:
            sp = sp_arr[i]
            if not np.isnan(sp) and abs(sp) > 2.0:
                damp *= 0.5

        if c3 and HAS_VIX:
            if spike_arr[i] <= 3:
                damp *= 0.5

        if c4 and HAS_FX:
            fx = fx_arr[i]
            if not np.isnan(fx) and fx > 1.5:
                damp *= 0.5

        if dd_dynamic:
            dd_now = (capital - peak) / peak if peak > 0 else 0
            if dd_now <= dd_thresh2:
                damp *= dd_scale2
            elif dd_now <= dd_thresh1:
                damp *= dd_scale1

        final_size = base_size * damp

        # リターン計算
        if q >= 4:
            daily_r = 0.7 * lr_arr[i] - amp * sr_arr[i]
        else:
            daily_r = lr_arr[i] - sr_arr[i]

        r = final_size * daily_r
        returns[i] = r

        capital *= (1 + r)
        if capital > peak:
            peak = capital

    return pd.Series(returns, index=df.index)


def evaluate(ret):
    """リターン系列 → 評価指標辞書"""
    s = ret.dropna()
    if len(s) == 0:
        return {}
    cum = (1 + s).cumprod()
    total = cum.iloc[-1] - 1
    ny = len(s) / 252
    ar = (1 + total) ** (1/ny) - 1
    std = s.std() * np.sqrt(252)
    sharpe = ar / std if std > 0 else 0

    pk = cum.cummax()
    dd = (cum - pk) / pk
    mdd = dd.min()

    dd_end = dd.idxmin()
    pk_date = cum[:dd_end].idxmax()
    try:
        dd_days = (dd_end - pk_date).days
    except:
        dd_days = 0

    wr = (s > 0).mean()
    w = s[s > 0]
    l = s[s < 0]
    pf = w.mean() / abs(l.mean()) if len(l) > 0 else np.inf
    calmar = ar / abs(mdd) if mdd != 0 else np.inf

    return {
        'annual_r': ar,
        'sharpe': sharpe,
        'max_dd': mdd,
        'dd_days': dd_days,
        'win_rate': wr,
        'profit_factor': pf,
        'calmar': calmar,
        'final_capital': cum.iloc[-1] * 100,  # 万円
        'dd_start': pk_date,
        'dd_end': dd_end,
    }


# ================================================================
# 6. シナリオ定義
# ================================================================
print("\n[5] シナリオ生成...")

scenarios = {}

# ------ パートA: cap × amp グリッド（メイン探索）------
caps = [2.0, 2.3, 2.5, 2.7, 3.0]
amps = [1.0, 1.1, 1.15, 1.2, 1.25, 1.3]

for c in caps:
    for a in amps:
        name = f"cap={c:.1f}_amp={a:.2f}"
        scenarios[name] = dict(cap=c, amp=a)

# ------ パートB: cap×amp + DD動的 ------
for c in [2.0, 2.3, 2.5, 3.0]:
    for a in [1.0, 1.15, 1.2, 1.3]:
        name = f"cap={c:.1f}_amp={a:.2f}_DD"
        scenarios[name] = dict(cap=c, amp=a, dd_dynamic=True)

# ------ パートC: cap×amp + イベント日(C1) ------
for c in [2.0, 2.3, 2.5, 3.0]:
    for a in [1.0, 1.15, 1.2, 1.3]:
        name = f"cap={c:.1f}_amp={a:.2f}_C1"
        scenarios[name] = dict(cap=c, amp=a, c1=True)

# ------ パートD: cap×amp + B1(弱sig) + B3(LS弱) ------
for c in [2.0, 2.5, 3.0]:
    for a in [1.15, 1.2, 1.3]:
        name = f"cap={c:.1f}_amp={a:.2f}_B1B3"
        scenarios[name] = dict(cap=c, amp=a, b1=True, b3=True)

# ------ パートE: フル組み合わせ（外部データあれば）------
for c in [2.0, 2.3, 2.5]:
    for a in [1.1, 1.15, 1.2]:
        # C全部
        name = f"cap={c:.1f}_amp={a:.2f}_Call"
        scenarios[name] = dict(cap=c, amp=a, c1=True, c2=True, c3=True, c4=True)
        # B+C全部
        name = f"cap={c:.1f}_amp={a:.2f}_BC"
        scenarios[name] = dict(cap=c, amp=a,
                               b1=True, b2=True, b3=True,
                               c1=True, c2=True, c3=True, c4=True)
        # B+C+DD
        name = f"cap={c:.1f}_amp={a:.2f}_BC_DD"
        scenarios[name] = dict(cap=c, amp=a,
                               b1=True, b2=True, b3=True,
                               c1=True, c2=True, c3=True, c4=True,
                               dd_dynamic=True)

# ------ ベースライン（現行H+）------
scenarios['H+_現行'] = dict(cap=3.0, amp=1.3)

total = len(scenarios)
print(f"  シナリオ数: {total}")

# ================================================================
# 7. バックテスト実行
# ================================================================
print(f"\n[6] バックテスト実行中... ({total}シナリオ)")

results = {}
all_returns = {}

for i, (name, params) in enumerate(scenarios.items()):
    if (i+1) % 20 == 0 or i == 0:
        print(f"  {i+1}/{total}...")

    ret = run_backtest(df, **params)
    ev  = evaluate(ret)
    results[name] = ev
    all_returns[name] = ret

print(f"  ✅ 完了")

# ================================================================
# 8. 結果表作成・ソート
# ================================================================
print("\n[7] 結果集計...")

rows = []
for name, ev in results.items():
    if not ev:
        continue
    rows.append({
        'シナリオ': name,
        '年率': ev['annual_r'],
        'Sharpe': ev['sharpe'],
        'MaxDD': ev['max_dd'],
        'DD日数': ev['dd_days'],
        '勝率': ev['win_rate'],
        '損益比': ev['profit_factor'],
        'Calmar': ev['calmar'],
        '最終資産(万)': ev['final_capital'],
    })

result_df = pd.DataFrame(rows)

# Calmar（年率÷|MaxDD|）でソート = リターンとDDのバランス指標
result_df.sort_values('Calmar', ascending=False, inplace=True)
result_df.reset_index(drop=True, inplace=True)

# ================================================================
# 9. コンソール出力（TOP30 + H+現行）
# ================================================================
print("\n" + "=" * 100)
print("【Calmar比 上位30シナリオ + H+現行】")
print("=" * 100)

header = (f"{'#':>3} {'シナリオ':<35} {'年率':>7} {'Sharpe':>7} "
          f"{'MaxDD':>8} {'DD日':>5} {'勝率':>6} {'損益比':>6} "
          f"{'Calmar':>7} {'資産(万)':>10}")
print(header)
print("-" * 100)

# TOP30
shown = set()
for idx, row in result_df.head(30).iterrows():
    ar = row['年率']
    md = row['MaxDD']
    mark = "★" if ar >= 0.45 and md >= -0.20 else " "
    print(f"{idx+1:>3} {row['シナリオ']:<35} {ar:>6.1%} {row['Sharpe']:>7.2f} "
          f"{md:>7.1%} {row['DD日数']:>5.0f} {row['勝率']:>5.1%} "
          f"{row['損益比']:>6.2f} {row['Calmar']:>7.2f} {row['最終資産(万)']:>9.0f} {mark}")
    shown.add(row['シナリオ'])

# H+現行（TOP30に入ってなければ追加表示）
if 'H+_現行' not in shown:
    row_h = result_df[result_df['シナリオ'] == 'H+_現行'].iloc[0]
    rank = result_df[result_df['シナリオ'] == 'H+_現行'].index[0] + 1
    print("-" * 100)
    print(f"{rank:>3} {row_h['シナリオ']:<35} {row_h['年率']:>6.1%} "
          f"{row_h['Sharpe']:>7.2f} {row_h['MaxDD']:>7.1%} "
          f"{row_h['DD日数']:>5.0f} {row_h['勝率']:>5.1%} "
          f"{row_h['損益比']:>6.2f} {row_h['Calmar']:>7.2f} "
          f"{row_h['最終資産(万)']:>9.0f}  (現行)")

print("\n★ = 年率45%以上 かつ MaxDD -20%以内")

# ================================================================
# 10. パレートフロンティア（年率 vs MaxDD トレードオフ）
# ================================================================
print("\n" + "=" * 100)
print("【パレートフロンティア: 年率 vs MaxDD の最適バランス上位】")
print("=" * 100)

# パレート最適 = 同じMaxDDで最も年率が高い or 同じ年率で最もMaxDDが浅い
pareto_df = result_df.copy()
pareto_df.sort_values('年率', ascending=False, inplace=True)

pareto = []
best_dd = -1.0  # 最悪のDD
for _, row in pareto_df.iterrows():
    if row['MaxDD'] > best_dd:
        pareto.append(row)
        best_dd = row['MaxDD']

pareto_df = pd.DataFrame(pareto)
print(f"\nパレート最適: {len(pareto_df)}シナリオ\n")

for _, row in pareto_df.iterrows():
    ar = row['年率']
    md = row['MaxDD']
    mark = "★" if ar >= 0.45 and md >= -0.20 else " "
    print(f"  {row['シナリオ']:<40} 年率:{ar:>6.1%}  MaxDD:{md:>7.1%}  "
          f"Sharpe:{row['Sharpe']:>5.2f}  Calmar:{row['Calmar']:>5.2f} {mark}")

# ================================================================
# 11. MaxDD発生時期（上位5シナリオ + H+現行）
# ================================================================
print("\n" + "=" * 70)
print("【MaxDD 発生時期】")
print("=" * 70)

top5 = list(result_df.head(5)['シナリオ'])
if 'H+_現行' not in top5:
    top5.append('H+_現行')

for name in top5:
    ev = results.get(name, {})
    if ev:
        print(f"  {name:<40} MaxDD: {ev['max_dd']:.1%}  "
              f"({ev['dd_start'].date()} → {ev['dd_end'].date()})")

# ================================================================
# 12. 年別リターン比較
# ================================================================
print("\n" + "=" * 100)
print("【年別リターン比較（Calmar上位3 + H+現行）】")
print("=" * 100)

compare = list(result_df.head(3)['シナリオ']) + ['H+_現行']
compare = list(dict.fromkeys(compare))  # 重複排除

years = sorted(df.index.year.unique())
header = f"{'年':<6}" + "".join([f"{n[:20]:>22}" for n in compare])
print(header)
print("-" * (6 + 22 * len(compare)))

for yr in years:
    mask = df.index.year == yr
    row_str = f"{yr:<6}"
    for name in compare:
        if name in all_returns:
            r_yr = all_returns[name][mask]
            total = (1 + r_yr).prod() - 1
            row_str += f"{'+'if total>=0 else ''}{total:.1%}".rjust(22)
        else:
            row_str += f"{'—':>22}"
    print(row_str)

# ================================================================
# 13. CSV出力
# ================================================================
out_csv = os.path.join(OUTPUT_DIR, 'filter_grid_results.csv')

export_df = result_df.copy()
export_df['年率']     = export_df['年率'].map(lambda x: f"{x:.2%}")
export_df['Sharpe']   = export_df['Sharpe'].map(lambda x: f"{x:.3f}")
export_df['MaxDD']    = export_df['MaxDD'].map(lambda x: f"{x:.2%}")
export_df['勝率']     = export_df['勝率'].map(lambda x: f"{x:.2%}")
export_df['損益比']   = export_df['損益比'].map(lambda x: f"{x:.3f}")
export_df['Calmar']   = export_df['Calmar'].map(lambda x: f"{x:.3f}")
export_df['最終資産(万)'] = export_df['最終資産(万)'].map(lambda x: f"{x:.0f}")

export_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n✅ CSV: {out_csv}")

# ================================================================
# 14. グラフ出力
# ================================================================
print("\n[8] グラフ生成...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # 日本語フォント（なければ英語フォールバック）
    for font in ['MS Gothic', 'Yu Gothic', 'Hiragino Sans', 'IPAGothic']:
        try:
            plt.rcParams['font.family'] = font
            break
        except:
            pass

    # --- グラフ1: 累積資産 + ドローダウン ---
    graph_names = list(result_df.head(3)['シナリオ']) + ['H+_現行']
    graph_names = list(dict.fromkeys(graph_names))

    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for i, name in enumerate(graph_names):
        if name not in all_returns:
            continue
        cum = (1 + all_returns[name]).cumprod() * 100
        lw = 2.5 if name == 'H+_現行' else 1.5
        ls = '-' if name == 'H+_現行' else ['--', '-.', ':', '-'][i % 4]
        ax1.plot(cum.index, cum.values, label=name, color=colors[i % len(colors)],
                 linewidth=lw, linestyle=ls)

    ax1.set_title('H+ Filter Grid — Cumulative Asset (100万円 start)', fontsize=13)
    ax1.set_ylabel('Asset (万円)')
    ax1.set_yscale('log')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    for i, name in enumerate(graph_names):
        if name not in all_returns:
            continue
        cum = (1 + all_returns[name]).cumprod()
        pk  = cum.cummax()
        dd  = (cum - pk) / pk * 100
        lw = 2.5 if name == 'H+_現行' else 1.5
        ls = '-' if name == 'H+_現行' else ['--', '-.', ':', '-'][i % 4]
        ax2.plot(dd.index, dd.values, label=name, color=colors[i % len(colors)],
                 linewidth=lw, linestyle=ls)

    ax2.axhline(-20, color='red', linestyle=':', alpha=0.5)
    ax2.set_title('Drawdown', fontsize=13)
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(fontsize=8, loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart1 = os.path.join(OUTPUT_DIR, 'filter_grid_chart.png')
    plt.savefig(chart1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {chart1}")

    # --- グラフ2: パレートフロンティア散布図 ---
    fig, ax = plt.subplots(figsize=(12, 8))

    x = result_df['MaxDD'] * 100  # %表示
    y = result_df['年率'] * 100

    ax.scatter(x, y, alpha=0.4, s=30, c='gray', label='All scenarios')

    # パレート最適をハイライト
    px = [r['MaxDD'] * 100 for _, r in pareto_df.iterrows()]
    py = [r['年率'] * 100 for _, r in pareto_df.iterrows()]
    ax.plot(px, py, 'r-o', markersize=6, linewidth=2, label='Pareto frontier')

    # H+現行を強調
    h_row = result_df[result_df['シナリオ'] == 'H+_現行'].iloc[0]
    ax.scatter([h_row['MaxDD']*100], [h_row['年率']*100],
               s=200, c='red', marker='*', zorder=5, label='H+ (current)')

    # ★ゾーン
    ax.axvline(-20, color='green', linestyle=':', alpha=0.5)
    ax.axhline(45, color='green', linestyle=':', alpha=0.5)
    ax.fill_between([-20, 0], [45, 45], [100, 100],
                    alpha=0.05, color='green', label='Target zone')

    ax.set_xlabel('MaxDD (%)')
    ax.set_ylabel('Annual Return (%)')
    ax.set_title('Return vs MaxDD — Pareto Frontier', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    chart2 = os.path.join(OUTPUT_DIR, 'filter_grid_pareto.png')
    plt.savefig(chart2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {chart2}")

except Exception as e:
    print(f"  ⚠ グラフ生成失敗: {e}")
    import traceback
    traceback.print_exc()

# ================================================================
# 15. 推奨サマリ
# ================================================================
print("\n" + "=" * 70)
print("【推奨シナリオ】")
print("=" * 70)

# 年率45%以上 & MaxDD -20%以内
target = result_df[(result_df['年率'] >= 0.45) & (result_df['MaxDD'] >= -0.20)]
if len(target) > 0:
    best = target.sort_values('Calmar', ascending=False).iloc[0]
    print(f"\n  ★ 推奨: {best['シナリオ']}")
    print(f"     年率:   {best['年率']:.1%}")
    print(f"     Sharpe: {best['Sharpe']:.2f}")
    print(f"     MaxDD:  {best['MaxDD']:.1%}")
    print(f"     Calmar: {best['Calmar']:.2f}")
    print(f"     資産:   {best['最終資産(万)']:.0f}万円")
else:
    print("\n  ⚠ 年率45%以上 & MaxDD-20%以内のシナリオが見つかりません")
    best = result_df.sort_values('Calmar', ascending=False).iloc[0]
    print(f"  → Calmar最高: {best['シナリオ']}")
    print(f"     年率: {best['年率']:.1%} / MaxDD: {best['MaxDD']:.1%}")

# 年率最高（MaxDD-20%以内）
if len(target) > 0:
    best_ar = target.sort_values('年率', ascending=False).iloc[0]
    print(f"\n  ★ 年率最高(DD制約内): {best_ar['シナリオ']}")
    print(f"     年率:   {best_ar['年率']:.1%}")
    print(f"     MaxDD:  {best_ar['MaxDD']:.1%}")

print("\n" + "=" * 70)
print("完了！")
print("=" * 70)
print(f"  CSV:   {out_csv}")
print(f"  グラフ: {os.path.join(OUTPUT_DIR, 'filter_grid_*.png')}")