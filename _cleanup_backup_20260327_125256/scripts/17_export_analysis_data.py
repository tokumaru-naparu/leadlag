# 17_export_analysis_data.py
# Opusが要求したデータを統合CSVとして出力するスクリプト（修正版）

import pandas as pd
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR  = BASE_DIR / 'data'


def pick_history_dir(base_dir: Path) -> Path:
    """履歴データ候補の中から、行数が多い方を採用する。"""
    candidates = [
        base_dir / 'data' / 'history',
        base_dir / 'scripts' / 'data' / 'history',
    ]

    best_dir = None
    best_rows = -1

    for d in candidates:
        p = d / 'signals.csv'
        if not p.exists():
            continue
        try:
            rows = len(pd.read_csv(p))
        except Exception:
            rows = -1
        if rows > best_rows:
            best_rows = rows
            best_dir = d

    if best_dir is None:
        raise FileNotFoundError(
            "signals.csv が見つかりません。候補: data/history, scripts/data/history"
        )

    print(f"  📁 採用履歴ディレクトリ: {best_dir} (signals.csv {best_rows}行)")
    return best_dir


HISTORY_DIR = pick_history_dir(BASE_DIR)

# ================================================================
# 列名定義（STEP1の出力から確定）
# ================================================================

# signals.csv のシグナル値列
SIGNAL_COLS = [
    'signal_food', 'signal_energy', 'signal_construction', 'signal_materials',
    'signal_pharma', 'signal_auto', 'signal_steel', 'signal_machinery',
    'signal_electronics', 'signal_it_services', 'signal_utilities',
    'signal_transport', 'signal_trading', 'signal_retail',
    'signal_banks', 'signal_finance', 'signal_realestate'
]

# returns.csv のリターン列
RETURN_COLS = [
    'oc_return_food', 'oc_return_energy', 'oc_return_construction',
    'oc_return_materials', 'oc_return_pharma', 'oc_return_auto',
    'oc_return_steel', 'oc_return_machinery', 'oc_return_electronics',
    'oc_return_it_services', 'oc_return_utilities', 'oc_return_transport',
    'oc_return_trading', 'oc_return_retail', 'oc_return_banks',
    'oc_return_finance', 'oc_return_realestate'
]

# market.csv の米国11業種リターン列
US_COLS = [
    'us_cc_materials', 'us_cc_communication', 'us_cc_energy',
    'us_cc_financials', 'us_cc_industrials', 'us_cc_tech',
    'us_cc_staples', 'us_cc_realestate', 'us_cc_utilities',
    'us_cc_healthcare', 'us_cc_discretionary'
]

# ================================================================
# データ読み込み
# ================================================================
print("=" * 60)
print("データ読み込み中...")
print("=" * 60)

def load(fname):
    path = HISTORY_DIR / fname
    df = pd.read_csv(path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    print(f"  ✅ {fname}: {len(df)}行")
    return df

sig  = load('signals.csv')
ret  = load('returns.csv')
trd  = load('trades.csv')
mkt  = load('market.csv')
perf = load('performance.csv')

# ================================================================
# 統合CSVの組み立て
# ================================================================
print("\n統合中...")

merged = pd.DataFrame(index=sig.index)

# --- シグナル関連 ---
merged['signal_strength'] = sig['signal_strength']   # ロングショート差
merged['signal_spread']   = sig['signal_spread']     # 別の強度指標（あれば活用）

# 17業種シグナル値（手段3：業種数動的変更に使用）
for col in SIGNAL_COLS:
    if col in sig.columns:
        merged[col] = sig[col]

# --- 売買記録 ---
# どの業種をロング・ショートしたか
for col in ['long_1', 'long_2', 'long_3', 'short_1', 'short_2', 'short_3']:
    if col in trd.columns:
        merged[col] = trd[col]

# 曜日・月（パターン分析用）
if 'weekday' in trd.columns:
    merged['weekday'] = trd['weekday']
if 'month' in trd.columns:
    merged['month'] = trd['month']

# --- リターン ---
# 戦略全体のリターン（ロングショート合計）
merged['strategy_return']   = trd['strategy_return']
merged['long_return']       = trd['long_return']
merged['short_return']      = trd['short_return']
merged['long_only_return']  = trd['long_only_return']
merged['is_correct']        = trd['is_correct']   # 勝ち=1 / 負け=0

# 17業種個別リターン（手段2：非対称ウェイトに使用）
for col in RETURN_COLS:
    if col in ret.columns:
        merged[col] = ret[col]

# --- 市場環境 ---
# 米国11業種リターン（VIXはデータになし → 米国全体リターンで代替）
available_us_cols = [c for c in US_COLS if c in mkt.columns]
for col in US_COLS:
    if col in mkt.columns:
        merged[col] = mkt[col]

# 米国市場の平均リターン（全11業種の平均）
merged['us_return_avg'] = mkt[available_us_cols].mean(axis=1) if available_us_cols else np.nan

# 米国業種間ボラティリティ（業種間のばらつき → 高いほど戦略が効く）
merged['us_sector_vol'] = mkt[available_us_cols].std(axis=1) if available_us_cols else np.nan

# --- 資産推移（手段1：サイジング効果確認用）---
if 'capital' in perf.columns:
    merged['capital']          = perf['capital']
if 'win_rate_20d' in perf.columns:
    merged['win_rate_20d']     = perf['win_rate_20d']   # 直近20日勝率

# ================================================================
# 出力
# ================================================================
out_path = OUTPUT_DIR / 'analysis_data_for_opus.csv'
merged.reset_index().to_csv(out_path, index=False, encoding='utf-8-sig')

print("\n" + "=" * 60)
print("✅ 出力完了")
print("=" * 60)
print(f"  ファイル: {out_path}")
print(f"  行数:     {len(merged)} 日")
print(f"  列数:     {len(merged.columns)} 列")
print(f"  期間:     {merged.index.min().date()} 〜 {merged.index.max().date()}")

print(f"\n列一覧（NaN件数付き）:")
for c in merged.columns:
    null_count = merged[c].isna().sum()
    mark = "  ⚠ NaN多い" if null_count > 100 else ""
    print(f"  {c:40s} NaN:{null_count:4d}件{mark}")

print("\n先頭3行:")
print(merged.head(3).to_string())