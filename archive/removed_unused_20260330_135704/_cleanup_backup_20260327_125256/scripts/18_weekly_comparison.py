# 17_weekly_comparison.py
# 2025年12月〜2026年3月の週次推移を現状 vs H+ で比較するスクリプト

import pandas as pd
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / 'data'


def pick_input_csv(base_dir: Path) -> Path:
    candidates = [
        base_dir / 'data' / 'analysis_data_for_opus.csv',
        base_dir / 'scripts' / 'data' / 'analysis_data_for_opus.csv',
    ]

    best_path = None
    best_rows = -1
    for p in candidates:
        if not p.exists():
            continue
        try:
            rows = len(pd.read_csv(p))
        except Exception:
            rows = -1
        if rows > best_rows:
            best_rows = rows
            best_path = p

    if best_path is None:
        raise FileNotFoundError(
            'analysis_data_for_opus.csv が見つかりません。候補: data/, scripts/data/'
        )

    print(f"  📁 入力CSV: {best_path} ({best_rows}行)")
    return best_path


CSV_PATH = pick_input_csv(BASE_DIR)

# ================================================================
# データ読み込み・期間絞り込み
# ================================================================
print("=" * 60)
print("データ読み込み中...")
print("=" * 60)

df = pd.read_csv(CSV_PATH, parse_dates=['date'])
df.set_index('date', inplace=True)

# 2025年12月〜2026年3月に絞る
start = '2025-12-01'
end   = '2026-03-31'
df = df[start:end].copy()

print(f"  期間: {df.index.min().date()} 〜 {df.index.max().date()}")
print(f"  日数: {len(df)} 営業日")

# ================================================================
# シグナル分位の計算（Q1〜Q5）
# ※ 全期間の分位ではなく、この期間内での相対的な強度
# ================================================================
df['sig_quintile'] = pd.qcut(
    df['signal_strength'],
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# ================================================================
# H+戦略のリターン計算
# ================================================================
def calc_hplus_return(row):
    """
    H+ Strategy:
      Step1: サイズ = signal_strength × 4 (下限0.3, 上限3.0)
      Step2: Q4/Q5の日はショート1.3倍・ロング0.7倍
    """
    size = min(max(row['signal_strength'] * 4, 0.3), 3.0)

    if row['sig_quintile'] >= 4:
        weighted_return = 0.7 * row['long_return'] - 1.3 * row['short_return']
    else:
        weighted_return = row['long_return'] - row['short_return']

    return weighted_return * size

df['hplus_return'] = df.apply(calc_hplus_return, axis=1)

# ================================================================
# 週次集計
# ================================================================
# 週の開始日（月曜日）でグループ化
df['week_start'] = df.index - pd.to_timedelta(df.index.dayofweek, unit='D')

weekly = df.groupby('week_start').agg(
    # 現状戦略
    current_cum_return = ('strategy_return', lambda x: (1 + x).prod() - 1),
    current_win_days   = ('is_correct', 'sum'),
    current_total_days = ('is_correct', 'count'),

    # H+戦略
    hplus_cum_return   = ('hplus_return', lambda x: (1 + x).prod() - 1),

    # 参考情報
    avg_signal         = ('signal_strength', 'mean'),
    avg_us_vol         = ('us_sector_vol', 'mean'),
).reset_index()

weekly.columns = [
    '週開始日', '現状_週間R', '現状_勝日数', '現状_取引日数',
    'H+_週間R', '平均シグナル強度', '米国業種間ボラ'
]

weekly['週番号'] = range(1, len(weekly) + 1)
weekly['週末日'] = weekly['週開始日'] + pd.Timedelta(days=4)

# ================================================================
# 累積資産推移（100万円スタート）
# ================================================================
capital = 1_000_000
hplus_capital = 1_000_000

capital_list = []
hplus_capital_list = []

for _, row in weekly.iterrows():
    capital       *= (1 + row['現状_週間R'])
    hplus_capital *= (1 + row['H+_週間R'])
    capital_list.append(capital)
    hplus_capital_list.append(hplus_capital)

weekly['現状_累積資産(万円)']  = [c / 10000 for c in capital_list]
weekly['H+_累積資産(万円)']    = [c / 10000 for c in hplus_capital_list]
weekly['現状_勝率']             = (weekly['現状_勝日数'] / weekly['現状_取引日数'] * 100).round(1).astype(str) + '%'

# ================================================================
# 表示
# ================================================================
print("\n" + "=" * 60)
print("週次推移比較（現状 vs H+）")
print("=" * 60)
print(f"{'週':<4} {'週開始':<12} {'週末':<12} "
      f"{'現状週間R':>10} {'H+週間R':>10} "
      f"{'現状資産':>10} {'H+資産':>10} "
      f"{'現状勝率':>8} {'シグナル':>8}")
print("-" * 90)

for _, row in weekly.iterrows():
    current_r = row['現状_週間R'] * 100
    hplus_r   = row['H+_週間R'] * 100

    # プラスなら+、マイナスならそのまま
    c_mark  = "+" if current_r >= 0 else ""
    h_mark  = "+" if hplus_r  >= 0 else ""

    print(
        f"Week{int(row['週番号']):<2} "
        f"{str(row['週開始日'].date()):<12} "
        f"{str(row['週末日'].date()):<12} "
        f"{c_mark}{current_r:>8.2f}% "
        f"{h_mark}{hplus_r:>8.2f}% "
        f"{row['現状_累積資産(万円)']:>9.1f}万 "
        f"{row['H+_累積資産(万円)']:>9.1f}万 "
        f"{row['現状_勝率']:>8} "
        f"{row['平均シグナル強度']:>8.3f}"
    )

# ================================================================
# サマリー
# ================================================================
print("\n" + "=" * 60)
print("4ヶ月間サマリー")
print("=" * 60)

total_days      = weekly['現状_取引日数'].sum()
total_win_days  = weekly['現状_勝日数'].sum()
current_total_r = (weekly['現状_累積資産(万円)'].iloc[-1] / 100 - 1) * 100
hplus_total_r   = (weekly['H+_累積資産(万円)'].iloc[-1] / 100 - 1) * 100

# 週ごとの勝敗
current_win_weeks = (weekly['現状_週間R'] > 0).sum()
hplus_win_weeks   = (weekly['H+_週間R'] > 0).sum()
total_weeks       = len(weekly)

print(f"\n  【現状戦略】")
print(f"    期間リターン : {current_total_r:+.2f}%")
print(f"    最終資産     : {weekly['現状_累積資産(万円)'].iloc[-1]:.1f}万円")
print(f"    日次勝率     : {total_win_days}/{total_days}日 ({total_win_days/total_days*100:.1f}%)")
print(f"    週次勝率     : {current_win_weeks}/{total_weeks}週 ({current_win_weeks/total_weeks*100:.1f}%)")

print(f"\n  【H+戦略】")
print(f"    期間リターン : {hplus_total_r:+.2f}%")
print(f"    最終資産     : {weekly['H+_累積資産(万円)'].iloc[-1]:.1f}万円")
print(f"    週次勝率     : {hplus_win_weeks}/{total_weeks}週 ({hplus_win_weeks/total_weeks*100:.1f}%)")

print(f"\n  【差分（H+ - 現状）】")
print(f"    リターン差   : {hplus_total_r - current_total_r:+.2f}pp")
print(f"    資産差       : {weekly['H+_累積資産(万円)'].iloc[-1] - weekly['現状_累積資産(万円)'].iloc[-1]:+.1f}万円")

# ================================================================
# CSV出力
# ================================================================
out_path = OUTPUT_DIR / 'weekly_comparison_dec2025_mar2026.csv'
weekly.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n✅ CSV出力: {out_path}")