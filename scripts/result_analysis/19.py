# 19_weekly_comparison_6months.py
# 2025年10月〜2026年3月の週次推移を現状 vs H+ で比較するスクリプト

import pandas as pd
import numpy as np
import os
from pathlib import Path

def resolve_base_dir(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / 'leadlag_CLAUDE.md').exists() and (p / 'scripts').exists():
            return p
    raise FileNotFoundError('プロジェクトルートを特定できませんでした。')


BASE_DIR = resolve_base_dir(Path(__file__).resolve().parent)
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

# 2025年10月〜2026年3月に絞る（4ヶ月→6ヶ月に拡張）
start = '2025-10-01'
end   = '2026-03-31'
df = df[start:end].copy()

print(f"  期間: {df.index.min().date()} 〜 {df.index.max().date()}")
print(f"  日数: {len(df)} 営業日")

# ================================================================
# シグナル分位の計算（Q1〜Q5）
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
df['week_start'] = df.index - pd.to_timedelta(df.index.dayofweek, unit='D')

weekly = df.groupby('week_start').agg(
    current_cum_return = ('strategy_return', lambda x: (1 + x).prod() - 1),
    current_win_days   = ('is_correct', 'sum'),
    current_total_days = ('is_correct', 'count'),
    hplus_cum_return   = ('hplus_return', lambda x: (1 + x).prod() - 1),
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
# 月の区切り線を入れて表示
# ================================================================
print("\n" + "=" * 60)
print("週次推移比較（現状 vs H+）　2025年10月〜2026年3月")
print("=" * 60)
print(f"{'週':<6} {'週開始':<12} {'週末':<12} "
      f"{'現状週間R':>10} {'H+週間R':>10} "
      f"{'現状資産':>10} {'H+資産':>10} "
      f"{'現状勝率':>8} {'シグナル':>8}")
print("-" * 90)

current_month = None
for _, row in weekly.iterrows():
    month = row['週開始日'].month

    # 月が変わったら区切り線
    if month != current_month:
        if current_month is not None:
            print("-" * 90)
        month_names = {10:'─── 10月 ───', 11:'─── 11月 ───',
                       12:'─── 12月 ───',  1:'─── 1月 ───',
                        2:'─── 2月 ───',   3:'─── 3月 ───'}
        print(f"  {month_names.get(month, '')}")
        current_month = month

    current_r = row['現状_週間R'] * 100
    hplus_r   = row['H+_週間R'] * 100
    c_mark    = "+" if current_r >= 0 else ""
    h_mark    = "+" if hplus_r  >= 0 else ""

    print(
        f"Week{int(row['週番号']):<3} "
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
print("6ヶ月間サマリー")
print("=" * 60)

total_days     = weekly['現状_取引日数'].sum()
total_win_days = weekly['現状_勝日数'].sum()
current_total_r = (weekly['現状_累積資産(万円)'].iloc[-1] / 100 - 1) * 100
hplus_total_r   = (weekly['H+_累積資産(万円)'].iloc[-1] / 100 - 1) * 100

current_win_weeks = (weekly['現状_週間R'] > 0).sum()
hplus_win_weeks   = (weekly['H+_週間R'] > 0).sum()
total_weeks       = len(weekly)

# 月別サマリー
print("\n  【月別リターン】")
print(f"  {'月':<8} {'現状':>8} {'H+':>8}")
print(f"  {'-'*26}")
for month_num, month_name in [(10,'10月'), (11,'11月'), (12,'12月'),
                               (1,'1月'),  (2,'2月'),  (3,'3月')]:
    mask = weekly['週開始日'].dt.month == month_num
    if mask.sum() == 0:
        continue
    c_r = weekly.loc[mask, '現状_週間R'].sum() * 100
    h_r = weekly.loc[mask, 'H+_週間R'].sum() * 100
    c_mark = "+" if c_r >= 0 else ""
    h_mark = "+" if h_r >= 0 else ""
    print(f"  {month_name:<8} {c_mark}{c_r:>6.2f}%  {h_mark}{h_r:>6.2f}%")

print(f"\n  【現状戦略】")
print(f"    期間リターン : {current_total_r:+.2f}%")
print(f"    最終資産     : {weekly['現状_累積資産(万円)'].iloc[-1]:.1f}万円")
print(f"    日次勝率     : {total_win_days}/{total_days}日 ({total_win_days/total_days*100:.1f}%)")
print(f"    週次勝率     : {current_win_weeks}/{total_weeks}週 ({current_win_weeks/total_weeks*100:.1f}%)")
print(f"    最大週間損失 : {weekly['現状_週間R'].min()*100:.2f}%")

print(f"\n  【H+戦略】")
print(f"    期間リターン : {hplus_total_r:+.2f}%")
print(f"    最終資産     : {weekly['H+_累積資産(万円)'].iloc[-1]:.1f}万円")
print(f"    週次勝率     : {hplus_win_weeks}/{total_weeks}週 ({hplus_win_weeks/total_weeks*100:.1f}%)")
print(f"    最大週間損失 : {weekly['H+_週間R'].min()*100:.2f}%")

print(f"\n  【差分（H+ - 現状）】")
print(f"    リターン差   : {hplus_total_r - current_total_r:+.2f}pp")
print(f"    資産差       : {weekly['H+_累積資産(万円)'].iloc[-1] - weekly['現状_累積資産(万円)'].iloc[-1]:+.1f}万円")

# ================================================================
# CSV出力
# ================================================================
out_path = OUTPUT_DIR / 'weekly_comparison_oct2025_mar2026.csv'
weekly.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n✅ CSV出力: {out_path}")