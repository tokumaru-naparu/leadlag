"""
analyze_yearly_ticker_performance.py

銘柄別パフォーマンスを「年ごと」に計算
各年の銘柄ランキングを出力

INPUT:
  - data/processed/trades_v2.csv
  - output/reports/ticker_profit_analysis.csv (銘柄情報)

OUTPUT:
  - output/reports/yearly_ticker_ranking.txt
  - output/reports/yearly_ticker_ranking.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402

# ================================================================
# 銘柄-業種マッピング（Phase 2で選定）
# ================================================================

TICKER_SECTOR = {
    2587: 'food',
    5020: 'energy',
    5233: 'construction',
    4004: 'materials',
    4536: 'pharma',
    7282: 'auto',
    5713: 'steel',
    6113: 'machinery',
    6971: 'electronics',
    9984: 'it_services',
    9501: 'utilities',
    9308: 'transport',
    8053: 'trading',
    2780: 'retail',
    8354: 'banks',
    8725: 'finance',
    8984: 'realestate',
}

SECTOR_NAMES = {
    'food': '食品',
    'energy': 'エネルギー',
    'construction': '建設',
    'materials': '素材化学',
    'pharma': '医薬品',
    'auto': '自動車',
    'steel': '鉄鋼非鉄',
    'machinery': '機械',
    'electronics': '電機精密',
    'it_services': '情報通信',
    'utilities': '電力ガス',
    'transport': '運輸物流',
    'trading': '商社卸売',
    'retail': '小売',
    'banks': '銀行',
    'finance': '金融',
    'realestate': '不動産',
}


def main() -> None:
    print("=" * 80)
    print("年ごとの銘柄パフォーマンス分析")
    print("=" * 80)

    # trades_v2とティッカー別利益データを読み込む
    print("\n[1] データを読み込む...")
    trades = pd.read_csv(PROCESSED_DIR / "trades_v2.csv")
    trades['date'] = pd.to_datetime(trades['date'])
    trades['year'] = trades['date'].dt.year

    ticker_profit_all = pd.read_csv(REPORTS_DIR / "ticker_profit_analysis.csv")

    print(f"  期間: {trades['year'].min()} ~ {trades['year'].max()}")
    print(f"  銘柄数: {len(ticker_profit_all)}")

    # 年ごとに銘柄パフォーマンスを計算
    print("\n[2] 年ごとの銘柄パフォーマンスを計算...")
    yearly_results = defaultdict(list)

    for year in sorted(trades['year'].unique()):
        year_data = trades[trades['year'] == year]

        # この年の各銘柄のパフォーマンス
        for ticker, sector in TICKER_SECTOR.items():
            # ロング日
            long_mask = year_data['long_1'].eq(sector) | year_data['long_2'].eq(sector) | year_data['long_3'].eq(sector)
            long_days = long_mask.sum()

            # ショート日
            short_mask = year_data['short_1'].eq(sector) | year_data['short_2'].eq(sector) | year_data['short_3'].eq(sector)
            short_days = short_mask.sum()

            total_days = long_days + short_days

            # この年のロング/ショート日のリターンを集計
            long_return = year_data[long_mask]['long_return'].sum() if long_days > 0 else 0
            short_return = -year_data[short_mask]['short_return'].sum() if short_days > 0 else 0

            total_return = long_return + short_return

            # 平均日次%
            if total_days > 0:
                avg_daily_pct = total_return / total_days * 100
            else:
                avg_daily_pct = 0

            if total_days > 0:  # 売買がある年のみ記録
                yearly_results[year].append({
                    'year': year,
                    'ticker': ticker,
                    'sector': sector,
                    'sector_name': SECTOR_NAMES[sector],
                    'long_days': long_days,
                    'short_days': short_days,
                    'total_days': total_days,
                    'total_return': total_return,
                    'avg_daily_pct': avg_daily_pct,
                })

    # 結果をファイルに保存
    print("\n[3] 結果を保存...")

    # TXT出力
    txt_path = REPORTS_DIR / "yearly_ticker_ranking.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("年ごとの銘柄別パフォーマンスランキング (2010-2026)\n")
        f.write("=" * 100 + "\n\n")

        for year in sorted(yearly_results.keys()):
            results = yearly_results[year]
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('avg_daily_pct', ascending=False)

            f.write(f"\n{'='*100}\n")
            f.write(f"{year}年\n")
            f.write(f"{'='*100}\n")
            f.write(f"{'順位':>3} {'銘柄':25} {'ロング日':>8} {'ショート日':>8} {'平均日次%':>12} {'累積%':>11}\n")
            f.write("-" * 100 + "\n")

            for rank, (_, row) in enumerate(results_df.iterrows(), 1):
                ticker = int(row['ticker'])
                sector_name = row['sector_name']
                long_days = int(row['long_days'])
                short_days = int(row['short_days'])
                avg_daily = row['avg_daily_pct']
                cum_return = row['total_return'] * 100

                f.write(f"{rank:3d} {ticker} ({sector_name:20}) {long_days:>8} {short_days:>8} ")
                f.write(f"{avg_daily:>+11.4f}% {cum_return:>+10.2f}%\n")

    print(f"  TXT: {txt_path}")

    # CSV出力（全年・全銘柄）
    csv_data = []
    for year in sorted(yearly_results.keys()):
        csv_data.extend(yearly_results[year])

    csv_df = pd.DataFrame(csv_data)
    csv_path = REPORTS_DIR / "yearly_ticker_ranking.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    print("\n[完了] 年ごとの銘柄パフォーマンス分析が完了しました")


if __name__ == "__main__":
    main()
