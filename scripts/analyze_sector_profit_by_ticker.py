"""
analyze_sector_profit_by_ticker.py

各銘柄が生み出した利益を正確に計算
方法: 各銘柄の実OCリターンを使用（簡略化なし）

INPUT:
  - yfinance: JP個別株の OHLC (2010-2026)
  - data/processed/trades_v2.csv (ロング/ショート業種)

OUTPUT:
  - output/reports/ticker_profit_analysis.csv
  - output/reports/ticker_profit_analysis.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

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
    print("=" * 70)
    print("銘柄別利益分析（個別OCリターン使用）")
    print("=" * 70)

    # Step 1: trades_v2.csvを読み込む
    print("\n[1] trades_v2.csvを読み込む...")
    trades_path = PROCESSED_DIR / "trades_v2.csv"
    trades = pd.read_csv(trades_path)
    trades['date'] = pd.to_datetime(trades['date'])
    print(f"  {len(trades)}日分のトレード記録")

    # Step 2: 各銘柄のOCリターンを取得
    print("\n[2] 各銘柄のOCリターンを取得（yfinance）...")
    oc_returns = {}
    for ticker in TICKER_SECTOR.keys():
        sector = TICKER_SECTOR[ticker]
        sector_name = SECTOR_NAMES[sector]
        print(f"  {sector_name:15s} ({ticker}) ...", end="", flush=True)

        try:
            df = yf.download(
                f"{ticker}.T",
                start="2010-04-06",
                end="2026-03-31",
                progress=False,
                auto_adjust=True,
            )

            if len(df) == 0:
                print(" [NO DATA]")
                continue

            # Open-to-Close リターン
            oc = ((df['Close'] - df['Open']) / df['Open']).reset_index()
            oc.columns = ['date', 'oc_return']
            oc['date'] = pd.to_datetime(oc['date'])
            oc_returns[ticker] = oc
            print(f" OK ({len(oc)}日)")

        except Exception as e:
            print(f" [ERROR: {str(e)[:30]}]")

    print(f"  成功: {len(oc_returns)}/{len(TICKER_SECTOR)}")

    # Step 3: 利益計算
    print("\n[3] 銘柄別利益を計算...")

    # 各銘柄のデータ利用可能期間を把握
    ticker_date_range = {}
    for ticker in oc_returns.keys():
        oc_df = oc_returns[ticker]
        dates = oc_df['date']
        ticker_date_range[ticker] = (dates.min(), dates.max())
        sector = TICKER_SECTOR[ticker]
        sector_name = SECTOR_NAMES[sector]
        print(f"  {sector_name:15s} ({ticker}): {dates.min().date()} ~ {dates.max().date()}")

    ticker_profit = defaultdict(float)
    ticker_long_days = defaultdict(int)
    ticker_short_days = defaultdict(int)

    for idx, row in trades.iterrows():
        date = row['date']

        # ロング3業種
        long_sectors = [row['long_1'], row['long_2'], row['long_3']]
        for sector in long_sectors:
            if pd.isna(sector):
                continue

            # 業種→銘柄を逆引き
            tickers_in_sector = [t for t, s in TICKER_SECTOR.items() if s == sector]
            if not tickers_in_sector:
                continue

            ticker = tickers_in_sector[0]  # Phase 2では各業種1銘柄

            if ticker not in oc_returns:
                continue

            # その日のデータを検索
            oc_df = oc_returns[ticker]
            oc_on_date = oc_df[oc_df['date'] == date]['oc_return']
            if len(oc_on_date) > 0:
                oc = oc_on_date.iloc[0]
                if not np.isnan(oc):
                    ticker_profit[ticker] += oc
                    ticker_long_days[ticker] += 1

        # ショート3業種
        short_sectors = [row['short_1'], row['short_2'], row['short_3']]
        for sector in short_sectors:
            if pd.isna(sector):
                continue

            tickers_in_sector = [t for t, s in TICKER_SECTOR.items() if s == sector]
            if not tickers_in_sector:
                continue

            ticker = tickers_in_sector[0]

            if ticker not in oc_returns:
                continue

            # その日のデータを検索
            oc_df = oc_returns[ticker]
            oc_on_date = oc_df[oc_df['date'] == date]['oc_return']
            if len(oc_on_date) > 0:
                oc = oc_on_date.iloc[0]
                if not np.isnan(oc):
                    ticker_profit[ticker] -= oc  # 反対売買
                    ticker_short_days[ticker] += 1

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(trades)} 日処理完了...")

    # Step 4: 結果を集計
    print("\n[4] 結果を集計...")
    results = []
    for ticker, sector in TICKER_SECTOR.items():
        if ticker not in oc_returns:
            continue

        profit = ticker_profit.get(ticker, 0.0)
        long_days = ticker_long_days.get(ticker, 0)
        short_days = ticker_short_days.get(ticker, 0)
        total_days = long_days + short_days

        # 平均日次リターン（%）
        if total_days > 0:
            avg_daily_return = profit / total_days * 100  # %単位
        else:
            avg_daily_return = 0.0

        # 初期資本100万で換算
        profit_yen = 1_000_000 * profit

        results.append({
            'ticker': ticker,
            'sector': sector,
            'sector_name': SECTOR_NAMES[sector],
            'long_days': long_days,
            'short_days': short_days,
            'total_days': total_days,
            'oc_return_sum': profit,
            'avg_daily_return_pct': avg_daily_return,
            'profit_yen': profit_yen,
        })

    df_results = pd.DataFrame(results)
    # 平均日次リターンでソート
    df_results = df_results.sort_values('avg_daily_return_pct', ascending=False)

    # Step 5: 検証
    print("\n[5] 検証...")
    total_profit_sum = df_results['oc_return_sum'].sum()
    strategy_cumulative = (trades['strategy_return'] + 1).prod() - 1

    print(f"  銘柄別利益合計: {total_profit_sum:.4f}")
    print(f"  strategy_return累積: {strategy_cumulative:.4f}")
    print(f"  乖離: {abs(total_profit_sum - strategy_cumulative):.4f}")

    # Step 6: ファイルに保存
    print("\n[6] 結果を保存...")
    csv_path = REPORTS_DIR / "ticker_profit_analysis.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    # テキストレポート
    txt_path = REPORTS_DIR / "ticker_profit_analysis.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("銘柄別利益分析 (2010-04-07 ~ 2026-03-30)\n")
        f.write("=" * 90 + "\n\n")

        f.write(f"{'順位':>3} {'銘柄':25} {'ロング日':>8} {'ショート日':>8} {'合計日':>7} ")
        f.write(f"{'平均日次%':>11} {'累積%':>11}\n")
        f.write("-" * 105 + "\n")

        for rank, (_, row) in enumerate(df_results.iterrows(), 1):
            ticker = int(row['ticker'])
            sector_name = row['sector_name']
            long_days = int(row['long_days'])
            short_days = int(row['short_days'])
            total_days = int(row['total_days'])
            avg_daily = row['avg_daily_return_pct']
            cum_return = row['oc_return_sum'] * 100

            f.write(f"{rank:3d} {ticker} ({sector_name:20}) {long_days:>8} {short_days:>8} {total_days:>7} ")
            f.write(f"{avg_daily:>+10.4f}% {cum_return:>+10.2f}%\n")

        f.write("-" * 90 + "\n")
        f.write(f"{'合計':23} {df_results['long_days'].sum():>7.0f}")
        f.write(f" {df_results['short_days'].sum():>7.0f}")
        f.write(f" {total_profit_sum*100:>+11.2f}% {df_results['profit_yen'].sum():>+11.0f}\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write("検証\n")
        f.write("=" * 90 + "\n")
        f.write(f"銘柄別利益合計:      {total_profit_sum:+.6f}\n")
        f.write(f"strategy_return累積: {strategy_cumulative:+.6f}\n")
        f.write(f"乖離:               {abs(total_profit_sum - strategy_cumulative):+.6f}\n")

    print(f"  TXT: {txt_path}")

    print("\n[完了] 銘柄別利益分析が完了しました")


if __name__ == "__main__":
    main()
