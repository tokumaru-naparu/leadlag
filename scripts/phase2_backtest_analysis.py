"""
phase2_backtest_analysis.py
=================================
Phase 2: 個別株版バックテスト分析
- 全17銘柄のバックテスト実行
- 仮採用8業種の貢献度分析
- 採用/除外の判定

INPUT:
  - data/processed/trades_v2.csv
  - yfinance（個別株OHLCデータ）

OUTPUT:
  - output/results/phase2_backtest_analysis.csv
  - output/reports/phase2_backtest_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR, REPORTS_DIR, OUTPUT_DIR

# ================================================================
# 銘柄マッピング
# ================================================================

CONFIRMED_TICKERS = {
    'energy': 5020,
    'trading': 8053,
    'finance': 8725,
    'construction': 5233,
    'it_services': 9984,
    'electronics': 6971,
    'steel': 5713,
    'utilities': 9501,
    'banks': 8354,
}

TENTATIVE_TICKERS = {
    'food': 2587,
    'materials': 4004,
    'pharma': 4536,
    'auto': 7282,
    'machinery': 6113,
    'transport': 9308,
    'retail': 2780,
    'realestate': 8984,
}

ALL_TICKERS = {**CONFIRMED_TICKERS, **TENTATIVE_TICKERS}

SECTOR_NAMES = {
    'energy': 'エネルギー',
    'trading': '商社',
    'finance': '金融',
    'construction': '建設',
    'it_services': 'IT通信',
    'electronics': '電機',
    'steel': '鉄鋼',
    'utilities': '電力ガス',
    'banks': '銀行',
    'food': '食品',
    'materials': '素材化学',
    'pharma': '医薬品',
    'auto': '自動車',
    'machinery': '機械',
    'transport': '運輸',
    'retail': '小売',
    'realestate': '不動産',
}

INITIAL_CAPITAL = 1_000_000


def fetch_and_cache_stocks() -> dict:
    """すべての銘柄データを取得・キャッシュ"""
    print("[データ取得] yfinanceから全銘柄データを取得...")

    stock_data = {}
    for sector, ticker in ALL_TICKERS.items():
        sector_name = SECTOR_NAMES[sector]
        print(f"  {sector_name:15s} ({ticker}) ...", end='', flush=True)

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

            # Open-Close リターンを計算
            oc_returns = (df['Close'] - df['Open']) / df['Open']
            oc_df = oc_returns.reset_index()
            oc_df.columns = ['date', 'oc_return']
            oc_df['date'] = pd.to_datetime(oc_df['date'])

            stock_data[ticker] = oc_df
            print(f" OK ({len(oc_df)}日)")

        except Exception as e:
            print(f" [ERROR]")

    print(f"\n  成功: {len(stock_data)}/{len(ALL_TICKERS)}")
    return stock_data


def run_backtest(trades_df: pd.DataFrame, stock_data: dict) -> tuple:
    """バックテスト実行"""
    print("\n[バックテスト実行] 個別株版...")

    # 銘柄別の損益追跡
    ticker_profit = defaultdict(float)
    ticker_long_days = defaultdict(int)
    ticker_short_days = defaultdict(int)

    # 日別パフォーマンス
    daily_perf = []
    equity = INITIAL_CAPITAL

    for idx, row in trades_df.iterrows():
        date = row['date']

        # ロング銘柄の平均リターン
        long_returns = []
        for col in ['long_1', 'long_2', 'long_3']:
            if pd.isna(row[col]):
                continue

            sector = row[col]
            ticker = ALL_TICKERS.get(sector)
            if not ticker or ticker not in stock_data:
                continue

            oc_df = stock_data[ticker]
            oc_row = oc_df[oc_df['date'] == date]
            if len(oc_row) > 0:
                oc_ret = oc_row.iloc[0]['oc_return']
                if not np.isnan(oc_ret):
                    long_returns.append(oc_ret)
                    ticker_profit[ticker] += oc_ret
                    ticker_long_days[ticker] += 1

        # ショート銘柄の平均リターン（反対）
        short_returns = []
        for col in ['short_1', 'short_2', 'short_3']:
            if pd.isna(row[col]):
                continue

            sector = row[col]
            ticker = ALL_TICKERS.get(sector)
            if not ticker or ticker not in stock_data:
                continue

            oc_df = stock_data[ticker]
            oc_row = oc_df[oc_df['date'] == date]
            if len(oc_row) > 0:
                oc_ret = oc_row.iloc[0]['oc_return']
                if not np.isnan(oc_ret):
                    short_returns.append(oc_ret)
                    ticker_profit[ticker] -= oc_ret
                    ticker_short_days[ticker] += 1

        # 日次リターン
        if long_returns and short_returns:
            long_avg = np.mean(long_returns)
            short_avg = np.mean(short_returns)
            daily_return = long_avg - short_avg
        else:
            daily_return = 0.0

        # 資本更新
        equity *= (1 + daily_return)

        daily_perf.append({
            'date': date,
            'daily_return': daily_return,
            'equity': equity,
        })

    perf_df = pd.DataFrame(daily_perf)

    # 統計計算
    returns = perf_df['daily_return'].fillna(0)
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    annual_return = (equity / INITIAL_CAPITAL) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0

    if returns.std() > 0:
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    else:
        sharpe = 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    stats = {
        'final_equity': equity,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
    }

    print(f"  最終資本: {equity:,.0f}円")
    print(f"  総リターン: {total_return*100:.2f}%")
    print(f"  Sharpe: {sharpe:.3f}, MaxDD: {max_dd*100:.2f}%")

    return perf_df, stats, ticker_profit, ticker_long_days, ticker_short_days


def analyze_tentative_sectors(
    trades_df: pd.DataFrame,
    stock_data: dict,
) -> pd.DataFrame:
    """仮採用業種の分析"""
    print("\n[分析] 仮採用8業種の貢献度...")

    results = []

    for sector, ticker in TENTATIVE_TICKERS.items():
        if ticker not in stock_data:
            continue

        oc_df = stock_data[ticker]
        long_rets = []
        short_rets = []

        for idx, row in trades_df.iterrows():
            date = row['date']
            oc_row = oc_df[oc_df['date'] == date]

            if len(oc_row) == 0:
                continue

            oc_ret = oc_row.iloc[0]['oc_return']
            if np.isnan(oc_ret):
                continue

            # ロング
            if row['long_1'] == sector or row['long_2'] == sector or row['long_3'] == sector:
                long_rets.append(oc_ret)

            # ショート
            if row['short_1'] == sector or row['short_2'] == sector or row['short_3'] == sector:
                short_rets.append(oc_ret)

        # 統計
        long_avg = np.mean(long_rets) if long_rets else np.nan
        long_win = (np.array(long_rets) > 0).sum() / len(long_rets) if long_rets else np.nan

        short_avg = np.mean(short_rets) if short_rets else np.nan
        short_win = (np.array(short_rets) < 0).sum() / len(short_rets) if short_rets else np.nan

        # 判定
        total_pnl = len(long_rets) * long_avg - len(short_rets) * short_avg if long_rets and short_rets else np.nan
        judgement = 'ADOPT' if not np.isnan(total_pnl) and total_pnl > 0 else 'MARGINAL' if not np.isnan(total_pnl) and total_pnl > -0.001 else 'REJECT'

        results.append({
            'sector': sector,
            'sector_name': SECTOR_NAMES[sector],
            'ticker': ticker,
            'long_days': len(long_rets),
            'long_avg_pct': long_avg * 100 if not np.isnan(long_avg) else np.nan,
            'long_win_rate': long_win * 100 if not np.isnan(long_win) else np.nan,
            'short_days': len(short_rets),
            'short_avg_pct': short_avg * 100 if not np.isnan(short_avg) else np.nan,
            'short_win_rate': short_win * 100 if not np.isnan(short_win) else np.nan,
            'total_pnl': total_pnl,
            'judgement': judgement,
        })

    df = pd.DataFrame(results)
    print("\n仮採用業種の評価:")
    for _, r in df.iterrows():
        print(f"  {r['sector_name']:10s}: {r['judgement']:8s} (Long: {r['long_avg_pct']:+.3f}%, Short: {r['short_avg_pct']:+.3f}%)")

    return df


def main():
    print("=" * 80)
    print("Phase 2: 個別株版バックテスト分析")
    print("=" * 80)

    # 1. データ読み込み
    print("\n[1] 取引データ読み込み...")
    trades_df = pd.read_csv(PROCESSED_DIR / 'trades_v2.csv')
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    print(f"    {len(trades_df)}日分のトレード記録")

    # 2. 株価データ取得
    stock_data = fetch_and_cache_stocks()

    # 3. バックテスト実行
    perf_df, stats, ticker_profit, ticker_long, ticker_short = run_backtest(trades_df, stock_data)

    # 4. 仮採用業種分析
    tentative_df = analyze_tentative_sectors(trades_df, stock_data)

    # 5. 出力
    print("\n[5] 結果を出力...")

    # CSV
    output_csv = OUTPUT_DIR / 'phase2_backtest_analysis.csv'
    perf_df.to_csv(output_csv, index=False)
    print(f"    パフォーマンス: {output_csv}")

    # レポート
    report_file = REPORTS_DIR / 'phase2_backtest_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 2: 個別株版バックテスト結果\n")
        f.write("=" * 80 + "\n\n")

        f.write("[成績]\n")
        f.write(f"  最終資本: ¥{stats['final_equity']:,.0f}\n")
        f.write(f"  総リターン: {stats['total_return']*100:+.2f}%\n")
        f.write(f"  年率リターン: {stats['annual_return']*100:+.2f}%\n")
        f.write(f"  Sharpe Ratio: {stats['sharpe']:.3f}\n")
        f.write(f"  Max Drawdown: {stats['max_dd']*100:.2f}%\n")
        f.write(f"  Calmar Ratio: {stats['calmar']:.3f}\n")
        f.write(f"  勝率: {stats['win_rate']*100:.2f}%\n")
        f.write(f"  取引日数: {len(perf_df)}\n\n")

        f.write("[仮採用業種の評価]\n")
        f.write(tentative_df.to_string(index=False))
        f.write("\n\n")

        f.write("[判定基準]\n")
        f.write("  ADOPT: 採用（トータルPnL > 0）\n")
        f.write("  MARGINAL: 検討中（-0.001 <= PnL <= 0）\n")
        f.write("  REJECT: 除外（PnL < -0.001）\n")

    print(f"    レポート: {report_file}")

    print("\n" + "=" * 80)
    print("完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
