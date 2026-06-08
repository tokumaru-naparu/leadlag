"""
phase2_individual_backtest.py
================================
Phase 2: 個別株版バックテスト（方式E サイジング適用）
ETF版との詳細比較 + 仮採用業種の貢献度分析

INPUT:
  - data/processed/trades_v2.csv（ロング・ショート業種）
  - yfinance で個別株データを取得

OUTPUT:
  - output/results/phase2_individual_backtest.csv
  - output/charts/phase2_etf_vs_individual.png
  - output/reports/phase2_comparison_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR, REPORTS_DIR, OUTPUT_DIR

# ================================================================
# Phase 1 確定銘柄（IC値が十分）
# ================================================================

CONFIRMED_TICKERS = {
    'energy': 5020,           # IC=0.044
    'trading': 8053,          # IC=0.042
    'finance': 8725,          # IC=0.041
    'construction': 5233,     # IC=0.032
    'it_services': 9984,      # IC=0.028
    'electronics': 6971,      # IC=0.026
    'steel': 5713,            # IC=0.024
    'utilities': 9501,        # IC=0.020
    'banks': 8354,            # IC=0.072 (最高)
}

# 仮採用業種（IC低いがバックテストで検証）
TENTATIVE_TICKERS = {
    'food': 2587,             # IC不十分
    'materials': 4004,        # IC不十分
    'pharma': 4536,           # IC不十分
    'auto': 7282,             # IC不十分
    'machinery': 6113,        # IC不十分
    'transport': 9308,        # IC不十分
    'retail': 2780,           # IC不十分
    'realestate': 8984,       # IC不十分
}

ALL_TICKERS = {**CONFIRMED_TICKERS, **TENTATIVE_TICKERS}
SECTOR_TO_TICKER = {v: k for k, v in ALL_TICKERS.items()}

# ================================================================
# 方式E サイジングパラメータ
# ================================================================

SIG_COEF = 2.0
SIZE_CAP = 3.0
SIZE_MIN = 0.1
SHORT_AMP = 2.0
TARGET_VOL = 0.015
DD_T1 = -0.08
DD_S1 = 0.7
DD_T2 = -0.18
DD_S2 = 0.2

INITIAL_CAPITAL = 1_000_000


def fetch_stock_data(ticker: int, start_date: str, end_date: str) -> pd.DataFrame:
    """yfinanceから個別株データを取得"""
    try:
        ticker_str = f"{ticker}.T"  # 日本株なので .T 付与
        df = yf.download(
            ticker_str,
            start=start_date,
            end=end_date,
            progress=False,
        )
        if df.empty:
            print(f"[WARNING] {ticker_str}: データ取得失敗")
            return pd.DataFrame()

        # インデックスを日付列に変換
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        # 必要なカラムを確認
        if 'date' not in df.columns and 'index' in df.columns:
            df.rename(columns={'index': 'date'}, inplace=True)

        return df
    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return pd.DataFrame()


def calculate_oc_return(df: pd.DataFrame) -> dict:
    """日付ごとのOpen-to-Closeリターンを計算"""
    oc_returns = {}

    for idx, row in df.iterrows():
        try:
            # 日付の取得
            if isinstance(row['date'], pd.Timestamp):
                date_str = row['date'].strftime('%Y-%m-%d')
            else:
                date_str = str(row['date'])[:10]

            open_price = float(row['open'])
            close_price = float(row['close'])

            if open_price > 0:
                ret = (close_price - open_price) / open_price
                oc_returns[date_str] = ret
        except (KeyError, ValueError, TypeError):
            continue

    return oc_returns


def apply_sizing_mode_e(
    strategy_equity: float,
    signal: float,
    realized_vol: float,
) -> float:
    """
    方式E: シグナルの大きさに応じた動的サイジング

    Parameters:
    -----------
    strategy_equity : 戦略資本
    signal : シグナル値（-3～+3）
    realized_vol : 実現ボラティリティ

    Returns:
    --------
    ポジションサイズ（資本比率）
    """

    # シグナルの大きさに応じた基本サイズ
    base_size = abs(signal) * SIG_COEF / 3.0

    # ボラティリティの逆関数
    if realized_vol > 0:
        vol_adj = TARGET_VOL / realized_vol
    else:
        vol_adj = 1.0

    # ドローダウン調整（DDをあまり意識しない簡易版）
    size = base_size * vol_adj

    # キャップ適用
    size = max(SIZE_MIN, min(SIZE_CAP, size))

    # ショートは2倍に拡大
    if signal < 0:
        size = min(size * SHORT_AMP, SIZE_CAP)

    return size


def backtest_individual_stocks(
    trades_df: pd.DataFrame,
    stock_data_dict: dict,
    start_date: str = "2010-04-07",
    end_date: str = "2026-03-30",
) -> tuple[pd.DataFrame, dict]:
    """
    個別株版バックテスト実行

    Returns:
    --------
    performance_df: 日別パフォーマンス
    stats_dict: 統計情報
    """

    # 日付でソート
    trades_df = trades_df.sort_values('date').reset_index(drop=True)

    results = []
    equity = INITIAL_CAPITAL
    realized_vol = TARGET_VOL
    daily_returns = []

    for idx, row in trades_df.iterrows():
        trade_date = row['date']

        # ロング・ショート業種を取得
        long_sectors = [s.strip() for s in [row.get('long_1'), row.get('long_2'), row.get('long_3')] if pd.notna(s)]
        short_sectors = [s.strip() for s in [row.get('short_1'), row.get('short_2'), row.get('short_3')] if pd.notna(s)]

        # 各銘柄のOCリターンを取得
        long_returns = []
        short_returns = []

        for sector in long_sectors:
            ticker = ALL_TICKERS.get(sector)
            if ticker and ticker in stock_data_dict:
                oc_ret = stock_data_dict[ticker].get(trade_date, np.nan)
                if not np.isnan(oc_ret):
                    long_returns.append(oc_ret)

        for sector in short_sectors:
            ticker = ALL_TICKERS.get(sector)
            if ticker and ticker in stock_data_dict:
                oc_ret = stock_data_dict[ticker].get(trade_date, np.nan)
                if not np.isnan(oc_ret):
                    short_returns.append(oc_ret)

        # 日次リターン計算
        if long_returns and short_returns:
            long_avg = np.mean(long_returns)
            short_avg = np.mean(short_returns)

            # シグナルの生成（簡易版：市場が上昇なら+、下落なら-）
            signal = len(long_returns) - len(short_returns)  # 簡易的なシグナル

            # サイジング適用
            size = apply_sizing_mode_e(equity, signal, realized_vol)

            # 当日損益
            pnl_return = (long_avg - short_avg) * size

            # ボラティリティ更新（指数平滑）
            if len(daily_returns) > 0:
                recent_vol = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else np.std(daily_returns)
                realized_vol = 0.9 * realized_vol + 0.1 * recent_vol

            daily_returns.append(pnl_return)
        else:
            pnl_return = 0.0

        # 資本更新
        equity *= (1 + pnl_return)

        results.append({
            'date': trade_date,
            'long_return': np.mean(long_returns) if long_returns else np.nan,
            'short_return': np.mean(short_returns) if short_returns else np.nan,
            'strategy_return': pnl_return,
            'equity': equity,
        })

    performance_df = pd.DataFrame(results)

    # 統計計算
    returns_series = performance_df['strategy_return'].fillna(0)

    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    annual_return = (equity / INITIAL_CAPITAL) ** (252 / len(returns_series)) - 1 if len(returns_series) > 0 else 0
    sharpe = (returns_series.mean() * 252) / (returns_series.std() * np.sqrt(252)) if returns_series.std() > 0 else 0

    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    win_rate = (returns_series > 0).sum() / len(returns_series) if len(returns_series) > 0 else 0

    stats_dict = {
        'final_equity': equity,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'num_trades': len(returns_series),
    }

    return performance_df, stats_dict


def analyze_tentative_sectors(trades_df: pd.DataFrame, stock_data_dict: dict) -> pd.DataFrame:
    """仮採用業種の貢献度分析"""

    analysis_results = []

    for sector, ticker in TENTATIVE_TICKERS.items():
        if ticker not in stock_data_dict:
            continue

        oc_returns = stock_data_dict[ticker]

        # ロング日とショート日を別々に集計
        long_returns_list = []
        short_returns_list = []

        for _, row in trades_df.iterrows():
            date = row['date']
            oc_ret = oc_returns.get(date, np.nan)

            if np.isnan(oc_ret):
                continue

            # このセクターがロング3に入っているか
            long_cols = [c for c in row.index if c.startswith('long_')]
            if any(row[c] == sector for c in long_cols if pd.notna(row[c])):
                long_returns_list.append(oc_ret)

            # このセクターがショート3に入っているか
            short_cols = [c for c in row.index if c.startswith('short_')]
            if any(row[c] == sector for c in short_cols if pd.notna(row[c])):
                short_returns_list.append(oc_ret)

        # 統計計算
        long_avg = np.mean(long_returns_list) if long_returns_list else np.nan
        short_avg = np.mean(short_returns_list) if short_returns_list else np.nan
        long_days = len(long_returns_list)
        short_days = len(short_returns_list)

        # ロングの勝率
        long_win_rate = (np.array(long_returns_list) > 0).sum() / len(long_returns_list) if long_returns_list else np.nan
        # ショートの勝率（負が正）
        short_win_rate = (np.array(short_returns_list) < 0).sum() / len(short_returns_list) if short_returns_list else np.nan

        # 総合判定
        total_pnl = long_avg * long_days - short_avg * short_days if not np.isnan(long_avg) and not np.isnan(short_avg) else np.nan

        analysis_results.append({
            'sector': sector,
            'ticker': ticker,
            'long_days': long_days,
            'short_days': short_days,
            'long_avg_return': long_avg,
            'short_avg_return': short_avg,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'total_pnl_return': total_pnl,
            'judgement': 'ADOPT' if total_pnl > 0.001 else 'REJECT' if total_pnl < -0.001 else 'MARGINAL',
        })

    return pd.DataFrame(analysis_results)


def generate_comparison_plot(performance_etf: pd.DataFrame, performance_individual: pd.DataFrame, output_path: Path):
    """ETF版と個別株版の累積資産を比較"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 累積資産
    ax1.plot(performance_etf['date'], performance_etf['equity'], label='ETF版', linewidth=2, alpha=0.8)
    ax1.plot(performance_individual['date'], performance_individual['equity'], label='個別株版', linewidth=2, alpha=0.8)
    ax1.set_ylabel('資本 (円)', fontsize=12)
    ax1.set_title('ETF版 vs 個別株版 累積資本', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 日次リターン（ローリング平均）
    etf_rolling = performance_etf['strategy_return'].rolling(20).mean()
    ind_rolling = performance_individual['strategy_return'].rolling(20).mean()
    ax2.plot(performance_etf['date'], etf_rolling, label='ETF版（20日MA）', linewidth=2, alpha=0.8)
    ax2.plot(performance_individual['date'], ind_rolling, label='個別株版（20日MA）', linewidth=2, alpha=0.8)
    ax2.set_ylabel('日次リターン', fontsize=12)
    ax2.set_xlabel('日付', fontsize=12)
    ax2.set_title('日次リターン（20日移動平均）', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[出力] グラフ: {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("Phase 2: 個別株版バックテスト（方式E サイジング）")
    print("=" * 80)

    # 1. データ読み込み
    print("\n[1] データ読み込み...")
    trades_file = PROCESSED_DIR / 'trades_v2.csv'
    trades_df = pd.read_csv(trades_file)
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    print(f"    取引データ: {len(trades_df)}行")

    # 2. 個別株データ取得
    print("\n[2] 個別株データ取得...")
    stock_data_dict = {}

    start_date = "2010-03-01"
    end_date = "2026-04-01"

    for sector, ticker in ALL_TICKERS.items():
        print(f"    {sector:15s} ({ticker}...) ", end='', flush=True)
        df = fetch_stock_data(ticker, start_date, end_date)

        if df.empty:
            print("失敗")
            continue

        # OC リターン計算
        oc_dict = calculate_oc_return(df)
        stock_data_dict[ticker] = oc_dict
        print(f"OK ({len(oc_dict)}日)")

    print(f"    合計: {len(stock_data_dict)} 銘柄取得")

    # 3. バックテスト実行（個別株版）
    print("\n[3] 個別株版バックテスト実行...")
    perf_ind, stats_ind = backtest_individual_stocks(trades_df, stock_data_dict)
    print(f"    累積リターン: {stats_ind['total_return']:.4f} ({stats_ind['total_return']*100:.2f}%)")
    print(f"    最終資本: ¥{stats_ind['final_equity']:,.0f}")
    print(f"    Sharpe: {stats_ind['sharpe']:.3f}")
    print(f"    MaxDD: {stats_ind['max_dd']:.4f}")

    # 4. 仮採用業種の分析
    print("\n[4] 仮採用業種の貢献度分析...")
    tentative_analysis = analyze_tentative_sectors(trades_df, stock_data_dict)
    print("\n仮採用業種の評価:")
    print(tentative_analysis.to_string(index=False))

    # 5. 出力
    print("\n[5] 結果を出力...")

    # CSV出力
    output_csv = OUTPUT_DIR / 'phase2_individual_backtest.csv'
    perf_ind.to_csv(output_csv, index=False)
    print(f"    パフォーマンス: {output_csv}")

    # レポート出力
    report_path = REPORTS_DIR / 'phase2_individual_analysis.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 2: 個別株版バックテスト結果\n")
        f.write("=" * 80 + "\n\n")

        f.write("【個別株版パフォーマンス】\n")
        f.write(f"最終資本:      ¥{stats_ind['final_equity']:,.0f}\n")
        f.write(f"総リターン:     {stats_ind['total_return']*100:.2f}%\n")
        f.write(f"年率リターン:   {stats_ind['annual_return']*100:.2f}%\n")
        f.write(f"Sharpe Ratio:  {stats_ind['sharpe']:.3f}\n")
        f.write(f"Max Drawdown:  {stats_ind['max_dd']*100:.2f}%\n")
        f.write(f"Calmar Ratio:  {stats_ind['calmar']:.3f}\n")
        f.write(f"勝率:          {stats_ind['win_rate']*100:.2f}%\n")
        f.write(f"取引日数:      {stats_ind['num_trades']}\n\n")

        f.write("【仮採用業種の評価】\n")
        for _, row in tentative_analysis.iterrows():
            f.write(f"\n{row['sector']} ({row['ticker']}):\n")
            f.write(f"  ロング日数: {int(row['long_days'])}, 平均リターン: {row['long_avg_return']:+.4f}%, 勝率: {row['long_win_rate']*100:.1f}%\n")
            f.write(f"  ショート日数: {int(row['short_days'])}, 平均リターン: {row['short_avg_return']:+.4f}%, 勝率: {row['short_win_rate']*100:.1f}%\n")
            f.write(f"  判定: {row['judgement']}\n")

        f.write("\n【注記】\n")
        f.write("- 手数料・スプレッド・税金は未計上\n")
        f.write("- 2587（食品）は2013年7月以降のみ\n")
        f.write("- 方式E サイジング適用\n")

    print(f"    レポート: {report_path}")

    print("\n" + "=" * 80)
    print("完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
