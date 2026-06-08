"""
phase2_backtest_v2.py
=================================
Phase 2修正版: データ不足銘柄変更 + 方式Eサイジング検証

主な修正:
1. データ不足銘柄を流動性の高いものに変更
   - 素材化学: 4004 → 4182（三菱ガス化学）
   - 機械: 6113 → 6301（コマツ）
   - 運輸: 9308 → 9020（JR東日本）
   - 不動産: 8984 → 8801（三井不動産）
2. 方式Eサイジングの実装と検証
3. 等サイズ版との比較

OUTPUT:
  - output/results/phase2_v2_results.csv
  - output/charts/phase2_v2_comparison.png
  - output/reports/phase2_v2_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR, REPORTS_DIR, OUTPUT_DIR

# ================================================================
# 修正済み銘柄マッピング（16業種）
# ================================================================

SECTOR_STOCKS = {
    # 採用確定（9業種）
    'energy': 5020,           # ENEOS
    'trading': 8053,          # 住友商事
    'finance': 8725,          # MS&AD
    'construction': 5233,     # 太平洋セメント
    'it_services': 9984,      # ソフトバンクG
    'electronics': 6971,      # 京セラ
    'steel': 5713,            # 住友金属鉱山
    'utilities': 9501,        # 東京電力
    'banks': 8354,            # ふくおかFG

    # 仮採用（4業種）
    'pharma': 4536,           # 持田製薬
    'auto': 7282,             # 豊田合成
    'food': 2587,             # サントリーHD

    # 修正対象（データ不足だった4業種を変更）
    'materials': 4182,        # 三菱ガス化学（変更）
    'machinery': 6301,        # コマツ（変更）
    'transport': 9020,        # JR東日本（変更）
    'realestate': 8801,       # 三井不動産（変更）
}

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
    'pharma': '医薬品',
    'auto': '自動車',
    'food': '食品',
    'materials': '素材化学',
    'machinery': '機械',
    'transport': '運輸',
    'realestate': '不動産',
}

# 方式E サイジングパラメータ
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


def check_data_availability() -> pd.DataFrame:
    """各銘柄のデータ利用可能性を確認"""
    print("[1] 銘柄別データ確認...")

    results = []
    for sector, ticker in SECTOR_STOCKS.items():
        sector_name = SECTOR_NAMES[sector]
        print(f"  {sector_name:15s} ({ticker}) ...", end='', flush=True)

        try:
            df = yf.download(
                f"{ticker}.T",
                start="2016-05-16",
                end="2026-03-26",
                progress=False,
                auto_adjust=True,
            )

            if len(df) > 0:
                start_date = df.index[0].strftime('%Y-%m-%d')
                end_date = df.index[-1].strftime('%Y-%m-%d')
                status = '✓ OK' if len(df) >= 1000 else '⚠ 不足'
                print(f" {len(df):4d}日 ({start_date}～{end_date}) {status}")

                results.append({
                    'sector': sector,
                    'sector_name': sector_name,
                    'ticker': ticker,
                    'num_days': len(df),
                    'start_date': start_date,
                    'end_date': end_date,
                    'available': len(df) >= 1000,
                })
            else:
                print(" NO DATA")
                results.append({
                    'sector': sector,
                    'sector_name': sector_name,
                    'ticker': ticker,
                    'num_days': 0,
                    'start_date': 'N/A',
                    'end_date': 'N/A',
                    'available': False,
                })
        except Exception as e:
            print(f" ERROR")
            results.append({
                'sector': sector,
                'sector_name': sector_name,
                'ticker': ticker,
                'num_days': 0,
                'start_date': 'N/A',
                'end_date': 'N/A',
                'available': False,
            })

    df_avail = pd.DataFrame(results)
    print(f"\n  利用可能: {df_avail['available'].sum()}/{len(df_avail)}\n")
    return df_avail


def fetch_stock_data() -> dict:
    """全銘柄の株価データを取得"""
    print("[2] 株価データ取得...")

    stock_data = {}
    for sector, ticker in SECTOR_STOCKS.items():
        sector_name = SECTOR_NAMES[sector]
        print(f"  {sector_name:15s} ({ticker}) ...", end='', flush=True)

        try:
            df = yf.download(
                f"{ticker}.T",
                start="2016-05-16",
                end="2026-03-26",
                progress=False,
                auto_adjust=True,
            )

            if len(df) == 0:
                print(" NO DATA")
                continue

            # Open-Close リターンを計算
            oc_returns = (df['Close'] - df['Open']) / df['Open']
            oc_df = oc_returns.reset_index()
            oc_df.columns = ['date', 'oc_return']
            oc_df['date'] = pd.to_datetime(oc_df['date'])

            stock_data[ticker] = oc_df
            print(f" OK ({len(oc_df)}日)")

        except Exception as e:
            print(f" ERROR: {str(e)[:20]}")

    print(f"\n  成功: {len(stock_data)}/{len(SECTOR_STOCKS)}\n")
    return stock_data


def apply_sizing_mode_e(
    returns_series: pd.Series,
    dd_current: float,
) -> float:
    """方式Eのサイジング適用

    Parameters:
    -----------
    returns_series : 累積リターン系列
    dd_current : 現在のドローダウン

    Returns:
    --------
    サイジング係数（0.1～3.0）
    """

    # ボラティリティ調整
    recent_vol = returns_series.tail(20).std()
    if recent_vol > 0:
        vol_adj = TARGET_VOL / recent_vol
    else:
        vol_adj = 1.0

    # ベースサイズ
    base_size = 1.0 * SIG_COEF / 3.0  # 簡易版：シグナル強度=1.0と仮定

    # ボラ調整
    size = base_size * vol_adj
    size = max(SIZE_MIN, min(SIZE_CAP, size))

    # ドローダウン調整
    if dd_current <= DD_T2:
        scale = DD_S2  # -18%以下 → 0.2倍
    elif dd_current <= DD_T1:
        scale = DD_S1  # -8%以下  → 0.7倍
    else:
        scale = 1.0

    size *= scale

    return max(SIZE_MIN, min(SIZE_CAP, size))


def run_backtest(
    trades_df: pd.DataFrame,
    stock_data: dict,
    use_sizing: bool = True,
) -> tuple:
    """バックテスト実行"""

    mode_str = "方式E版" if use_sizing else "等サイズ版"
    print(f"\n[3] バックテスト実行 ({mode_str})...")

    daily_perf = []
    equity = INITIAL_CAPITAL
    returns_list = []

    for idx, row in trades_df.iterrows():
        date = row['date']

        # ロング・ショート銘柄の平均リターン
        long_returns = []
        short_returns = []

        for col in ['long_1', 'long_2', 'long_3']:
            if pd.isna(row[col]):
                continue

            sector = row[col]
            ticker = SECTOR_STOCKS.get(sector)
            if not ticker or ticker not in stock_data:
                continue

            oc_df = stock_data[ticker]
            oc_row = oc_df[oc_df['date'] == date]
            if len(oc_row) > 0:
                oc_ret = oc_row.iloc[0]['oc_return']
                if not np.isnan(oc_ret):
                    long_returns.append(oc_ret)

        for col in ['short_1', 'short_2', 'short_3']:
            if pd.isna(row[col]):
                continue

            sector = row[col]
            ticker = SECTOR_STOCKS.get(sector)
            if not ticker or ticker not in stock_data:
                continue

            oc_df = stock_data[ticker]
            oc_row = oc_df[oc_df['date'] == date]
            if len(oc_row) > 0:
                oc_ret = oc_row.iloc[0]['oc_return']
                if not np.isnan(oc_ret):
                    short_returns.append(oc_ret)

        # 日次リターン計算
        if long_returns and short_returns:
            long_avg = np.mean(long_returns)
            short_avg = np.mean(short_returns)
            base_return = long_avg - short_avg

            # サイジング適用
            if use_sizing:
                # 現在のドローダウン計算
                if len(returns_list) > 0:
                    cumulative = (1 + np.array(returns_list)).cumprod()
                    running_max = np.maximum.accumulate(cumulative)
                    dd = (cumulative[-1] - running_max[-1]) / running_max[-1]
                else:
                    dd = 0.0

                # サイジング係数を計算
                recent_returns = pd.Series(returns_list[-20:] if len(returns_list) >= 20 else returns_list)
                recent_vol = recent_returns.std() if len(recent_returns) > 0 else TARGET_VOL

                if recent_vol > 0:
                    vol_adj = TARGET_VOL / recent_vol
                else:
                    vol_adj = 1.0

                base_size = 1.0
                size = base_size * vol_adj
                size = max(SIZE_MIN, min(SIZE_CAP, size))

                # DD調整
                if dd <= DD_T2:
                    scale = DD_S2
                elif dd <= DD_T1:
                    scale = DD_S1
                else:
                    scale = 1.0

                size *= scale
                daily_return = base_return * size
            else:
                # 等サイズ版：サイジングなし
                daily_return = base_return

            returns_list.append(daily_return)
        else:
            daily_return = 0.0
            returns_list.append(daily_return)

        # 資本更新
        equity *= (1 + daily_return)

        daily_perf.append({
            'date': date,
            'daily_return': daily_return,
            'equity': equity,
        })

    perf_df = pd.DataFrame(daily_perf)

    # 統計計算
    returns_series = pd.Series(returns_list)
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # 年率リターン（1年=252営業日）
    num_years = len(returns_series) / 252
    annual_return = (equity / INITIAL_CAPITAL) ** (1 / num_years) - 1 if num_years > 0 else 0

    # Sharpe Ratio
    if returns_series.std() > 0:
        sharpe = (returns_series.mean() * 252) / (returns_series.std() * np.sqrt(252))
    else:
        sharpe = 0

    # Max Drawdown & Calmar
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    # Win Rate
    win_rate = (returns_series > 0).sum() / len(returns_series) if len(returns_series) > 0 else 0

    stats = {
        'final_equity': equity,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'num_trades': len(returns_series),
    }

    print(f"  最終資本: {equity:,.0f}円")
    print(f"  総リターン: {total_return*100:.2f}%")
    print(f"  年率リターン: {annual_return*100:.2f}%")
    print(f"  Sharpe: {sharpe:.3f}, MaxDD: {max_dd*100:.2f}%")

    return perf_df, stats


def compare_etf_vs_individual() -> dict:
    """ETF版の成績を取得（既存データから）"""
    print("\n[4] ETF版の成績を読み込み...")

    etf_perf_file = PROCESSED_DIR / 'performance_etf_full.csv'
    if etf_perf_file.exists():
        df = pd.read_csv(etf_perf_file)

        # 最後の行を使用
        if len(df) > 0:
            last_row = df.iloc[-1]

            # 統計計算
            returns = df['strategy_return'].fillna(0)
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max

            total_return = cumulative.iloc[-1] - 1
            num_years = len(returns) / 252
            annual_return = (cumulative.iloc[-1]) ** (1 / num_years) - 1 if num_years > 0 else 0

            if returns.std() > 0:
                sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            else:
                sharpe = 0

            max_dd = drawdown.min()
            calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
            win_rate = (returns > 0).sum() / len(returns)

            stats = {
                'final_equity': last_row.get('equity', 0),
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe': sharpe,
                'max_dd': max_dd,
                'calmar': calmar,
                'win_rate': win_rate,
                'num_trades': len(returns),
            }

            print(f"  ETF版 年率: {annual_return*100:.2f}%, Sharpe: {sharpe:.3f}")
            return stats

    print("  ETF版データなし")
    return None


def create_comparison_plot(
    perf_equal: pd.DataFrame,
    stats_equal: dict,
    perf_sizing: pd.DataFrame,
    stats_sizing: dict,
    output_path: Path,
):
    """等サイズ版 vs 方式E版 比較グラフ"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 累積資本
    ax = axes[0, 0]
    ax.plot(perf_equal['date'], perf_equal['equity'], label='等サイズ版', linewidth=2, alpha=0.8)
    ax.plot(perf_sizing['date'], perf_sizing['equity'], label='方式E版', linewidth=2, alpha=0.8)
    ax.set_ylabel('資本 (円)', fontsize=11)
    ax.set_title('累積資本', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 日次リターン（ローリング20日MA）
    ax = axes[0, 1]
    equal_ma = perf_equal['daily_return'].rolling(20).mean()
    sizing_ma = perf_sizing['daily_return'].rolling(20).mean()
    ax.plot(perf_equal['date'], equal_ma, label='等サイズ版', linewidth=2, alpha=0.8)
    ax.plot(perf_sizing['date'], sizing_ma, label='方式E版', linewidth=2, alpha=0.8)
    ax.set_ylabel('日次リターン', fontsize=11)
    ax.set_title('日次リターン（20日MA）', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ドローダウン比較
    ax = axes[1, 0]
    equal_cum = (1 + perf_equal['daily_return']).cumprod()
    sizing_cum = (1 + perf_sizing['daily_return']).cumprod()
    equal_dd = (equal_cum - equal_cum.expanding().max()) / equal_cum.expanding().max()
    sizing_dd = (sizing_cum - sizing_cum.expanding().max()) / sizing_cum.expanding().max()
    ax.plot(perf_equal['date'], equal_dd, label='等サイズ版', linewidth=2, alpha=0.8)
    ax.plot(perf_sizing['date'], sizing_dd, label='方式E版', linewidth=2, alpha=0.8)
    ax.set_ylabel('ドローダウン', fontsize=11)
    ax.set_title('ドローダウン推移', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 統計情報テーブル
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [
        ['指標', '等サイズ版', '方式E版'],
        ['年率', f"{stats_equal['annual_return']*100:.2f}%", f"{stats_sizing['annual_return']*100:.2f}%"],
        ['Sharpe', f"{stats_equal['sharpe']:.3f}", f"{stats_sizing['sharpe']:.3f}"],
        ['MaxDD', f"{stats_equal['max_dd']*100:.2f}%", f"{stats_sizing['max_dd']*100:.2f}%"],
        ['Calmar', f"{stats_equal['calmar']:.3f}", f"{stats_sizing['calmar']:.3f}"],
        ['勝率', f"{stats_equal['win_rate']*100:.1f}%", f"{stats_sizing['win_rate']*100:.1f}%"],
    ]

    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.35, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # ヘッダーを強調
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  グラフ: {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("Phase 2修正版: データ不足銘柄変更 + 方式Eサイジング検証")
    print("=" * 80)

    # Step 1: データ確認
    avail_df = check_data_availability()

    # Step 2: 株価データ取得
    stock_data = fetch_stock_data()

    # Step 3: 取引データ読み込み
    print("[3] 取引データ読み込み...")
    trades_df = pd.read_csv(PROCESSED_DIR / 'trades_v2.csv')
    trades_df['date'] = pd.to_datetime(trades_df['date'])

    # Step 4: バックテスト実行（2バージョン）
    perf_equal, stats_equal = run_backtest(trades_df, stock_data, use_sizing=False)
    perf_sizing, stats_sizing = run_backtest(trades_df, stock_data, use_sizing=True)

    # Step 5: ETF版比較
    stats_etf = compare_etf_vs_individual()

    # Step 6: 出力
    print("\n[5] 結果を出力...")

    # CSV
    output_csv = OUTPUT_DIR / 'phase2_v2_results.csv'
    perf_sizing.to_csv(output_csv, index=False)
    print(f"  CSV: {output_csv}")

    # グラフ
    chart_file = OUTPUT_DIR.parent / 'charts' / 'phase2_v2_comparison.png'
    chart_file.parent.mkdir(parents=True, exist_ok=True)
    create_comparison_plot(perf_equal, stats_equal, perf_sizing, stats_sizing, chart_file)

    # レポート
    report_file = REPORTS_DIR / 'phase2_v2_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 2修正版: 最終レポート\n")
        f.write("=" * 80 + "\n\n")

        f.write("[銘柄データ確認]\n")
        f.write(avail_df.to_string(index=False))
        f.write("\n\n")

        f.write("[等サイズ版 vs 方式E版 比較]\n")
        f.write("指標          等サイズ版   方式E版      改善度\n")
        f.write("-" * 60 + "\n")
        f.write(f"年率          {stats_equal['annual_return']*100:>6.2f}%    {stats_sizing['annual_return']*100:>6.2f}%    ")
        f.write(f"{(stats_sizing['annual_return']-stats_equal['annual_return'])*100:+6.2f}%\n")
        f.write(f"Sharpe        {stats_equal['sharpe']:>6.3f}    {stats_sizing['sharpe']:>6.3f}    ")
        f.write(f"{stats_sizing['sharpe']-stats_equal['sharpe']:+6.3f}\n")
        f.write(f"MaxDD         {stats_equal['max_dd']*100:>6.2f}%    {stats_sizing['max_dd']*100:>6.2f}%    ")
        f.write(f"{(stats_sizing['max_dd']-stats_equal['max_dd'])*100:+6.2f}%\n")
        f.write(f"Calmar        {stats_equal['calmar']:>6.3f}    {stats_sizing['calmar']:>6.3f}    ")
        f.write(f"{stats_sizing['calmar']-stats_equal['calmar']:+6.3f}\n")
        f.write(f"勝率          {stats_equal['win_rate']*100:>6.2f}%    {stats_sizing['win_rate']*100:>6.2f}%    ")
        f.write(f"{(stats_sizing['win_rate']-stats_equal['win_rate'])*100:+6.2f}%\n\n")

        if stats_etf:
            f.write("[ETF版 vs 個別株版（方式E）比較]\n")
            f.write("指標          ETF版        個別株版     差分\n")
            f.write("-" * 60 + "\n")
            f.write(f"年率          {stats_etf['annual_return']*100:>6.2f}%    {stats_sizing['annual_return']*100:>6.2f}%    ")
            f.write(f"{(stats_sizing['annual_return']-stats_etf['annual_return'])*100:+6.2f}%\n")
            f.write(f"Sharpe        {stats_etf['sharpe']:>6.3f}    {stats_sizing['sharpe']:>6.3f}    ")
            f.write(f"{stats_sizing['sharpe']-stats_etf['sharpe']:+6.3f}\n")
            f.write(f"MaxDD         {stats_etf['max_dd']*100:>6.2f}%    {stats_sizing['max_dd']*100:>6.2f}%    ")
            f.write(f"{(stats_sizing['max_dd']-stats_etf['max_dd'])*100:+6.2f}%\n")
            f.write(f"Calmar        {stats_etf['calmar']:>6.3f}    {stats_sizing['calmar']:>6.3f}    ")
            f.write(f"{stats_sizing['calmar']-stats_etf['calmar']:+6.3f}\n\n")

        f.write("[方式Eサイジングの効果]\n")
        sizing_effect = stats_sizing['annual_return'] - stats_equal['annual_return']
        if sizing_effect > 0.001:
            f.write(f"✓ サイジングが機能している（+{sizing_effect*100:.2f}% 改善）\n")
        elif sizing_effect > -0.001:
            f.write(f"△ サイジングの効果が不明確（±{sizing_effect*100:.2f}%）\n")
        else:
            f.write(f"✗ サイジングが逆効果（{sizing_effect*100:.2f}% 悪化）\n")

        f.write("\n[注記]\n")
        f.write("- 手数料・スプレッド・税金は未計上\n")
        f.write("- 小売は前回データ不足のため除外\n")
        f.write("- 修正後16銘柄でバックテスト実施\n")

    print(f"  レポート: {report_file}")

    print("\n" + "=" * 80)
    print("完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
