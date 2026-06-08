"""
01_fetch_data.py
日米業種ETFのOpen/Closeデータを取得してCSVに保存する
論文: 部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略
期間: 2010-01-01 〜 2025-12-31
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import datetime

# ============================================================
# 設定
# ============================================================

START_DATE = "2010-01-01"
# 終了日は実行時の今日を使う
END_DATE = datetime.date.today().strftime('%Y-%m-%d')

# 米国業種ETF（S&P500 Select Sector SPDR）
# 論文記載の11業種
US_TICKERS = [
    "XLB",   # Materials（素材）
    "XLC",   # Communication Services（通信サービス）
    "XLE",   # Energy（エネルギー）
    "XLF",   # Financials（金融）
    "XLI",   # Industrials（資本財）
    "XLK",   # Information Technology（IT）
    "XLP",   # Consumer Staples（生活必需品）
    "XLRE",  # Real Estate（不動産）
    "XLU",   # Utilities（公益）
    "XLV",   # Health Care（ヘルスケア）
    "XLY",   # Consumer Discretionary（一般消費財）
]

# 日本業種ETF（NEXT FUNDS TOPIX-17業種別）
# 論文記載の17業種
JP_TICKERS = [
    "1617.T",  # 食品
    "1618.T",  # エネルギー資源
    "1619.T",  # 建設・資材
    "1620.T",  # 素材・化学
    "1621.T",  # 医薬品
    "1622.T",  # 自動車・輸送機
    "1623.T",  # 鉄鋼・非鉄
    "1624.T",  # 機械
    "1625.T",  # 電機・精密
    "1626.T",  # 情報通信・サービスその他
    "1627.T",  # 電力・ガス
    "1628.T",  # 運輸・物流
    "1629.T",  # 商社・卸売
    "1630.T",  # 小売
    "1631.T",  # 銀行
    "1632.T",  # 金融（除く銀行）
    "1633.T",  # 不動産
]

# ============================================================
# データ保存先
# ============================================================

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# 取得関数
# ============================================================

def fetch_ohlc(tickers: list[str], label: str) -> pd.DataFrame:
    """
    複数ティッカーのOpen・Closeデータを取得してひとつのDataFrameにまとめる
    
    返り値のカラム例:
        (Open, XLB), (Open, XLE), ..., (Close, XLB), (Close, XLE), ...
    """
    print(f"\n=== {label} のデータ取得開始 ===")
    print(f"対象: {tickers}")
    print(f"期間: {START_DATE} 〜 {END_DATE}")

    raw = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,   # 株式分割・配当を自動調整（論文と同じ処理）
        progress=True,
    )

    # Open と Close だけ抽出
    df = raw[["Open", "Close"]]

    print(f"取得完了: {len(df)} 営業日分")
    print(f"期間: {df.index[0].date()} 〜 {df.index[-1].date()}")

    # 欠損値の確認
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("\n⚠️  欠損値あり（上場前・廃止後など）:")
        print(missing)
    else:
        print("✅ 欠損値なし")

    return df


def save_csv(df: pd.DataFrame, filename: str):
    """DataFrameをCSVに保存"""
    path = DATA_DIR / filename
    df.to_csv(path)
    print(f"💾 保存: {path}")


# ============================================================
# メイン処理
# ============================================================

if __name__ == "__main__":

    # 米国ETF取得
    us_df = fetch_ohlc(US_TICKERS, "米国業種ETF")
    save_csv(us_df, "us_etf.csv")

    # 日本ETF取得
    jp_df = fetch_ohlc(JP_TICKERS, "日本業種ETF")
    save_csv(jp_df, "jp_etf.csv")

    # 簡易サマリー表示
    print("\n" + "="*50)
    print("取得完了サマリー")
    print("="*50)
    print(f"米国ETF: {us_df.shape[0]}日 × {len(US_TICKERS)}銘柄")
    print(f"日本ETF: {jp_df.shape[0]}日 × {len(JP_TICKERS)}銘柄")
    print(f"\n保存先: {DATA_DIR}")
    print("\n次のステップ: 02_backtest.py を実行してください")