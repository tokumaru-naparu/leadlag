"""
02_preprocess.py
取得したCSVデータを読み込んで、バックテスト用のリターンデータを作る

やること:
  1. CSVを読み込む
  2. リターンを計算する
     ・米国: Close-to-Close（前日終値→当日終値）← シグナルの材料
     ・日本: Open-to-Close（当日始値→当日終値） ← 戦略の評価対象
  3. 日米共通営業日だけに絞る
  4. 結果を確認して保存する
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ============================================================
# 1. CSVを読み込む
# ============================================================

def load_etf(filename: str) -> pd.DataFrame:
    """
    CSVを読み込む。
    カラムが (Price, Ticker) の2階層になってるので整理する。
    """
    df = pd.read_csv(DATA_DIR / filename, header=[0, 1], index_col=0, parse_dates=True)
    return df


print("=== データ読み込み ===")
us_raw = load_etf("us_etf.csv")
jp_raw = load_etf("jp_etf.csv")

print(f"米国ETF: {us_raw.shape}")
print(f"日本ETF: {jp_raw.shape}")

# ============================================================
# 2. Open / Close を分離する
# ============================================================

# us_raw.columns は ("Open", "XLB"), ("Close", "XLB"), ... という構造
us_open  = us_raw["Open"]   # 形: (日付 × 銘柄数)
us_close = us_raw["Close"]
jp_open  = jp_raw["Open"]
jp_close = jp_raw["Close"]

# ============================================================
# 3. リターンを計算する
# ============================================================

# 米国: Close-to-Close リターン
# 計算式: (今日の終値 / 昨日の終値) - 1
# pct_change() がこれをやってくれる
us_cc = us_close.pct_change()   # cc = Close-to-Close

# 日本: Open-to-Close リターン  
# 計算式: (今日の終値 / 今日の始値) - 1
# 注意: pct_change()は使えない（同じ日の別の列同士の計算）
jp_oc = (jp_close - jp_open) / jp_open   # oc = Open-to-Close

print("\n=== リターン計算完了 ===")
print(f"米国 Close-to-Close リターン: {us_cc.shape}")
print(f"日本 Open-to-Close リターン : {jp_oc.shape}")

# ============================================================
# 4. 日米共通営業日だけに絞る
# ============================================================
# 米国と日本は祝日が違うので取引日がズレる
# 両方にデータがある日だけを使う

common_dates = us_cc.index.intersection(jp_oc.index)

us_cc = us_cc.loc[common_dates]
jp_oc = jp_oc.loc[common_dates]

print(f"\n=== 共通営業日に絞った結果 ===")
print(f"共通営業日数: {len(common_dates)}")
print(f"期間: {common_dates[0].date()} 〜 {common_dates[-1].date()}")

# ============================================================
# 5. 欠損値の処理
# ============================================================
# XLC・XLREは上場前がNaNになってる
# バックテスト内でウィンドウ計算するときに自然と除外されるので
# ここでは確認だけする

print("\n=== 欠損値確認 ===")
print("米国 (最初の5行):")
print(us_cc.head())
print("\n日本 (最初の5行):")
print(jp_oc.head())

us_nan = us_cc.isnull().sum()
us_nan = us_nan[us_nan > 0]
if not us_nan.empty:
    print(f"\n⚠️  米国の欠損: {us_nan.to_dict()}")
else:
    print("\n✅ 米国: 欠損なし（XLC・XLREの上場前以外）")

jp_nan = jp_oc.isnull().sum()
jp_nan = jp_nan[jp_nan > 0]
if not jp_nan.empty:
    print(f"⚠️  日本の欠損: {jp_nan.to_dict()}")
else:
    print("✅ 日本: 欠損なし")

# ============================================================
# 6. 保存する
# ============================================================

us_cc.to_csv(DATA_DIR / "us_cc_returns.csv")
jp_oc.to_csv(DATA_DIR / "jp_oc_returns.csv")

print(f"\n=== 保存完了 ===")
print(f"💾 data/us_cc_returns.csv （米国 Close-to-Close リターン）")
print(f"💾 data/jp_oc_returns.csv （日本 Open-to-Close リターン）")

# ============================================================
# 7. 簡易チェック: 論文の基本統計量と比較
# ============================================================
# 論文 表1 より: 1629.T（商社・卸売）の年率リターン = 20.45%
# Open-to-Closeリターンの年率化: 平均 × 252

print("\n=== 論文との整合性チェック ===")
print("論文 表1 の年率リターン（参考値）と比較:")
print("（注: 論文はClose-to-Closeベース、こちらはOpen-to-Closeなので多少ズレる）")

annual_ret = jp_oc.mean() * 252 * 100  # %表示
print("\n日本業種ETF 年率リターン（Open-to-Close）:")
for ticker, ret in annual_ret.items():
    print(f"  {ticker}: {ret:.2f}%")

print("\n次のステップ: 03_backtest.py でMOMから実装していきます")