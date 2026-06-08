"""
phase1_ic_test.py
各業種の候補株から「USシグナルを最も反映する銘柄」を選定する

INPUT:
  - data/processed/signals_etf_full.csv（Step 0 で生成、2010〜現在）
  - yfinance で各候補株の OHLC を取得

OUTPUT:
  - output/reports/phase1_ic_ranking.csv
    カラム: sector, ticker, IC_full, IC_2010_2015, IC_2016_2020, IC_2021_now,
            n_days, positive_periods, win_rate, decision

PREREQUISITE:
  step0_calc_full_signals.py を先に実行して signals_etf_full.csv を生成すること。
  signals.csv（2021〜のみ）はサンプル数が少なすぎるため使用禁止。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402

# ================================================================
# 候補株リスト（各業種 3〜5 銘柄）
# ================================================================

CANDIDATES: dict[str, list[int]] = {
    # ── 採用B 確定済み（変更不要） ──────────────────────────
    "energy":       [5020, 5019, 1605],
    "construction": [5233, 1801, 1802, 1803],
    "steel":        [5713, 5802, 5401],
    "electronics":  [6971, 6861, 6758, 6702],
    "it_services":  [9984, 9432, 9433, 4307],
    "utilities":    [9501, 9502, 9503, 9531],
    "trading":      [8053, 8001, 8058, 8031],
    "finance":      [8725, 8750, 8766, 8604],

    # ── 採用C → 候補大量追加 ────────────────────────────────

    # 食品: 大手食品・飲料・たばこ・調味料
    "food": [
        2802, 2503, 2502, 2914,   # 既存
        2282,  # 日本ハム
        2269,  # 明治HD
        2871,  # ニチレイ
        2897,  # 日清食品HD
        2810,  # ハウス食品G
        2201,  # 森永製菓
        2670,  # エービーシー・マート（参考）
        2531,  # 宝HD（酒造）
        2587,  # サントリー食品
    ],

    # 素材化学: 化学・ガラス・紙パ・ゴム
    "materials": [
        4063, 4188, 3407, 4452,   # 既存
        4911,  # 資生堂
        4183,  # 三井化学
        4631,  # DIC
        5332,  # TOTO
        5201,  # AGC（旭硝子）
        3861,  # 王子HD
        5101,  # 横浜ゴム
        4004,  # レゾナック（昭和電工）
        4208,  # UBE（宇部興産）
        4021,  # 日産化学
        4151,  # 協和キリン
    ],

    # 医薬品: 大手製薬・バイオ
    "pharma": [
        4502, 4519, 4568, 4523,   # 既存
        4507,  # 塩野義製薬
        4530,  # 久光製薬
        4536,  # 参天製薬
        4578,  # 大塚HD
        4543,  # テルモ
        4324,  # 電通グループ（参考除外用）
        4506,  # 住友ファーマ
        4516,  # 日本新薬
        2124,  # JAC Recruitment（除外用）
        4571,  # ナノキャリア
        4587,  # ペプチドリーム
    ],

    # 自動車: 自動車・二輪・部品
    "auto": [
        7203, 7267, 7269, 7201,   # 既存
        7270,  # SUBARU
        7261,  # マツダ
        7205,  # 日野自動車
        7211,  # 三菱自動車
        7259,  # アイシン
        7248,  # カルソニックカンセイ→フォルビア
        7240,  # NOK
        7282,  # 豊田合成
        5108,  # ブリヂストン
        7272,  # ヤマハ発動機
        6902,  # デンソー
        7276,  # 小糸製作所
    ],

    # 機械: 産業機械・重工・精密
    "machinery": [
        6367, 6302, 6273, 7011,   # 既存
        6301,  # コマツ
        6326,  # クボタ
        6201,  # 豊田自動織機
        6770,  # アルプスアルパイン
        6472,  # NTN
        6471,  # 日本精工（NSK）
        6506,  # 安川電機
        6645,  # オムロン
        6103,  # オークマ
        6113,  # アマダ
        7013,  # IHI
        6361,  # 荏原製作所
    ],

    # 運輸物流: 海運・陸運・空運・倉庫
    "transport": [
        9020, 9101, 9104, 9107,   # 既存
        9022,  # JR東海
        9021,  # JR西日本
        9001,  # 東武鉄道
        9005,  # 東急
        9064,  # ヤマトHD
        9062,  # 日本通運
        9006,  # 京急電鉄
        9308,  # 乾汽船
        9202,  # ANA HD
        9201,  # JAL
        9070,  # トナミHD
    ],

    # 小売: 専門店・百貨店・GMS・EC
    "retail": [
        9983, 3382, 8267, 3092,   # 既存
        3099,  # 三越伊勢丹HD
        8270,  # ユニーグループ→廃止→代替
        2651,  # ローソン
        2502,  # アサヒ（参考）
        3086,  # Jフロント（大丸）
        8233,  # 高島屋
        3197,  # すかいらーく
        7453,  # 良品計画
        2780,  # コメ兵HD
        3048,  # ビックカメラ
        9843,  # ニトリHD
        2753,  # あみやき亭
        3088,  # マツキヨコクミン
        4755,  # 楽天グループ
    ],

    # 銀行: 都銀・地銀・信託
    "banks": [
        8306, 8316, 8411,         # 既存
        8309,  # 三井住友トラスト
        8308,  # りそなHD
        8354,  # ふくおかFG
        8355,  # 静岡銀行
        8359,  # 八十二銀行
        8377,  # ほくほくFG
        8398,  # 千葉銀行
        8331,  # 千葉銀行（重複確認用）
        8253,  # クレディセゾン
        8601,  # 大和証券G
        8697,  # JPX
    ],

    # 不動産: デベロッパー・REIT・管理
    "realestate": [
        8801, 8802, 8830,         # 既存
        3003,  # ヒューリック
        8815,  # 東急不動産HD
        3231,  # 野村不動産HD
        8984,  # 大和ハウスリート（J-REIT除外→個別株）
        1925,  # 大和ハウス工業
        1928,  # 積水ハウス
        3289,  # 東急不動産（重複確認）
        8905,  # イオンモール
        3254,  # プレサンスコーポレーション
        8876,  # リログループ
        3278,  # ケネディクス（上場廃止可能性）
    ],
}

SECTOR_NAMES: dict[str, str] = {
    "food": "食品", "energy": "エネルギー", "construction": "建設",
    "materials": "素材化学", "pharma": "医薬品", "auto": "自動車",
    "steel": "鉄鋼非鉄", "machinery": "機械", "electronics": "電機精密",
    "it_services": "情報通信", "utilities": "電力ガス", "transport": "運輸物流",
    "trading": "商社卸売", "retail": "小売", "banks": "銀行",
    "finance": "金融", "realestate": "不動産",
}

# 期間分割（安定性チェック用）
PERIODS = {
    "IC_2010_2015": ("2010-01-01", "2015-12-31"),
    "IC_2016_2020": ("2016-01-01", "2020-12-31"),
    "IC_2021_now":  ("2021-01-01", "2099-12-31"),
}

# 採用判定閾値
THRESH_A = 0.05   # 採用A: IC_full >= 0.05
THRESH_B = 0.02   # 採用B: 0.02 <= IC_full < 0.05
MIN_DAYS = 750    # 最低サンプル数


# ================================================================
# ユーティリティ
# ================================================================

def calc_ic(signal: pd.Series, ret_next_day: pd.Series) -> tuple[float, int]:
    """signal[t] と ret[t]（翌日OC）の相関を計算。ret は既に1日シフト済みで渡すこと。"""
    common = signal.index.intersection(ret_next_day.index)
    if len(common) < 20:
        return np.nan, 0
    s = signal.loc[common].values
    r = ret_next_day.loc[common].values
    mask = ~(np.isnan(s) | np.isnan(r))
    if mask.sum() < 20:
        return np.nan, 0
    ic = np.corrcoef(s[mask], r[mask])[0, 1]
    return float(ic), int(mask.sum())


def decide(ic_full: float, n_days: int, positive_periods: int) -> str:
    """分岐判定ルール（leadlag_CLAUDE.md に定義）"""
    if np.isnan(ic_full):
        return "C"
    if ic_full <= -0.03:
        return "D"
    if n_days < MIN_DAYS:
        return "C"
    if positive_periods < 2:
        return "C"
    if ic_full >= THRESH_A:
        return "A"
    if ic_full >= THRESH_B:
        return "B"
    return "C"


# ================================================================
# メイン
# ================================================================

def main() -> None:
    print("=" * 70)
    print("Phase 1: IC テスト（フル期間版）")
    print("=" * 70)

    # signals_etf_full.csv を読み込む
    sig_path = PROCESSED_DIR / "signals_etf_full.csv"
    if not sig_path.exists():
        print(f"\n[ERROR] {sig_path} が見つかりません。")
        print("先に step0_calc_full_signals.py を実行してください。")
        return

    signals = pd.read_csv(sig_path, parse_dates=["date"]).set_index("date")
    print(f"\nシグナル期間: {signals.index[0].date()} ~ {signals.index[-1].date()} ({len(signals)}日)")

    # 候補株の OC リターンを一括取得
    all_tickers = sorted({f"{t}.T" for tlist in CANDIDATES.values() for t in tlist})
    print(f"\n[1] 候補株 {len(all_tickers)} 銘柄のデータ取得中...")
    raw = yf.download(all_tickers, start="2010-01-01", auto_adjust=True, progress=False)

    # OC リターン（1 日シフトして翌日 OC にする）
    oc_all: dict[str, pd.Series] = {}
    for ticker_str in all_tickers:
        try:
            if len(all_tickers) == 1:
                op = raw["Open"]
                cl = raw["Close"]
            else:
                op = raw["Open"][ticker_str]
                cl = raw["Close"][ticker_str]
            oc = ((cl - op) / op).shift(-1)   # shift(-1): signal[t] と ret[t+1] を合わせる
            oc.index = pd.to_datetime(oc.index)
            oc_all[ticker_str] = oc
        except Exception as e:
            print(f"  警告: {ticker_str} の処理失敗 ({e})")

    print(f"  {len(oc_all)} 銘柄取得完了")

    # 業種ごとに IC 計算 → 判定
    print("\n[2] IC 計算・判定...")
    all_results = []
    sector_decisions = {}

    for sector_key, ticker_list in CANDIDATES.items():
        sector_name = SECTOR_NAMES.get(sector_key, sector_key)
        signal_col  = f"signal_{sector_key}"

        if signal_col not in signals.columns:
            print(f"\n  [{sector_name}] シグナル列なし → スキップ")
            continue

        sig_series = signals[signal_col]
        print(f"\n  【{sector_name}】")

        rankings = []
        for ticker in ticker_list:
            ticker_str = f"{ticker}.T"
            oc = oc_all.get(ticker_str)
            if oc is None:
                continue

            # 全期間 IC
            ic_full, n = calc_ic(sig_series, oc)

            # 期間別 IC
            period_ics = {}
            for pname, (pstart, pend) in PERIODS.items():
                sig_p = sig_series[(sig_series.index >= pstart) & (sig_series.index <= pend)]
                oc_p  = oc[(oc.index >= pstart) & (oc.index <= pend)]
                ic_p, _ = calc_ic(sig_p, oc_p)
                period_ics[pname] = ic_p

            # プラス期間数
            positive_periods = sum(
                1 for v in period_ics.values()
                if not np.isnan(v) and v > 0
            )

            # 勝率
            common = sig_series.index.intersection(oc.index)
            if len(common) > 0:
                oc_vals = oc.loc[common].dropna()
                win_rate = float((oc_vals > 0).mean())
            else:
                win_rate = np.nan

            dec = decide(ic_full, n, positive_periods)

            rankings.append({
                "sector":           sector_key,
                "sector_name":      sector_name,
                "ticker":           ticker,
                "IC_full":          ic_full,
                **period_ics,
                "n_days":           n,
                "positive_periods": positive_periods,
                "win_rate":         win_rate,
                "decision":         dec,
            })

        if not rankings:
            print(f"    データなし")
            continue

        # IC_full でソート
        rankings.sort(key=lambda x: (x["IC_full"] if not np.isnan(x["IC_full"]) else -99), reverse=True)

        # 表示
        print(f"    {'Rank':<4} {'Ticker':<8} {'IC_full':>8} "
              f"{'2010-15':>8} {'2016-20':>8} {'2021-':>8} "
              f"{'n':>6} {'pos':>4} {'decision'}")
        print(f"    {'-'*72}")
        for i, r in enumerate(rankings, 1):
            print(
                f"    {i:<4} {r['ticker']:<8} {r['IC_full']:>+8.4f} "
                f"{r['IC_2010_2015']:>+8.4f} {r['IC_2016_2020']:>+8.4f} {r['IC_2021_now']:>+8.4f} "
                f"{r['n_days']:>6} {r['positive_periods']:>4}    {r['decision']}"
            )

        best = rankings[0]
        print(f"    → 採用: {best['ticker']}.T  判定={best['decision']}")
        sector_decisions[sector_key] = best["decision"]
        all_results.extend(rankings)

    # ================================================================
    # ポートフォリオ全体の判定
    # ================================================================
    print("\n" + "=" * 70)
    print("ポートフォリオ全体の判定")
    print("=" * 70)

    count_a = sum(1 for d in sector_decisions.values() if d == "A")
    count_b = sum(1 for d in sector_decisions.values() if d == "B")
    count_c = sum(1 for d in sector_decisions.values() if d == "C")
    count_d = sum(1 for d in sector_decisions.values() if d == "D")

    print(f"\n  採用A: {count_a}業種")
    print(f"  採用B: {count_b}業種（条件付き）")
    print(f"  再探索C: {count_c}業種")
    print(f"  逆張りD: {count_d}業種")

    if count_a >= 14:
        portfolio_decision = "P1"
        print("\n  判定: P1 → Phase 2 へ進む（単一構成）")
    elif count_a + count_b >= 14:
        portfolio_decision = "P2"
        print("\n  判定: P2 → Phase 2 へ進む（B業種は複数案で比較）")
        c_sectors = [s for s, d in sector_decisions.items() if d == "C"]
        if c_sectors:
            print(f"  [*] 再探索C の業種: {', '.join(c_sectors)}")
            print(f"      → CANDIDATES に候補を追加して Phase 1 を再実行すること")
    else:
        portfolio_decision = "P3"
        print("\n  判定: P3 → 候補ユニバース再設計が必要。Phase 2 には進まない。")
        c_sectors = [s for s, d in sector_decisions.items() if d in ("C", "D")]
        print(f"  要対応業種: {', '.join(c_sectors)}")

    # CSV 保存
    result_df = pd.DataFrame(all_results)
    out_path = REPORTS_DIR / "phase1_ic_ranking.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\n[完了] 結果保存: {out_path}")
    print(f"  portfolio_decision = {portfolio_decision}")

    if portfolio_decision in ("P1", "P2"):
        print("\n次のステップ: python scripts/phase2_full_backtest.py")
    else:
        print("\n次のステップ: CANDIDATES を拡張して phase1_ic_test.py を再実行")


if __name__ == "__main__":
    main()
