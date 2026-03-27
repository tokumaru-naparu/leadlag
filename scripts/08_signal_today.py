# 08_signal_today.py
# 毎日のシグナル計算スクリプト（ペーパートレード・実運用共通）
# 米国市場引け後（18:00以降）に実行する
# タスクスケジューラで自動実行推奨

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    PROCESSED_DIR, LEGACY_SCRIPTS_HISTORY_DIR, LEGACY_HISTORY_DIR,
    RESULTS_DIR, pick_largest_csv
)

# ================================================================
# 確定パラメータ
# ================================================================
PARAMS = {
    'sig_coef':   2.0,
    'size_cap':   3.0,    # 信用取引上限
    'size_min':   0.1,
    'short_amp':  2.0,
    'target_vol': 0.015,
    'dd_t1':     -0.08,   # 8%下落で発動（-6%から変更）
    'dd_s1':      0.7,
    'dd_t2':     -0.18,
    'dd_s2':      0.2,
}

# 片道手数料（bps）。例: 5.0 = 0.05%
FEE_BPS_PER_SIDE = float(os.getenv('TRADE_FEE_BPS_PER_SIDE', '5.0'))

# 業種名マッピング
SECTOR_NAMES = {
    'food':       '食品       (1617.T)',
    'energy':     'エネルギー  (1618.T)',
    'construction':'建設      (1619.T)',
    'materials':  '素材化学   (1620.T)',
    'pharma':     '医薬品     (1621.T)',
    'auto':       '自動車     (1622.T)',
    'steel':      '鉄鋼非鉄   (1623.T)',
    'machinery':  '機械       (1624.T)',
    'electronics':'電機精密   (1625.T)',
    'it_services':'情報通信   (1626.T)',
    'utilities':  '電力ガス   (1627.T)',
    'transport':  '運輸物流   (1628.T)',
    'trading':    '商社卸売   (1629.T)',
    'retail':     '小売       (1630.T)',
    'banks':      '銀行       (1631.T)',
    'finance':    '金融       (1632.T)',
    'realestate': '不動産     (1633.T)',
}

# ================================================================
# データ読み込み
# ================================================================
def load_csv(filename):
    candidates = [
        PROCESSED_DIR / filename,
        LEGACY_SCRIPTS_HISTORY_DIR / filename,
        LEGACY_HISTORY_DIR / filename,
    ]
    path = pick_largest_csv(candidates)
    if path is None:
        raise FileNotFoundError(f"{filename} が見つかりません")
    df = pd.read_csv(path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

# ================================================================
# DD水準の計算（過去の運用記録から）
# ================================================================
def calc_current_dd():
    """
    過去の日次リターン記録から現在のDD水準を計算する。
    記録ファイルがない場合は0（ピーク状態）を返す。
    """
    log_path = RESULTS_DIR / 'paper_trade_log.csv'

    if not log_path.exists():
        return 0.0, 1.0, 1.0  # dd, capital, peak

    log_df = pd.read_csv(log_path)
    if len(log_df) == 0:
        return 0.0, 1.0, 1.0

    capital = log_df['capital'].iloc[-1]
    peak    = log_df['peak'].iloc[-1]
    dd      = (capital - peak) / peak
    return dd, capital, peak

# ================================================================
# シグナル計算
# ================================================================
def calc_signal(signals_df):
    """最新日のシグナルを取得して業種をランキング"""
    today_sig = signals_df.iloc[-1]
    today_date = signals_df.index[-1]

    # シグナル列を取得
    signal_cols = [c for c in signals_df.columns
                   if c.startswith('signal_')
                   and c not in ('signal_strength', 'signal_spread')]

    # 各業種のシグナル値
    sector_signals = {}
    for col in signal_cols:
        sector = col.replace('signal_', '')
        sector_signals[sector] = today_sig[col]

    # ランキング（降順）
    ranked = sorted(sector_signals.items(), key=lambda x: x[1], reverse=True)

    # ロング3業種・ショート3業種
    long_sectors  = ranked[:3]
    short_sectors = ranked[-3:][::-1]  # 最も低いものから

    signal_strength = today_sig.get('signal_strength', 0)

    return today_date, signal_strength, long_sectors, short_sectors, sector_signals

# ================================================================
# ポジションサイズ計算
# ================================================================
def calc_position_size(signal_strength, recent_vol, current_dd):
    """
    方式E: シグナル比例 × ボラターゲット × DD動的縮小
    """
    # ボラ調整
    if recent_vol > 0:
        vol_adj = PARAMS['target_vol'] / recent_vol
    else:
        vol_adj = 1.0

    # ベースサイズ
    base_size = np.clip(
        signal_strength * PARAMS['sig_coef'],
        PARAMS['size_min'],
        PARAMS['size_cap']
    )
    size = np.clip(base_size * vol_adj, PARAMS['size_min'], PARAMS['size_cap'])

    # DD動的縮小
    if current_dd <= PARAMS['dd_t2']:
        dd_scale = PARAMS['dd_s2']
        dd_regime = f"⚠️  警戒2（{PARAMS['dd_t2']*100:.0f}%以下）"
    elif current_dd <= PARAMS['dd_t1']:
        dd_scale = PARAMS['dd_s1']
        dd_regime = f"⚡ 警戒1（{PARAMS['dd_t1']*100:.0f}%以下）"
    else:
        dd_scale = 1.0
        dd_regime = "✅ 通常"

    final_size = size * dd_scale

    return final_size, dd_scale, dd_regime, vol_adj

# ================================================================
# メイン処理
# ================================================================
def main():
    today = datetime.date.today()

    print("=" * 60)
    print(f"📊 PCA SUB リードラグ戦略 シグナル計算")
    print(f"   実行日: {today.strftime('%Y年%m月%d日 (%a)')}")
    print("=" * 60)

    # 土日チェック
    if today.weekday() >= 5:
        print(f"\n⏭️  本日は{['月','火','水','木','金','土','日'][today.weekday()]}曜日のためスキップします")
        return

    # 祝日チェック
    try:
        import jpholiday
        if jpholiday.is_holiday(today):
            print(f"\n⏭️  本日は祝日のためスキップします")
            return
    except ImportError:
        print("  ※ jpholidayが未インストールのため祝日チェックをスキップ")

    # データ読み込み
    print("\n📂 データ読み込み中...")
    try:
        signals = load_csv('signals.csv')
        trades  = load_csv('trades.csv')
        print(f"  ✅ signals.csv: {len(signals)}行")
        print(f"  ✅ trades.csv:  {len(trades)}行")
    except Exception as e:
        print(f"  ❌ エラー: {e}")
        return

    # 最新データの確認
    latest_date = signals.index[-1]
    print(f"\n  最新データ日: {latest_date.date()}")
    if (today - latest_date.date()).days > 5:
        print(f"  ⚠️  データが古い可能性があります（{(today - latest_date.date()).days}日前）")

    # ボラティリティ計算（直近20日）
    base_r = trades['long_return'] - trades['short_return']
    recent_vol = base_r.rolling(20).std().iloc[-1]

    # シグナル計算
    sig_date, signal_strength, long_sectors, short_sectors, all_signals = calc_signal(signals)

    # 現在のDD水準
    current_dd, capital, peak = calc_current_dd()

    # ポジションサイズ計算
    final_size, dd_scale, dd_regime, vol_adj = calc_position_size(
        signal_strength, recent_vol, current_dd
    )

    # シグナル強度の分位（参考）
    all_strengths = signals['signal_strength'].dropna()
    percentile = (all_strengths < signal_strength).mean() * 100

    # ================================================================
    # 出力
    # ================================================================
    print("\n" + "=" * 60)
    print("📈 本日のシグナル")
    print("=" * 60)

    print(f"\n  シグナル強度: {signal_strength:.4f}")
    print(f"  強度パーセンタイル: {percentile:.0f}%ile（過去{len(all_strengths)}日中）")

    # 強度の評価
    if percentile >= 80:
        strength_comment = "🔥 非常に強い（Q5）"
    elif percentile >= 60:
        strength_comment = "💪 強い（Q4）"
    elif percentile >= 40:
        strength_comment = "📊 普通（Q3）"
    elif percentile >= 20:
        strength_comment = "📉 弱い（Q2）"
    else:
        strength_comment = "😴 非常に弱い（Q1）"
    print(f"  評価: {strength_comment}")

    print(f"\n{'─'*60}")
    print(f"  🟢 ロング（買い）3業種:")
    for i, (sector, sig_val) in enumerate(long_sectors, 1):
        name = SECTOR_NAMES.get(sector, sector)
        print(f"    {i}. {name}  （シグナル: {sig_val:+.4f}）")

    print(f"\n  🔴 ショート（売り）3業種:")
    for i, (sector, sig_val) in enumerate(short_sectors, 1):
        name = SECTOR_NAMES.get(sector, sector)
        print(f"    {i}. {name}  （シグナル: {sig_val:+.4f}）")

    print(f"\n{'─'*60}")
    print(f"  📊 ポジションサイズ計算:")
    print(f"    直近20日ボラ:   {recent_vol:.4f}（目標: {PARAMS['target_vol']}）")
    print(f"    ボラ調整:       {vol_adj:.2f}倍")
    print(f"    ベースサイズ:   {np.clip(signal_strength * PARAMS['sig_coef'], PARAMS['size_min'], PARAMS['size_cap']):.2f}倍")
    print(f"    ボラ調整後:     {np.clip(np.clip(signal_strength * PARAMS['sig_coef'], PARAMS['size_min'], PARAMS['size_cap']) * vol_adj, PARAMS['size_min'], PARAMS['size_cap']):.2f}倍")
    print(f"\n    DD状況:         {dd_regime}")
    print(f"    現在DD:         {current_dd*100:.1f}%  （ピークから）")
    print(f"    DD縮小スケール: {dd_scale:.1f}倍")
    print(f"\n  ✅ 最終ポジションサイズ: {final_size:.2f}倍")

    # 実際の取引金額例
    capital_example = 1_000_000  # 100万円
    position_value  = capital_example * final_size
    long_each       = position_value / 3
    short_each      = position_value / 3

    print(f"\n{'─'*60}")
    print(f"  💰 取引金額（100万円の場合）:")
    print(f"    総ポジション:  {position_value:,.0f}円")
    print(f"    ロング各業種:  {long_each:,.0f}円")
    print(f"    ショート各業種:{short_each:,.0f}円")

    fee_rate_side = FEE_BPS_PER_SIDE / 10000.0
    gross_notional = position_value * 2.0  # long + short の建玉合計
    est_fee_entry = gross_notional * fee_rate_side
    est_fee_roundtrip = gross_notional * fee_rate_side * 2.0

    print(f"\n{'─'*60}")
    print(f"  🧾 手数料見積（片道 {FEE_BPS_PER_SIDE:.2f} bps）:")
    print(f"    建玉合計(ロング+ショート): {gross_notional:,.0f}円")
    print(f"    新規建て分の概算手数料:   {est_fee_entry:,.0f}円")
    print(f"    往復分の概算手数料:       {est_fee_roundtrip:,.0f}円")

    # ================================================================
    # 17業種全シグナル（参考）
    # ================================================================
    print(f"\n{'─'*60}")
    print(f"  📋 全17業種シグナル一覧（参考）:")
    sorted_all = sorted(all_signals.items(), key=lambda x: x[1], reverse=True)
    for rank, (sector, val) in enumerate(sorted_all, 1):
        name = SECTOR_NAMES.get(sector, sector)
        bar  = "🟢" if rank <= 3 else "🔴" if rank >= 15 else "  "
        print(f"    {rank:2d}. {bar} {name:<18} {val:+.4f}")

    # ================================================================
    # ログ保存
    # ================================================================
    log_path = RESULTS_DIR / 'paper_trade_log.csv'
    log_row = {
        'date':             str(today),
        'signal_date':      str(sig_date.date()),
        'signal_strength':  signal_strength,
        'signal_pct':       percentile,
        'long_1':           long_sectors[0][0],
        'long_2':           long_sectors[1][0],
        'long_3':           long_sectors[2][0],
        'short_1':          short_sectors[0][0],
        'short_2':          short_sectors[1][0],
        'short_3':          short_sectors[2][0],
        'final_size':       final_size,
        'dd_scale':         dd_scale,
        'current_dd':       current_dd,
        'capital':          capital,
        'peak':             peak,
        'actual_return':    None,  # 翌日終値確認後に手動記入
        'fee_bps_per_side': FEE_BPS_PER_SIDE,
        'gross_notional':   gross_notional,
        'fee_entry_est':    est_fee_entry,
        'fee_roundtrip_est': est_fee_roundtrip,
        'note':             '',
    }

    if log_path.exists():
        log_df = pd.read_csv(log_path)
        # 同じ日が既にあれば上書き
        log_df = log_df[log_df['date'] != str(today)]
        log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
    else:
        log_df = pd.DataFrame([log_row])

    log_df.to_csv(log_path, index=False, encoding='utf-8-sig')

    print(f"\n{'=' * 60}")
    print(f"✅ ログ保存: {log_path}")
    print(f"\n⏰ 明日の寄り付きで上記の取引を実行してください")
    print(f"   （ペーパートレードの場合は記録のみ）")
    print("=" * 60)

if __name__ == '__main__':
    main()