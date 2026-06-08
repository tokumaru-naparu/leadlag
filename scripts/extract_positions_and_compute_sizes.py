from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent

PARAMS = {
    'sig_coef':   2.0,
    'size_cap':   3.0,
    'size_min':   0.1,
    'target_vol': 0.015,
    'dd_t1':     -0.08,
    'dd_s1':      0.7,
    'dd_t2':     -0.18,
    'dd_s2':      0.2,
}


def clamp_size(val):
    return float(np.clip(val, PARAMS['size_min'], PARAMS['size_cap']))


def calc_final_size(signal_strength, recent_vol, current_dd):
    if recent_vol > 0:
        vol_adj = PARAMS['target_vol'] / recent_vol
    else:
        vol_adj = 1.0

    base_size = clamp_size(signal_strength * PARAMS['sig_coef'])
    size = clamp_size(base_size * vol_adj)

    if current_dd <= PARAMS['dd_t2']:
        dd_scale = PARAMS['dd_s2']
    elif current_dd <= PARAMS['dd_t1']:
        dd_scale = PARAMS['dd_s1']
    else:
        dd_scale = 1.0

    final_size = size * dd_scale
    return final_size, dd_scale, vol_adj


def main():
    # Paths
    results_path = ROOT / 'output' / 'results' / 'results.csv'
    test_path = ROOT / 'output' / 'results' / 'test_mar20_to_today.csv'
    log_path = ROOT / 'output' / 'results' / 'paper_trade_log.csv'
    trades_path = ROOT / 'data' / 'processed' / 'trades.csv'
    out_path = ROOT / 'output' / 'daily_positions_with_sizes.csv'

    # Read files
    results = pd.read_csv(results_path, encoding='utf-8-sig', parse_dates=['日付'])
    results.set_index('日付', inplace=True)

    test_df = None
    if test_path.exists():
        test_df = pd.read_csv(test_path, parse_dates=['date'])
        test_df.set_index('date', inplace=True)

    if log_path.exists():
        log_df = pd.read_csv(log_path, parse_dates=['date'])
        log_df.set_index('date', inplace=True)
    else:
        log_df = pd.DataFrame()

    trades = pd.read_csv(trades_path, parse_dates=['date'])
    trades.set_index('date', inplace=True)
    base_r = trades['long_return'] - trades['short_return']

    # Collect all dates from results, test, and log
    dates = set(results.index.date)
    if test_df is not None:
        dates |= set(test_df.index.date)
    if not log_df.empty:
        dates |= set(log_df.index.date)

    rows = []
    for d in sorted(dates):
        date = pd.Timestamp(d)

        # Extract longs/shorts
        row = {'date': date.date().isoformat()}

        if date in results.index:
            r = results.loc[date]
            row['long_1'] = r.get('ロング1', '')
            row['long_2'] = r.get('ロング2', '')
            row['long_3'] = r.get('ロング3', '')
            row['short_1'] = r.get('ショート1', '')
            row['short_2'] = r.get('ショート2', '')
            row['short_3'] = r.get('ショート3', '')
            sig_strength = r.get('シグナル強度', np.nan)
        else:
            # try test file or log
            sig_strength = np.nan
            if test_df is not None and date in test_df.index:
                t = test_df.loc[date]
                row['long_1'] = t.get('long_1', '')
                row['long_2'] = t.get('long_2', '')
                row['long_3'] = t.get('long_3', '')
                row['short_1'] = t.get('short_1', '')
                row['short_2'] = t.get('short_2', '')
                row['short_3'] = t.get('short_3', '')
        
        # signal_strength fallback from log
        if (pd.isna(sig_strength) or sig_strength == '') and (not log_df.empty) and (date in log_df.index):
            sig_strength = log_df.loc[date].get('signal_strength', np.nan)

        # compute recent vol up to date
        recent_vol = np.nan
        try:
            base_up_to = base_r.loc[:date]
            if len(base_up_to) >= 2:
                recent_vol = base_up_to.rolling(20).std().iloc[-1]
                if np.isnan(recent_vol):
                    recent_vol = base_up_to.std()
        except Exception:
            recent_vol = np.nan

        # compute current_dd using last log entry prior to or on date
        current_dd = 0.0
        if not log_df.empty:
            prior = log_df.loc[:date]
            if len(prior) > 0:
                last = prior.iloc[-1]
                capital = last.get('capital', np.nan)
                peak = last.get('peak', np.nan)
                if pd.notna(capital) and pd.notna(peak) and peak != 0:
                    current_dd = (capital - peak) / peak

        # final_size: prefer logged value
        final_size = np.nan
        dd_scale = np.nan
        vol_adj = np.nan
        if (not log_df.empty) and (date in log_df.index) and pd.notna(log_df.loc[date].get('final_size', np.nan)):
            final_size = float(log_df.loc[date].get('final_size'))
            dd_scale = float(log_df.loc[date].get('dd_scale', np.nan)) if 'dd_scale' in log_df.columns else np.nan
        else:
            if pd.isna(sig_strength):
                final_size = np.nan
            else:
                final_size, dd_scale, vol_adj = calc_final_size(float(sig_strength), float(recent_vol) if pd.notna(recent_vol) else 0.0, float(current_dd))

        # amounts using example capital 1_000_000
        capital_example = 1_000_000
        if pd.notna(final_size):
            position_value = capital_example * final_size
            long_each = position_value / 3.0
            short_each = position_value / 3.0
        else:
            position_value = np.nan
            long_each = np.nan
            short_each = np.nan

        row.update({
            'final_size': final_size,
            'dd_scale': dd_scale,
            'recent_vol': recent_vol,
            'current_dd': current_dd,
            'position_value': position_value,
            'long_each': long_each,
            'short_each': short_each,
        })

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.sort_values('date', inplace=True)
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Wrote {out_path} ({len(out_df)} rows)")


if __name__ == '__main__':
    main()
