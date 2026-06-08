from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent

# セクター名→ティッカー対応（必要なら拡張）
SECTOR_TICKER = {
    'food':'1617.T','energy':'1618.T','construction':'1619.T','materials':'1620.T',
    'pharma':'1621.T','auto':'1622.T','steel':'1623.T','machinery':'1624.T',
    'electronics':'1625.T','it_services':'1626.T','utilities':'1627.T','transport':'1628.T',
    'trading':'1629.T','retail':'1630.T','banks':'1631.T','finance':'1632.T','realestate':'1633.T'
}


def to_ticker(x):
    if pd.isna(x) or str(x).strip() == '':
        return None
    s = str(x).strip()
    if s.endswith('.T'):
        return s
    return SECTOR_TICKER.get(s, None)


def main():
    in_path = ROOT / 'output' / 'daily_positions_with_sizes.csv'
    out_path = ROOT / 'output' / 'today_pnl.csv'
    if not in_path.exists():
        print('daily_positions_with_sizes.csv not found')
        return

    df = pd.read_csv(in_path, parse_dates=['date'])
    # target today's date (workspace date)
    today = pd.Timestamp('2026-03-27').date()
    row = df[df['date'] == pd.Timestamp(today).isoformat()]
    if len(row) == 0:
        print('No row for today in daily_positions_with_sizes.csv')
        return

    row = row.iloc[0]

    longs = [row.get('long_1'), row.get('long_2'), row.get('long_3')]
    shorts= [row.get('short_1'), row.get('short_2'), row.get('short_3')]
    long_each = float(row['long_each']) if not pd.isna(row['long_each']) else 0.0
    short_each= float(row['short_each']) if not pd.isna(row['short_each']) else 0.0
    fee_bps = float(row.get('fee_bps_per_side', 5.0))

    tickers = {}
    for i,s in enumerate(longs, start=1):
        t = to_ticker(s)
        if t:
            tickers[f'long{i}'] = t
    for i,s in enumerate(shorts, start=1):
        t = to_ticker(s)
        if t:
            tickers[f'short{i}'] = t

    prices = {}
    for role,t in tickers.items():
        try:
            info = yf.download(t, period='2d', interval='1d', progress=False)
            prev_close = float(info['Close'].iloc[-2])
            last_price = float(info['Close'].iloc[-1])
            prices[role] = (t, prev_close, last_price)
        except Exception as e:
            prices[role] = (t, np.nan, np.nan)

    long_pnl = 0.0
    short_pnl = 0.0
    for k,v in prices.items():
        t, p0, p1 = v
        if np.isnan(p0) or np.isnan(p1):
            continue
        if k.startswith('long'):
            pnl = long_each * (p1 / p0 - 1.0)
            long_pnl += pnl
        else:
            pnl = short_each * (1.0 - p1 / p0)
            short_pnl += pnl

    gross_notional = (long_each + short_each) * 2.0
    fee_rate = fee_bps / 10000.0
    fee_entry = gross_notional * fee_rate
    fee_roundtrip = fee_entry * 2.0
    net_pnl_after_entry_fees = long_pnl + short_pnl - fee_entry

    out = {
        'date': str(today),
        'long_pnl': long_pnl,
        'short_pnl': short_pnl,
        'fee_entry': fee_entry,
        'fee_roundtrip': fee_roundtrip,
        'net_pnl_after_entry_fees': net_pnl_after_entry_fees,
    }

    print(out)
    pd.DataFrame([out]).to_csv(out_path, index=False)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
