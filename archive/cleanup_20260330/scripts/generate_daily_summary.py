import pandas as pd
from pathlib import Path

# Paths
base = Path(__file__).parent.parent
overall_file = base / 'output' / 'overall_cumulative_percent.txt'
test_file = base / 'output' / 'results' / 'test_mar20_to_today.csv'
out_file = base / 'output' / 'daily_summary.txt'

# Load overall cumulative percent (tab-separated)
rows = []
with open(overall_file, encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()]
for i,l in enumerate(lines):
    if i==0:
        continue
    date, val = l.split('\t')
    rows.append({'date':pd.to_datetime(date), 'cum_pct':float(val)})

df1 = pd.DataFrame(rows).set_index('date')

# Load test_mar20_to_today cum_curve if exists
if test_file.exists():
    tdf = pd.read_csv(test_file, parse_dates=['date'])
    # cum_curve exists, convert to percent
    if 'cum_curve' in tdf.columns:
        tdf = tdf[['date','cum_curve']].set_index('date')
        tdf['cum_pct'] = (tdf['cum_curve'] - 1.0) * 100.0
        df2 = tdf[['cum_pct']]
    else:
        df2 = pd.DataFrame(columns=['cum_pct'])
else:
    df2 = pd.DataFrame(columns=['cum_pct'])

# Combine (union) and sort
combined = pd.concat([df1, df2]).sort_index()
combined = combined[~combined.index.duplicated(keep='first')]

# Compute daily change and win/loss (skip first row)
combined['daily_change'] = combined['cum_pct'].diff()

def label(x):
    if pd.isna(x):
        return ''
    if x > 0:
        return '勝ち'
    if x < 0:
        return '負け'
    return '横ばい'

combined['result'] = combined['daily_change'].apply(label)

# Format to 2 decimals
out_lines = ['日付\t累積リターン(%)\t当日増減(%)\t勝敗']
for idx, row in combined.iterrows():
    d = idx.strftime('%Y-%m-%d')
    cum = '' if pd.isna(row['cum_pct']) else f"{row['cum_pct']:.2f}"
    change = '' if pd.isna(row['daily_change']) else f"{row['daily_change']:.2f}"
    res = row['result']
    out_lines.append(f"{d}\t{cum}\t{change}\t{res}")

# Save and print
with open(out_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines))
print('\n'.join(out_lines))
