import pandas as pd

in_file = 'c:/Users/hg317/Desktop/projects/leadlag/output/cumulative_returns_from_march_1.csv'
out_file = 'c:/Users/hg317/Desktop/projects/leadlag/output/overall_cumulative_percent.txt'

# Load
df = pd.read_csv(in_file, parse_dates=['date'])
# compute row-wise mean across numeric columns
num = df.select_dtypes(include=['number'])
mean_series = num.mean(axis=1) * 100  # percent

# format
out_lines = ['date\t累積リターン (%)']
for d, v in zip(df['date'].dt.strftime('%Y-%m-%d'), mean_series):
    out_lines.append(f"{d}\t{v:.2f}")

# write and print
with open(out_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines))

print('\n'.join(out_lines))
