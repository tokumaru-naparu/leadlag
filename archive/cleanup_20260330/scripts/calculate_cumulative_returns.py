import pandas as pd

# Load returns data
returns_file = "c:/Users/hg317/Desktop/projects/leadlag/data/history/returns.csv"
returns_data = pd.read_csv(returns_file)

# Filter data from March 1, 2026
returns_data['date'] = pd.to_datetime(returns_data['date'])
march_1_date = pd.Timestamp('2026-03-01')
filtered_returns = returns_data[returns_data['date'] >= march_1_date]

# Calculate cumulative returns
filtered_returns.set_index('date', inplace=True)
cumulative_returns = (1 + filtered_returns).cumprod() - 1

# Save cumulative returns to a new CSV file
output_file = "c:/Users/hg317/Desktop/projects/leadlag/output/cumulative_returns_from_march_1.csv"
cumulative_returns.to_csv(output_file)

print(f"Cumulative returns from March 1 saved to {output_file}")