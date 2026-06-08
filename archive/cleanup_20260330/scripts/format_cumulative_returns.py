import pandas as pd

# Load cumulative returns data
cumulative_file = "c:/Users/hg317/Desktop/projects/leadlag/output/cumulative_returns_from_march_1.csv"
cumulative_data = pd.read_csv(cumulative_file)

# Convert cumulative returns to percentage format
cumulative_data.set_index('date', inplace=True)
cumulative_data_percentage = cumulative_data * 100

# Save the formatted data to a new CSV file
output_file = "c:/Users/hg317/Desktop/projects/leadlag/output/cumulative_returns_percentage.csv"
cumulative_data_percentage.to_csv(output_file)

print(f"Cumulative returns in percentage format saved to {output_file}")