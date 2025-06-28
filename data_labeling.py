import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('btc_data_test.csv')

# Ensure sorted by timestamp
df = df.sort_values('Timestamp').reset_index(drop=True)

# Lookahead configuration: 120 minutes
lookahead_minutes = 120

# Original columns to retain
original_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Datetime']

# Calculate 2-hour forward percentage change (in percent, not fraction)
future_close = df['Close'].shift(-lookahead_minutes)
pct_change_percent = ((future_close - df['Close']) / df['Close']) * 100

# Assign score
df['Score'] = pct_change_percent

# Drop rows that don't have full 2-hour lookahead
df = df.iloc[:-lookahead_minutes].reset_index(drop=True)

# Save labeled dataset
df[original_cols + ['Score']].to_csv('btc_data_train_labeled.csv', index=False)

# Print statistics
print("Saved labeled data.")
print(f"Score range: {df['Score'].min():.3f}% to {df['Score'].max():.3f}%")
print(f"Score mean: {df['Score'].mean():.3f}%")
print(f"Score std: {df['Score'].std():.3f}%")

# Plot histogram of scores
plt.figure(figsize=(10, 6))
plt.hist(df['Score'], bins=100, edgecolor='black')
plt.title('Distribution of 2-Hour Percentage Price Change')
plt.xlabel('2-Hour % Change')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
