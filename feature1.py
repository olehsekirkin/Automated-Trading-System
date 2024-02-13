import pandas as pd
import numpy as np
from pathlib import Path

# Load your existing .csv file
input_csv_path = "C:\\Users\\YOURUSERNAME\\Desktop\\preprocessed_stock_data.csv"
df = pd.read_csv(input_csv_path)

# Calculate technical indicators
df["SMA"] = df["Close"].rolling(window=20).mean()  # Simple Moving Average
df["EMA"] = df["Close"].ewm(span=20, adjust=False).mean()  # Exponential Moving Average
df["RSI"] = 100 - (100 / (1 + (df["Close"].diff(1) / df["Close"]).cumsum() / 14))  # Relative Strength Index
df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()  # Moving Average Convergence Divergence

# Bollinger Bands
df["rolling_mean"] = df["Close"].rolling(window=20).mean()
df["rolling_std"] = df["Close"].rolling(window=20).std()
df["upper_band"] = df["rolling_mean"] + (df["rolling_std"] * 2)
df["lower_band"] = df["rolling_mean"] - (df["rolling_std"] * 2)
df.drop(["rolling_mean", "rolling_std"], axis=1, inplace=True)

# Stochastic Oscillator
df["%K"] = (df["Close"] - df["Low"].rolling(window=14).min()) / (df["High"].rolling(window=14).max() - df["Low"].rolling(window=14).min()) * 100
df["%D"] = df["%K"].rolling(window=3).mean()

# Average True Range (ATR)
df["TR"] = np.maximum.reduce([df["High"] - df["Low"], abs(df["High"] - df["Close"].shift(1)), abs(df["Low"] - df["Close"].shift(1))])
df["ATR"] = df["TR"].rolling(window=14).mean()
df.drop("TR", axis=1, inplace=True)

# Momentum
df["Momentum"] = df["Close"].diff(10)

# Save the new DataFrame with technical indicators to a new .csv file
output_csv_path = str(Path.home() / "Desktop" / "feature1.csv")
df.to_csv(output_csv_path, index=False)

print(f"Technical indicators calculated and saved to: {output_csv_path}")
