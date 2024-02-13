import os
import pandas as pd
import yfinance as yf
import locale

# Set the locale to "en_US.utf-8"
locale.setlocale(locale.LC_NUMERIC, "en_US.utf-8")


# Function to fetch time series data from Yahoo Finance within a specified date range
def get_yahoo_finance_data(symbol, start_date, end_date, interval="1d"):
    stock = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    return stock


# Function to save DataFrame to .csv file on desktop with date format as MM-DD-YY
def save_to_csv(dataframe, file_name):
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    file_path = os.path.join(desktop_path, file_name)

    # Round numeric columns to 3 decimal places (can change this in case you need more or less)
    dataframe = dataframe.round(3)

    # Convert the date column to the desired format (MM-DD-YY)
    dataframe.index = dataframe.index.strftime("%m-%d-%Y")

    dataframe.to_csv(file_path, index=True)
    print(f"Data saved to {file_path}")

# Parameters: 
symbol = "NVDA"  # Replace with the desired stock symbol
interval = "1d"  # You can change this to other intervals like "1wk", "1mo", ...

# Input for start and end dates in MM-DD-YYYY format
start_date = input("Enter the start date (MM-DD-YYYY): ")
end_date = input("Enter the end date (MM-DD-YYYY): ")

# Convert input dates to pandas datetime
start_date = pd.to_datetime(start_date, format="%m-%d-%Y")
end_date = pd.to_datetime(end_date, format="%m-%d-%Y")

# Fetch data within the specified date range
stock_data = get_yahoo_finance_data(symbol, start_date, end_date, interval)

# Display the obtained data in the console
if not stock_data.empty:
    print("Data obtained:")
    print(stock_data)

    # Save data to .csv file on desktop
    save_to_csv(stock_data, f"{symbol}_data.csv")
else:
    print(f"No data available for {symbol} within the specified date range.")
