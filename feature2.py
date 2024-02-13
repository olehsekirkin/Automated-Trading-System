import pandas as pd
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator, MACD, PSARIndicator, STCIndicator

# Load your CSV file
input_csv_path = 'C:\\Users\\olehs\\Desktop\\preprocessed_stock_data.csv'
output_csv_path = 'C:\\Users\\olehs\\Desktop\\feature2.csv'

df = pd.read_csv(input_csv_path)

# Calculate trend indicators
df['adx'] = ADXIndicator(df['High'], df['Low'], df['Close']).adx()

# Manually calculate Aroon indicator
window = 14  # You can adjust the window as needed
high_window = df['High'].rolling(window=window, min_periods=1).apply(lambda x: x.argmax(), raw=True)
low_window = df['Low'].rolling(window=window, min_periods=1).apply(lambda x: x.argmin(), raw=True)

df['aroon_up'] = (window - high_window) / window * 100
df['aroon_down'] = (window - low_window) / window * 100

df['cci'] = CCIIndicator(df['High'], df['Low'], df['Close']).cci()
df['ema'] = EMAIndicator(df['Close']).ema_indicator()
macd = MACD(df['Close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['psar'] = PSARIndicator(df['High'], df['Low'], df['Close']).psar()
df['stc'] = STCIndicator(df['Close']).stc()

# Save the result to a new CSV file on the desktop
df.to_csv(output_csv_path, index=False)

print(f'Technical indicators have been calculated and saved to {output_csv_path}')
