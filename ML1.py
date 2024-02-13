import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the .csv file
df = pd.read_csv("C:\\Users\\YOURUSERNAME\\Desktop\\cleaned_file.csv")

# Drop unnecessary columns for this MLM
df = df[["Close", "SMA", "EMA", "RSI", "MACD", "upper_band", "lower_band", "%K", "%D", "ATR", "Momentum"]]

# Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Define the input features (X) and target variable (y)
X = df_scaled[:-1]  # Use all columns except the last row
y = df_scaled[1:, 0]  # Target variable is "Close" column shifted by 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model.fit(X_train, y_train, epochs=50, batch_size=32)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predictions = model.predict(X_test)

# Inverse transform the predictions and actual values to the original scale
predictions_inv = scaler.inverse_transform(np.concatenate((X_test[:, :, 0], predictions), axis=1))[:, 1]
y_test_inv = scaler.inverse_transform(np.concatenate((X_test[:, :, 0], y_test.reshape(-1, 1)), axis=1))[:, 1]

# Calculate and print metrics (e.g., Mean Squared Error)
mse = np.mean((predictions_inv - y_test_inv)**2)
print(f"Mean Squared Error: {mse}")
