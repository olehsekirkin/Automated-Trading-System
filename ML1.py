import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Load the .csv file
df = pd.read_csv("C:\\Users\\YOURUSERNAME\\Desktop\\cleaned_file1.csv")

# Drop unnecessary columns for this MLM
df = df[["Close", "SMA", "EMA", "RSI", "MACD", "upper_band", "lower_band", "%K", "%D", "ATR", "Momentum"]]

# Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Define the input features (X) and target variable (y)
X = df_scaled[:-1]  # Use all columns except the last row
y = df_scaled[1:, 0]  # Target variable is "Close" column shifted by 1

# Reshape input data
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Make predictions on the test set
predictions = model.predict(X_test)

# Concatenate X_test with predictions along axis=1
concatenated_data = np.concatenate((X_test[:, :, 0], predictions), axis=1)
# Inverse transform the concatenated data to the original scale
data_inv = scaler.inverse_transform(concatenated_data)
# Extract predictions_inv from the transformed data
predictions_inv = data_inv[:, 1]

# Similarly, for y_test
y_test_inv = scaler.inverse_transform(np.concatenate((X_test[:, :, 0], y_test.reshape(-1, 1)), axis=1))[:, 1]

# Calculate Mean Squared Error
mse = np.mean((predictions_inv - y_test_inv)**2)
print(f"Mean Squared Error: {mse}")
