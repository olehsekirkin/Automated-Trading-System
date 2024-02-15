import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

# Load the dataset
data = pd.read_csv("C:\\Users\\YOURUSERNAME\\Desktop\\cleaned_file1.csv")

# Convert Date column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Set Date column as index
data.set_index('Date', inplace=True)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define the number of timesteps
n_steps = 60

# Split data into input and output
X, y = [], []
for i in range(len(scaled_data) - n_steps - 1):
    X.append(scaled_data[i:(i+n_steps), :])
    y.append(scaled_data[i + n_steps, 4])  # Closing price (target)

X, y = np.array(X), np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Manual inverse transformation for y_test
y_test_inv = y_test * scaler.scale_[4] + scaler.min_[4]

# Manual inverse transformation for y_pred
y_pred_inv = y_pred * scaler.scale_[4] + scaler.min_[4]

# Model evaluation. Decided to go with Mean Absolute Error (MAE), Mean Absolute Scaled Error (MASE), Accuracy Percent, Root Mean Squared Error (RMSE), Mean Absolute Percent Error (MAPE)
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)

plt.figure(figsize=(14, 7))
plt.plot(y_test_inv, label="True")
plt.plot(y_pred_inv, label="Predicted")
plt.title("LSTM Model Prediction vs True Value")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend()
plt.show()
