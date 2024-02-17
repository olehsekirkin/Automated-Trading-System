import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

def load_and_prepare_data(n_steps):
# Load the dataset
    data = pd.read_csv("C:\\Users\\YOURUSERNAME\\Desktop\\cleaned_file1.csv")

# Convert Date column to datetime type
    data['Date'] = pd.to_datetime(data['Date'])

# Set Date column as index
    data.set_index('Date', inplace=True)

# Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

# Split data into input and output
    X, y = [], []
    for i in range(len(scaled_data) - n_steps - 1):
        X.append(scaled_data[i:(i+n_steps), :])
        y.append(scaled_data[i + n_steps, 4])  # Adjust this index according to your target column
    X, y = np.array(X), np.array(y)

    return X, y, scaler

def run_experiment(n_steps, test_size, epochs, batch_size):
    X, y, scaler = load_and_prepare_data(n_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

# Define the LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(units=50),
        Dense(units=1)
    ])

# Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

# Make predictions
    y_pred = model.predict(X_test)
    y_test_inv = y_test * scaler.scale_[4] + scaler.min_[4]
    y_pred_inv = y_pred * scaler.scale_[4] + scaler.min_[4]

# Model evaluation
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)

    return mse, rmse, mae, mape, y_test_inv, y_pred_inv

# n_steps is set to go through all the numbers between 10 and 100, same for test_size but from 0.1 to 0.5, epochs go from 10 to 100 and batch_size will use
# use 32, 64 or 128.
n_steps_options = (10,100)
test_size_options = (0.1, 0.5)
epochs_options = (10,100)
batch_size_options = [32, 64, 128]

results = []

for n_steps in n_steps_options:
    for test_size in test_size_options:
        for epochs in epochs_options:
            for batch_size in batch_size_options:
                mse, rmse, mae, mape, y_test_inv, y_pred_inv = run_experiment(n_steps, test_size, epochs, batch_size)
                results.append({
                    'n_steps': n_steps,
                    'test_size': test_size,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape,
                    'y_test_inv': y_test_inv,
                    'y_pred_inv': y_pred_inv
                })

results_df = pd.DataFrame(results)
best_config_index = results_df['RMSE'].idxmin()
best_config = results_df.iloc[best_config_index]

print("Best configuration based on RMSE:", best_config[['n_steps', 'test_size', 'epochs', 'batch_size', 'MSE', 'RMSE', 'MAE', 'MAPE']])

# Plotting the results for the best configuration
plt.figure(figsize=(14, 7))
plt.plot(best_config['y_test_inv'], label="True")
plt.plot(best_config['y_pred_inv'], label="Predicted")
plt.title("LSTM Model Prediction vs True Value (Best Configuration)")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend()
plt.show()
