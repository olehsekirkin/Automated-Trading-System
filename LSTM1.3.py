# To do:
# RSI, moving average? Add technical indicators? Market sentiment data?
# Risk management options (stop loss, position sizing, ...)
# Real time prediction, need for new data everytime to train
# Backtest

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from alpha_vantage.timeseries import TimeSeries

def load_and_prepare_data(n_steps, api_key, symbol):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='full')

    data = data[['4. close']]
    data.columns = ['Close']
    data.index = pd.to_datetime(data.index)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
  
    X, y = [], []
    for i in range(len(scaled_data) - n_steps - 1):
        X.append(scaled_data[i:(i+n_steps), :])
        y.append(scaled_data[i + n_steps, 0])
    X, y = np.array(X), np.array(y)

    return X, y, scaler

def run_experiment(n_steps, test_size, epochs, batch_size, api_key, symbol):
    X, y, scaler = load_and_prepare_data(n_steps, api_key, symbol)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(units=50),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = model.predict(X_test)
    y_test_inv = y_test * scaler.scale_[0] + scaler.min_[0]
    y_pred_inv = y_pred * scaler.scale_[0] + scaler.min_[0]

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)

    return mse, rmse, mae, mape, y_test_inv, y_pred_inv

api_key = 'YOUR_ALPHAVANTAGE_API_KEY'  # ALPHA VANTAGE API KEY
symbol = 'AAPL'  # STOCK TICKER

n_steps_options = (10, 100)
test_size_options = (0.1, 0.5)
epochs_options = (10, 100)
batch_size_options = [32, 64, 128]

results = []

for n_steps in n_steps_options:
    for test_size in test_size_options:
        for epochs in epochs_options:
            for batch_size in batch_size_options:
                mse, rmse, mae, mape, y_test_inv, y_pred_inv = run_experiment(n_steps, test_size, epochs, batch_size, api_key, symbol)
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

plt.figure(figsize=(14, 7))
plt.plot(best_config['y_test_inv'], label="True")
plt.plot(best_config['y_pred_inv'], label="Predicted")
plt.title("LSTM Model Prediction vs True Value (Best Configuration)")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend()
plt.show()
