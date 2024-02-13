# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load your .csv data
file_path = "C:\\Users\\YOURUSERNAME\\Desktop\\cleaned_file.csv"
data = pd.read_csv(file_path)

# Remove leading and trailing spaces from column names
data.columns = data.columns.str.strip()

# Define features (X) and target variable (y)
features = data[["Open", "High", "Low", "Close", "Adj Close", "Volume", "SMA", "EMA", "RSI", "MACD", "upper_band", "lower_band", "%K", "%D", "ATR", "Momentum"]]
target = data["Close"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Now you can use the trained model to make predictions on new data
