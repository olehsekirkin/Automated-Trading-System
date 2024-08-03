# Data-Driven Finance & Machine Learning Application

<p align="center">
  <img src="https://startagist.com/wp-content/uploads/2024/02/Finance-sectors-AI.jpg" alt="Finance and AI" width="540px" height="320px">
</p>

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
  - [Data Collection](#1-data-collection)
  - [Data Preprocessing](#2-data-preprocessing)
  - [Feature Engineering](#3-feature-engineering)
  - [Machine Learning Models](#4-machine-learning-models)
  - [Backtesting](#5-backtesting)
- [Model Descriptions](#model-descriptions)
  - [LSTM Model](#lstm-model)
  - [Random Forest Regression Model](#random-forest-regression-model)
  - [AdaBoost for Decision Trees](#adaboost-for-decision-trees)

## Introduction

This project represents a comprehensive exploration of machine learning models (MLM) in the realm of financial analysis. It encompasses the entire process from data collection to model evaluation and integration with brokerage APIs, showcasing the power of artificial intelligence in financial decision-making.

## Project Overview

### 1. Data Collection
- Utilizes `yfinance` library for gathering stock data
- Customizable date range and intervals
- Outputs include: date, open, high, low, close, adjusted close, and volume

### 2. Data Preprocessing
- Cleans and structures raw data
- Handles duplicates and missing values
- Generates a refined CSV for further analysis

### 3. Feature Engineering
Incorporates various technical indicators:
- Feature Set 1: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic Oscillators, ATR, Momentum
- Feature Set 2: ADX, Aroon Indicators, CCI, EMA, MACD, PSAR, STC

### 4. Machine Learning Models
Implements two distinct models:
- LSTM (Long Short-Term Memory) for temporal dependency analysis
- Random Forest Regression for robust predictions

### 5. Backtesting
- LSTM: Optimizes n_steps, epochs, batch_size, and test_size
- Random Forest: Focuses on test_size and n_estimators optimization

## Model Descriptions

### LSTM Model
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/LSTM_Cell.svg/1200px-LSTM_Cell.svg.png" alt="LSTM Cell" width="350px" height="215px">
</p>

Long Short-Term Memory (LSTM) models excel at capturing long-term dependencies in sequential data, making them ideal for time series analysis in finance.

In this project, I've implemented an LSTM (Long Short-Term Memory) model for stock price prediction. Here's a brief overview of my approach:

1. Data Acquisition: I utilize the Alpha Vantage API to fetch historical stock data for any specified symbol. This allows me to work with up-to-date market information.

2. Data Preprocessing: I extract the closing prices and normalize them using MinMaxScaler. Then, I structure the data into sequences of a specified length (n_steps) to create input-output pairs for training. This step is crucial for preparing the data in a format suitable for the LSTM model.

3. Model Architecture: My LSTM model consists of two LSTM layers followed by a Dense layer. The first LSTM layer has 50 units and returns sequences, while the second LSTM layer also has 50 units but doesn't return sequences. The final Dense layer has 1 unit for outputting the prediction.

4. Model Training: I compile the model using the Adam optimizer and mean squared error as the loss function. The model is then trained on the prepared sequences for a specified number of epochs and batch size.

5. Prediction and Evaluation: After training, I use the model to predict on the test set. I inverse-transform the predictions to get actual price values and evaluate performance using multiple metrics: MSE, RMSE, MAE, and MAPE.

6. Hyperparameter Tuning: To optimize the model, I experiment with different combinations of hyperparameters:
   - Sequence length (n_steps): 10 and 100
   - Test set size: 10% and 50%
   - Number of epochs: 10 and 100
   - Batch size: 32, 64, and 128

7. Optimization: My script runs multiple experiments with these different hyperparameter combinations. It then selects the best configuration based on the lowest RMSE.

8. Visualization: Finally, I plot the best model's predictions against the true values for visual comparison.

This implementation focuses on leveraging LSTM's ability to capture temporal dependencies in stock price movements. By systematically experimenting with different configurations, I aim to find the optimal model for accurate price prediction. The flexible design allows for easy testing of various hyperparameters, making it adaptable to different stock data and market conditions.

Future enhancements I'm considering include incorporating additional technical indicators, integrating market sentiment data, implementing risk management options, and developing real-time prediction capabilities.

### Random Forest Regression Model
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:567/1*Mb8awDiY9T6rsOjtNTRcIg.png" alt="Random Forest" width="350px" height="200px">
</p>

Random Forest Regression combines multiple decision trees to create a robust and interpretable model, suitable for both numerical and categorical data.

### AdaBoost for Decision Trees
<p align="center">
  <img src="https://editor.analyticsvidhya.com/uploads/98218100.JPG" alt="AdaBoost" width="350px" height="200px">
</p>

AdaBoost enhances decision tree performance by combining weak learners into a strong classifier, focusing on hard-to-classify instances to improve overall accuracy.

---

For more details on implementation and usage, please refer to the individual script documentation.
