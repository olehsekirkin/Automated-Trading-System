# Automated-Trading-System
Automated Trading System: Data-driven Approach with Machine Learning Models

<p align="center">
  <img src="https://media.geeksforgeeks.org/wp-content/cdn-uploads/20200522142429/Why-Python-Is-Used-For-Developing-Automated-Trading-Strategy1.png" alt="Title" width="380px" height="195px">
</p>

## Description

Embarking on the exciting journey of machine learning models (MLM) marks a significant milestone in my exploration of the vast realm of artificial intelligence. In this project, I've immersed myself in the intricacies of MLM, tackling the entire process independently – from the foundational step of data collection to the intricate evaluation and seamless integration with brokerage APIs.

### Understanding Machine Learning Models:

Machine learning models, a subset of artificial intelligence, empower systems to learn patterns and make predictions or decisions without explicit programming. In the financial domain, MLM play a pivotal role, offering insights and predictions that can influence investment strategies. This project delves into the nuances of MLM, employing a comprehensive approach that encompasses data collection, preprocessing, feature engineering, and the implementation of two distinct models - Long Short-Term Memory (LSTM) and a decision tree model.

### Project Overview:

1. Data Collection:
The journey begins with the meticulous collection of relevant data. Accurate and comprehensive datasets form the foundation for effective machine learning, and in this project, I've taken the initiative to gather data essential for financial analysis. The datacollection.py serves this purpose, thanks to the yfinance library I could input a starting date, an ending date and the interval (1d, 1wk, 1mo, ...) and get in response the dataset with all the information I needed about a stock of my choice. The numeric columns have 3 decimals by default, this can be changed too. The .csv goes straight to the desktop.

This default .csv includes the date, open, high, low, close, adj close and volume of the selected stock.

2. Data Preprocessing:
Raw data is often untamed and requires careful preprocessing. In this phase, I've applied techniques to clean, structure, and enhance the data, ensuring its suitability for input into the machine learning models. Any duplicates, missing values, etc are dropped and a new .csv is created and sent to the desktop, for future usage.

3. Feature Engineering:
To extract meaningful insights, I've delved into feature engineering, incorporating various technical indicators, price-based metrics (included in the default .csv), and trend indicators. These engineered features enrich the dataset, providing valuable information for the machine learning algorithms.

Feature 1 includes: Open, High, Low, Close, Adj Close, Volume, SMA, EMA, RSI, MACD, upper band, lower band, %K, %D, ATR, momentum.
Feature 2 includes: Open, High, Low, Close, Adj Close, Volume, adx, aroon up, aroon down, CCI, EMA, MACD, MACD signal, PSAR, STC.

%K being the Fast Stochastic Oscillator, %D the Slow Stochastic Oscillator, SMA and EMA are Simple Moving Average and Exponential Moving Average, RSI is Relative Strength Index, MACD Moving Average Convergence Divergence. ADX stands for Average Directional Index, CCI is Commodity Channel Index, PSAR Parabolic Stop and Reverse and STC Super Trend Indicator.

4. Machine Learning Models:
The heart of the project lies in the implementation of two distinct models. The LSTM model, known for its ability to capture temporal dependencies, is employed to analyze sequential data, while a decision tree model adds a different perspective, leveraging a tree-like model of decisions for robust predictions. ML1.py contains the LSTM model and ML2.py contains the Decision Tree Model.

The model evaluation is already integrated within the training of the MLMs (ML1.py and ML2.py), so for the LSTM model I went with MSE (Mean Squared Error), RMSE (Root Mean Squared Error), MAE (Mean Absolute Error) and MAPE (Mean Absolute Percentage Error). For the Decision Tree model I went with MSE, RMSE, MAE and R2 (Coefficient of Determination).

Now I really have to do a quick explanation of what both LTSM model and DT model are, because this will help us understand what we are working with.


### LSTM model
The Long Short-Term Memory (LSTM) model represents a breakthrough in the field of recurrent neural networks (RNNs), specifically designed to address the limitations of traditional RNNs when dealing with long sequences of data. LSTMs are particularly adept at learning and retaining information over extended periods, making them suitable for tasks involving time series data, natural language processing, and other applications with complex temporal dependencies.

At the core of the LSTM architecture are memory cells equipped with specialized gates: input gate, forget gate, and output gate. These gates regulate the flow of information within the network, allowing LSTMs to selectively store, update, or discard information at different time steps. The input gate determines which information from the current input is relevant and should be stored in the memory cell. The forget gate decides what information to discard from the memory cell, preventing the model from being overwhelmed by irrelevant details. Lastly, the output gate controls the information that is used to make predictions or influence subsequent computations.

One of the key advantages of LSTMs is their ability to mitigate the vanishing gradient problem commonly encountered in traditional RNNs. The vanishing gradient problem arises when the gradients of the loss function become extremely small during backpropagation, hindering the training of the model. LSTMs address this issue by maintaining a more stable gradient flow through the use of the gating mechanisms, allowing them to capture and utilize long-term dependencies in the data.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/LSTM_Cell.svg/1200px-LSTM_Cell.svg.png" alt="Title" width="350px" height="215px">
</p>

### Decision Tree model
A decision tree model is a versatile and interpretable machine learning algorithm that is employed for both classification and regression tasks. The model takes the form of a hierarchical tree structure, where each node represents a decision based on a specific feature or attribute. The tree branches out into different outcomes, with each internal node representing a decision point, and each leaf node representing the final decision or prediction.

The construction of a decision tree involves recursively splitting the dataset based on features that provide the most significant information gain. Information gain measures the reduction in uncertainty or entropy after a particular split, and the algorithm aims to maximize this gain at each decision node. This process continues until the tree is grown to a predefined depth or until further splitting fails to yield substantial information gain.
Decision trees are known for their transparency and interpretability, as the resulting model is easily visualized and understood.

They are particularly useful for tasks where the decision-making process needs to be explained or validated by domain experts. Additionally, decision trees can handle both numerical and categorical data, making them applicable to a wide range of real-world problems. However, to prevent overfitting, techniques such as pruning or setting a minimum number of samples per leaf are often employed during the model-building process.

<p align="center">
  <img src="https://365datascience.com/resources/blog/rr6cuudl59r-decision-trees-image1.png" alt="Title" width="350px" height="200px">
</p>

5. Backtesting: Both models have variables that you can modify to try and get the best of it but, what is the best combination? The LSTM model has n_steps, epochs, batch_size and test_size, there is thousand of different combinations that you need to try before deciding which one is the most accurate. But that changes in "ML1.2.py", where I added "#CONFIGURATION", where you can modify which configurations you want to try and let Python work through them, giving you only the best (based on either RMSE, MSE, MAE or MAPE, your choice)

## What I Learned

I gained hands-on experience in data collection, preprocessing, and feature engineering. Leveraging libraries like numpy, pandas, and yfinance, I created a robust dataset for financial analysis.

Implementing Long Short-Term Memory (LSTM) and Decision Tree models provided insights into handling sequential data and transparent decision-making processes. The LSTM's ability to capture long-term dependencies and the Decision Tree's interpretability were key takeaways.

The project enhanced my understanding of model evaluation, performance metrics, and effective communication of complex concepts. Overall, it laid the foundation for further exploration into advanced machine learning techniques.

Libraries used for this project so far that I've learned from: pandas, yfinance, numpy, pathlib, ta.trend, scikit, keras, tensorflow

## What's next

This is my biggest project so far. I feel like I still have A LOT to learn about Machine Learning models and application, so I will continue learning as I continue along this project. As the project itself, I want to implement model evaluation techniques, implement a trading system, integrate all of this with a brokerage API and some kind of monitoring system.
