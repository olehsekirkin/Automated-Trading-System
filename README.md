# FinML: Data-Driven Finance

## Introduction

Embarking on the exciting journey of machine learning models (MLM) marks a significant milestone in my exploration of the vast realm of artificial intelligence. In this project, I've immersed myself in the intricacies of MLM, tackling the entire process independently – from the foundational step of data collection to the intricate evaluation and seamless integration with brokerage APIs.

### Understanding Machine Learning Models:

Machine learning models, a subset of artificial intelligence, empower systems to learn patterns and make predictions or decisions without explicit programming. In the financial domain, MLM play a pivotal role, offering insights and predictions that can influence investment strategies. This project delves into the nuances of MLM, employing a comprehensive approach that encompasses data collection, preprocessing, feature engineering, and the implementation of various models.

## Project Overview:

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
The heart of the project lies in the implementation of two distinct models. The LSTM model, known for its ability to capture temporal dependencies, is employed to analyze sequential data, while a Random Regression model adds a different perspective, leveraging a tree-like model of decisions for robust predictions. LSTM1.py contains the LSTM model and randomforestregression.py contains the Random Regression Model.

The model evaluation is already integrated within the training of the MLMs (LSTM1.py and randomforestregression.py), so for the LSTM model I went with MSE (Mean Squared Error), RMSE (Root Mean Squared Error), MAE (Mean Absolute Error) and MAPE (Mean Absolute Percentage Error). For the Random Regression model I went with MSE, RMSE, MAE and R2 (Coefficient of Determination).

  5. Backtesting: Both models have variables that you can modify to try and get the best of it but, what is the best combination? The LSTM model has n_steps, epochs, batch_size and test_size, there is thousand of different combinations that you need to try before deciding which one is the most accurate. But that changes in "LSTM1.2.py", where I added "#CONFIGURATION", where you can modify which configurations you want to try and let Python work through them, giving you only the best (based on either RMSE, MSE, MAE or MAPE, your choice).

For the Random Regression model is much easier, playing around with the test_size is enough, then the other variable is n_estimators, which represents the number of trees in the forest, the higher the number the better the model, but the more time and computational power you need.

Now I really have to do a quick explanation of what both LTSM model and DT model are, because this will help us understand what we are working with.

### LSTM model
The Long Short-Term Memory (LSTM) model represents a breakthrough in the field of recurrent neural networks (RNNs), specifically designed to address the limitations of traditional RNNs when dealing with long sequences of data. LSTMs are particularly adept at learning and retaining information over extended periods, making them suitable for tasks involving time series data, natural language processing, and other applications with complex temporal dependencies.

At the core of the LSTM architecture are memory cells equipped with specialized gates: input gate, forget gate, and output gate. These gates regulate the flow of information within the network, allowing LSTMs to selectively store, update, or discard information at different time steps. The input gate determines which information from the current input is relevant and should be stored in the memory cell. The forget gate decides what information to discard from the memory cell, preventing the model from being overwhelmed by irrelevant details. Lastly, the output gate controls the information that is used to make predictions or influence subsequent computations.

One of the key advantages of LSTMs is their ability to mitigate the vanishing gradient problem commonly encountered in traditional RNNs. The vanishing gradient problem arises when the gradients of the loss function become extremely small during backpropagation, hindering the training of the model. LSTMs address this issue by maintaining a more stable gradient flow through the use of the gating mechanisms, allowing them to capture and utilize long-term dependencies in the data.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/LSTM_Cell.svg/1200px-LSTM_Cell.svg.png" alt="Title" width="350px" height="215px">
</p>

### Random Regression model
A Random Regression model is a versatile and interpretable machine learning algorithm that is employed for both classification and regression tasks. The model takes the form of a hierarchical tree structure, where each node represents a decision based on a specific feature or attribute. The tree branches out into different outcomes, with each internal node representing a decision point, and each leaf node representing the final decision or prediction.

The construction of a Random Regression involves recursively splitting the dataset based on features that provide the most significant information gain. Information gain measures the reduction in uncertainty or entropy after a particular split, and the algorithm aims to maximize this gain at each decision node. This process continues until the tree is grown to a predefined depth or until further splitting fails to yield substantial information gain.
Random Regressions are known for their transparency and interpretability, as the resulting model is easily visualized and understood.

They are particularly useful for tasks where the decision-making process needs to be explained or validated by domain experts. Additionally, Random Regressions can handle both numerical and categorical data, making them applicable to a wide range of real-world problems. However, to prevent overfitting, techniques such as pruning or setting a minimum number of samples per leaf are often employed during the model-building process.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:567/1*Mb8awDiY9T6rsOjtNTRcIg.png" alt="Title" width="350px" height="200px">
</p>

### AdaBoost for Decision Trees classifiers

AdaBoost, short for Adaptive Boosting, is a machine learning ensemble technique that combines multiple weak classifiers to create a strong classifier. The core principle behind AdaBoost is to fit a sequence of weak learners (i.e., models that are only slightly better than random guessing) on repeatedly modified versions of the data. The predictions from all of them are then combined through a weighted majority vote (or sum) to produce the final prediction. The data modifications at each iteration consist of applying weights to each of the training samples. Initially, these weights are all equal, but on each subsequent round, the weights of incorrectly classified instances are increased so that the weak learner is forced to focus on the hard cases and thus improve the overall model performance.

Decision Trees are a popular choice for use with AdaBoost due to their capacity to handle complex data structures and their tendency to overfit their training data. When used in AdaBoost:

- Initialization: Each data point is initially given the same weight.
- Iteration:
A Decision Tree is trained on the weighted data. This tree is typically a stump (a tree with a single decision node and two leaves), which constitutes a weak learner.
The tree is used to classify the training data, and the error is calculated based on the weighted instances. The error is the sum of the weights associated with the incorrectly classified instances.
A weight (alpha) is assigned to the stump based on its accuracy; more accurate stumps are given more weight.
The weights of the training instances are updated: weights are increased for those instances that were misclassified, making them more important for the next iteration.
The process repeats, each time focusing more on the instances that previous stumps misclassified.
- Combination: After many rounds, the algorithm combines the stumps into a single model, where each stump contributes to an ensemble prediction weighted by its accuracy.

<p align="center">
  <img src="https://editor.analyticsvidhya.com/uploads/98218100.JPG" alt="Title" width="350px" height="200px">
</p>

## What I Learned

I gained hands-on experience in data collection, preprocessing, and feature engineering. Leveraging libraries like numpy, pandas, and yfinance, I created a robust dataset for financial analysis.

Implementing Long Short-Term Memory (LSTM) and Random Regression models provided insights into handling sequential data and transparent decision-making processes. The LSTM's ability to capture long-term dependencies and the Random Regression's interpretability were key takeaways.

The project enhanced my understanding of model evaluation, performance metrics, and effective communication of complex concepts. Overall, it laid the foundation for further exploration into advanced machine learning techniques.

Libraries used for this project so far that I've learned from: pandas, yfinance, numpy, pathlib, ta.trend, scikit, keras, tensorflow

## What's next

This is my biggest project so far. I feel like I still have A LOT to learn about Machine Learning models and application, so I will continue learning as I continue along this project. As the project itself, I want to implement a trading system, integrate all of this with a brokerage API and some kind of monitoring system.
