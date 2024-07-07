import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\\olehs\\Desktop\\cleaned_file2.csv')

df['Target'] = (df['Close'] > df['Open']).astype(int)

X = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'adx', 'aroon_up', 'aroon_down', 'cci', 'ema', 'macd', 'macd_signal', 'psar', 'stc']]
y = df['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

param_grid = {
    'estimator__max_depth': [1, 2, 3],
    'n_estimators': (50, 300),
    'learning_rate': (0.01, 1)
}
ada_boost = GridSearchCV(AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=42),
                         param_grid=param_grid, cv=5, n_jobs=-1)

ada_boost.fit(X_train, y_train)

print("Best parameters found: ", ada_boost.best_params_)

y_pred = ada_boost.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Classification Report
print(classification_report(y_test, y_pred))

# Feature Importances
# Extract the best estimator
best_model = ada_boost.best_estimator_
# Get feature importances from the AdaBoost model (assuming it's the best model found)
feature_importances = best_model.feature_importances_

# Plotting feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances, y=X.columns)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')

results = pd.DataFrame(ada_boost.cv_results_)
plt.figure(figsize=(10, 6))
sns.lineplot(data=results, x='param_n_estimators', y='mean_test_score', marker='o')
plt.title('Grid Search Performance')
plt.xlabel('Number of Estimators')
plt.ylabel('Mean Test Score')

plt.show()
