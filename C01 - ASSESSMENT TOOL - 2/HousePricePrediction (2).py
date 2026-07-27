# House Price Prediction using Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
house_data = pd.read_csv("house_price.csv")

print("Dataset Preview")
print(house_data.head())

print("\nDataset Information")
print(house_data.info())

print("\nMissing Values")
print(house_data.isnull().sum())

# Features and Target
X = house_data.drop("Price", axis=1)
y = house_data["Price"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=7
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# Create and Train Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Prediction
predicted_price = lr_model.predict(X_test)

print("\nPredicted House Prices")
print(predicted_price.round(2))

print("\nModel Coefficients")
print(lr_model.coef_)

print("Intercept")
print(lr_model.intercept_)

# Accuracy
accuracy = r2_score(y_test, predicted_price)
print("\nR² Score =", round(accuracy, 2))
