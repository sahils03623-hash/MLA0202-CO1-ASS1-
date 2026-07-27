# CreditCardFraudDetection.py
# Credit Card Fraud Detection using Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample Dataset
data = {
    "TransactionAmount": [100, 250, 5000, 120, 7000, 300, 9000, 450, 15000, 200],
    "LocationRisk": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "OTPVerified": [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "Fraud": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

print("Credit Card Fraud Dataset")
print(df)

X = df[["TransactionAmount", "LocationRisk", "OTPVerified"]]
y = df["Fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nPredicted Fraud Status:")
print(predictions)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy, 2))
