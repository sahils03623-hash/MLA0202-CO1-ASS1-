# LoanApprovalPrediction.py
# Loan Approval Prediction using Decision Tree Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample Dataset
data = {
    "Income": [25000, 40000, 55000, 30000, 70000, 65000, 45000, 80000, 35000, 90000],
    "CreditScore": [620, 700, 750, 650, 800, 780, 690, 820, 640, 850],
    "LoanAmount": [150000, 200000, 250000, 180000, 300000, 280000, 220000, 350000, 170000, 400000],
    "LoanApproved": [0, 1, 1, 0, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

print("Loan Approval Dataset")
print(df)

# Features and Target
X = df[["Income", "CreditScore", "LoanAmount"]]
y = df["LoanApproved"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

print("\nPredicted Loan Approval:")
print(predictions)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy, 2))
