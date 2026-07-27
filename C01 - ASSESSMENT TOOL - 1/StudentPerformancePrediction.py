# Student Performance Prediction using Multiple Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
student_data = pd.read_csv("student_performance.csv")

print("Dataset Preview")
print(student_data.head())

print("\nMissing Values")
print(student_data.isnull().sum())

# Features and Target
X = student_data.drop("FinalMarks", axis=1)
y = student_data["FinalMarks"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=10
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# Create and Train Model
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

# Prediction
predicted_marks = mlr_model.predict(X_test)

print("\nPredicted Final Marks")
print(predicted_marks.round(2))

print("\nRegression Coefficients")
print(mlr_model.coef_)

print("Intercept")
print(mlr_model.intercept_)

# Accuracy
accuracy = r2_score(y_test, predicted_marks)
print("\nR² Score =", round(accuracy, 2))
