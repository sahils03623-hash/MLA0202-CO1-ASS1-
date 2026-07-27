# MedicalDiseasePrediction.py
# Medical Disease Prediction using Decision Tree Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample Dataset
data = {
    "Glucose": [80, 120, 150, 95, 170, 130, 110, 180, 140, 100],
    "BloodPressure": [70, 80, 90, 72, 95, 85, 78, 96, 88, 75],
    "BMI": [22, 28, 35, 24, 37, 31, 27, 40, 33, 25],
    "Disease": [0, 0, 1, 0, 1, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

print("Medical Disease Dataset")
print(df)

X = df[["Glucose", "BloodPressure", "BMI"]]
y = df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nPredicted Disease Status:")
print(predictions)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy, 2))
