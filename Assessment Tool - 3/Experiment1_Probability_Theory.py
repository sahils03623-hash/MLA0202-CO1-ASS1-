# ==========================================================
# Experiment 1 : Probability Theory
# Dataset : Breast Cancer Wisconsin Diagnostic Dataset
# ==========================================================

from sklearn.datasets import load_breast_cancer
import pandas as pd

# Step 1 : Load Dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Target labels
# 0 = Malignant
# 1 = Benign

print("========================================")
print("Breast Cancer Wisconsin Dataset")
print("========================================")

print("\nTotal Samples :", len(y))

# Step 2 : Calculate Class Probabilities
malignant_count = (y == 0).sum()
benign_count = (y == 1).sum()

total = len(y)

P_malignant = malignant_count / total
P_benign = benign_count / total

print("\nClass Counts")
print("----------------------")
print("Malignant :", malignant_count)
print("Benign    :", benign_count)

print("\nPrior Probabilities")
print("----------------------")
print("P(Malignant) =", round(P_malignant,4))
print("P(Benign)    =", round(P_benign,4))

# Step 3 : Predict New Instance using Prior Probability
print("\nPredicting New Data Instance...")
print("--------------------------------")

new_instance = X.iloc[0]

print("New Instance (First Sample):")
print(new_instance)

if P_benign > P_malignant:
    prediction = "Benign"
else:
    prediction = "Malignant"

print("\nPredicted Class :", prediction)

# Step 4 : Display Conclusion
print("\n========================================")
print("Result")
print("========================================")
print("Probability of Malignant =", round(P_malignant,4))
print("Probability of Benign    =", round(P_benign,4))
print("Predicted Class =", prediction)
