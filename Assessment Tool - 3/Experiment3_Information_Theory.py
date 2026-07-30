# ==========================================================
# Experiment 3 : Information Theory
# Dataset : Play Tennis Dataset
# ==========================================================

import pandas as pd
import math

# ----------------------------------------------------------
# Step 1 : Create Play Tennis Dataset
# ----------------------------------------------------------

data = {
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain",
                "Overcast","Sunny","Sunny","Rain","Sunny",
                "Overcast","Overcast","Rain"],

    "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool",
                    "Cool","Mild","Cool","Mild","Mild",
                    "Mild","Hot","Mild"],

    "Humidity": ["High","High","High","High","Normal","Normal",
                 "Normal","High","Normal","Normal","Normal",
                 "High","Normal","High"],

    "Wind": ["Weak","Strong","Weak","Weak","Weak","Strong",
             "Strong","Weak","Weak","Weak","Strong",
             "Strong","Weak","Strong"],

    "Play": ["No","No","Yes","Yes","Yes","No",
             "Yes","No","Yes","Yes","Yes",
             "Yes","Yes","No"]
}

df = pd.DataFrame(data)

print("========================================")
print("Play Tennis Dataset")
print("========================================")
print(df)

# ----------------------------------------------------------
# Step 2 : Entropy Function
# ----------------------------------------------------------

def entropy(target):
    values = target.value_counts()
    total = len(target)
    ent = 0

    for count in values:
        p = count / total
        ent -= p * math.log2(p)

    return ent

# ----------------------------------------------------------
# Step 3 : Information Gain Function
# ----------------------------------------------------------

def information_gain(data, attribute, target):

    total_entropy = entropy(data[target])

    weighted_entropy = 0

    for value in data[attribute].unique():

        subset = data[data[attribute] == value]

        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])

    return total_entropy - weighted_entropy

# ----------------------------------------------------------
# Step 4 : Calculate Entropy
# ----------------------------------------------------------

target_entropy = entropy(df["Play"])

print("\nEntropy of Play =", round(target_entropy, 4))

# ----------------------------------------------------------
# Step 5 : Calculate Information Gain
# ----------------------------------------------------------

print("\nInformation Gain")

attributes = ["Outlook", "Temperature", "Humidity", "Wind"]

gains = {}

for attribute in attributes:

    gain = information_gain(df, attribute, "Play")

    gains[attribute] = gain

    print(attribute, "=", round(gain, 4))

# ----------------------------------------------------------
# Step 6 : Best Attribute
# ----------------------------------------------------------

best_attribute = max(gains, key=gains.get)

print("\nHighest Information Gain =", best_attribute)

# ----------------------------------------------------------
# Step 7 : Result
# ----------------------------------------------------------

print("\n========================================")
print("Result")
print("========================================")
print("Entropy =", round(target_entropy, 4))
print("Best Attribute =", best_attribute)
