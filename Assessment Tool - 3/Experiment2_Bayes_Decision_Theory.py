# ==========================================================
# Experiment 2 : Bayes Decision Theory
# Dataset : SMS Spam Collection Dataset
# ==========================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ----------------------------------------------------------
# Step 1 : Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None,
                 names=["Label", "Message"])

print("========================================")
print("SMS Spam Collection Dataset")
print("========================================")

print("\nTotal Messages :", len(df))

print("\nClass Counts")
print(df["Label"].value_counts())

# ----------------------------------------------------------
# Step 2 : Split Dataset
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["Message"],
    df["Label"],
    test_size=0.2,
    random_state=42
)

# ----------------------------------------------------------
# Step 3 : Convert Text to Numeric Form
# ----------------------------------------------------------
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------------------------------------
# Step 4 : Train Naive Bayes Model
# ----------------------------------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ----------------------------------------------------------
# Step 5 : Calculate Posterior Probability
# ----------------------------------------------------------
accuracy = accuracy_score(y_test, model.predict(X_test_vec))

print("\nModel Accuracy :", round(accuracy*100,2), "%")

sample_message = ["Congratulations! You have won a FREE mobile. Claim now."]

sample_vector = vectorizer.transform(sample_message)

posterior = model.predict_proba(sample_vector)

print("\nPosterior Probabilities")
print("--------------------------------")
print("Ham  :", round(posterior[0][0],4))
print("Spam :", round(posterior[0][1],4))

prediction = model.predict(sample_vector)[0]

print("\nPrediction :", prediction)

# ----------------------------------------------------------
# Step 6 : Result
# ----------------------------------------------------------
print("\n========================================")
print("Result")
print("========================================")
print("The posterior probabilities were calculated")
print("using Naive Bayes and the message was")
print("classified as:", prediction)
