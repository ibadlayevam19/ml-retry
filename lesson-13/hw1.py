# Bernoulli Naive Bayes for SMS Spam Classification (Binary Features)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

# Step 2: Convert labels to binary (spam=1, ham=0)
data["label"] = data["label"].map({"ham": 0, "spam": 1})

X = data["message"]
y = data["label"]

# Step 3: Binary CountVectorizer
vectorizer = CountVectorizer(binary=True)
X_vectorized = vectorizer.fit_transform(X)

# Step 4: Train-test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=42, stratify=y
)

# Step 5: Train Bernoulli Naive Bayes
model = BernoulliNB()
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(cm)
