# ==============================
# Gaussian Naive Bayes (Iris)
# ==============================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Train-test split (70/30)
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
)

# Model
gnb = GaussianNB()
gnb.fit(X_train_iris, y_train_iris)

# Prediction
y_pred_iris = gnb.predict(X_test_iris)

# Evaluation
print("===== Gaussian Naive Bayes on Iris Dataset =====")
print("Accuracy:", accuracy_score(y_test_iris, y_pred_iris))
print("Classification Report:")
print(classification_report(y_test_iris, y_pred_iris))


# ==============================
# Multinomial Naive Bayes (SMS Spam)
# ==============================

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", names=["label", "message"])

# Convert labels to binary
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X_text = df["message"]
y_text = df["label"]

# Vectorization (word frequencies)
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X_text)

# Train-test split
X_train_sms, X_test_sms, y_train_sms, y_test_sms = train_test_split(
    X_vec, y_text, test_size=0.3, random_state=42, stratify=y_text
)

# Model
mnb = MultinomialNB()
mnb.fit(X_train_sms, y_train_sms)

# Prediction
y_pred_sms = mnb.predict(X_test_sms)

# Evaluation
print("\n===== Multinomial Naive Bayes on SMS Spam Dataset =====")
print("Accuracy:", accuracy_score(y_test_sms, y_pred_sms))
print("Classification Report:")
print(classification_report(y_test_sms, y_pred_sms))
