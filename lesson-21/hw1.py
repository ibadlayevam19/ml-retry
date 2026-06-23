# Boosting and Stacking - Binary Classification (Breast Cancer Dataset)

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

X, y = load_breast_cancer(return_X_y=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# AdaBoost (Boosting)
# --------------------------------------------------

ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=0.1,
    random_state=42
)

ada.fit(X_train, y_train)

# Predictions
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)

print("AdaBoost Results:")
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Train F1-score: {f1_score(y_train, y_train_pred):.4f}")
print(f"Test F1-score : {f1_score(y_test, y_test_pred):.4f}")

# --------------------------------------------------
# Gradient Boosting (3 Experiments)
# --------------------------------------------------

gb_experiments = [
    {"n_estimators": 50, "learning_rate": 0.1},
    {"n_estimators": 100, "learning_rate": 0.1},
    {"n_estimators": 100, "learning_rate": 0.05},
]

results = []

for params in gb_experiments:
    gb = GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)

    results.append({
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    })

gb_results = pd.DataFrame(results)
print("\nGradient Boosting Results:")
print(gb_results)

# --------------------------------------------------
# Stacking Classifier
# --------------------------------------------------

# Base models (with scaling where needed)
estimators = [
    ("lr", Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000))
    ])),
    ("dt", DecisionTreeClassifier(random_state=42)),
    ("knn", Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ]))
]

# Meta-learner
meta_learner = LogisticRegression(max_iter=2000)

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_learner,
    cv=5
)

stack.fit(X_train, y_train)

# Predictions
y_pred_stack = stack.predict(X_test)
y_proba_stack = stack.predict_proba(X_test)[:, 1]

print("\nStacking Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_stack):.4f}")
print(f"ROC-AUC : {roc_auc_score(y_test, y_proba_stack):.4f}")
