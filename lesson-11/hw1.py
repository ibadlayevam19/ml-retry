# Cross Validation: Model Selection with the Wine Dataset
# One complete copy-paste runnable script

import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    cross_validate
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===============================
# 1. Data Loading and Exploration
# ===============================

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="target")

print("First 5 rows:")
print(X.head(), "\n")

print("Dataset shape:", X.shape)
print("\nSummary statistics:")
print(X.describe(), "\n")

print("Class distribution:")
print(y.value_counts(), "\n")

print("Discussion:")
print("- The dataset is slightly imbalanced but not severely.")
print("- Some features (e.g., proline, color_intensity) show large ranges → possible outliers.\n")

# ==================================
# 2. Baseline Model (Train-Test Split)
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline_model = LogisticRegression(max_iter=500)
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred)

print("Baseline Logistic Regression Test Accuracy:", baseline_accuracy, "\n")

# ==========================
# 3. K-Fold Cross Validation
# ==========================

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

kf_scores = cross_val_score(
    LogisticRegression(max_iter=500),
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("K-Fold CV Scores:", kf_scores)
print("Mean CV Accuracy:", kf_scores.mean())
print("Std CV Accuracy:", kf_scores.std(), "\n")

print("Interpretation:")
print("- CV mean is more reliable because it uses multiple train/test splits.")
print("- Standard deviation shows stability; lower std means more consistent performance.\n")

# =================================
# 4. Stratified K-Fold Cross Validation
# =================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_reg = LogisticRegression(max_iter=500)
rf = RandomForestClassifier(random_state=42)

log_scores = cross_val_score(log_reg, X, y, cv=skf, scoring="accuracy")
rf_scores = cross_val_score(rf, X, y, cv=skf, scoring="accuracy")

print("Stratified CV - Logistic Regression:")
print("Mean:", log_scores.mean(), "Std:", log_scores.std())

print("\nStratified CV - Random Forest:")
print("Mean:", rf_scores.mean(), "Std:", rf_scores.std(), "\n")

print("Analysis:")
print("- Random Forest is usually more accurate.")
print("- Logistic Regression often shows lower variance (more stable).")
print("- Added complexity of RF should be justified by accuracy gain.\n")

# =====================================
# 5. Cross Validation with Multiple Metrics
# =====================================

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

log_cv = cross_validate(log_reg, X, y, cv=skf, scoring=scoring)
rf_cv = cross_validate(rf, X, y, cv=skf, scoring=scoring)

print("Logistic Regression - Multiple Metrics:")
for metric in scoring:
    print(
        f"{metric}: mean={log_cv['test_' + metric].mean():.4f}, "
        f"std={log_cv['test_' + metric].std():.4f}"
    )

print("\nRandom Forest - Multiple Metrics:")
for metric in scoring:
    print(
        f"{metric}: mean={rf_cv['test_' + metric].mean():.4f}, "
        f"std={rf_cv['test_' + metric].std():.4f}"
    )

print("\nObservations:")
print("- Random Forest generally performs better across all metrics.")
print("- F1 and recall often highlight bigger differences in multiclass problems.\n")

# =========================================
# 6. Hyperparameter Tuning with GridSearchCV
# =========================================

params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 4, 6]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    params,
    scoring="accuracy",
    cv=skf,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best CV Accuracy:", grid.best_score_)
print("Best Hyperparameters:", grid.best_params_)
print("Best Estimator:", grid.best_estimator_, "\n")

best_model = grid.best_estimator_
best_model.fit(X_train, y_train)

best_test_accuracy = accuracy_score(y_test, best_model.predict(X_test))

print("Tuned Random Forest Test Accuracy:", best_test_accuracy)
print("Baseline Logistic Regression Test Accuracy:", baseline_accuracy)
