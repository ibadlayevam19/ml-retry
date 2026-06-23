# ================================
# KNN Customer Churn Classification
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------
# 1. Load Dataset
# -----------------------------
# Change this path to your downloaded CSV file
df = pd.read_csv("Telco-Customer-Churn.csv")

print("\nFirst 5 rows:")
print(df.head())

# -----------------------------
# 2. Data Exploration
# -----------------------------
print("\nDataset Info:")
print(df.info())

print("\nChurn distribution:")
print(df["Churn"].value_counts())

# Convert TotalCharges to numeric (some values are spaces)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Plot churn distribution
plt.figure()
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

# Correlation heatmap for numeric features
plt.figure(figsize=(6, 4))
sns.heatmap(df[["tenure", "MonthlyCharges", "TotalCharges"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Groupby example
print("\nChurn rate by Contract type:")
print(df.groupby("Contract")["Churn"].value_counts(normalize=True))

# -----------------------------
# 3. Data Cleaning & Encoding
# -----------------------------
df = df.dropna()

# Convert target to binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop customerID
df = df.drop("customerID", axis=1)

X = df.drop("Churn", axis=1)
y = df["Churn"]

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", pd.get_dummies(X[categorical_cols], drop_first=True).columns, [])
    ],
    remainder="drop"
)

# Manual one-hot encoding for categorical features
X = pd.get_dummies(X, drop_first=True)

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Train Default KNN Model
# -----------------------------
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

print("\nDefault KNN Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 6. Optimize K Value
# -----------------------------
k_values = [1, 3, 5, 7, 10, 15, 20]
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)

plt.figure()
plt.plot(k_values, accuracies, marker="o")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy")
plt.show()

best_k = k_values[np.argmax(accuracies)]
print("Best K:", best_k)

# Cross-validation for best k
cv_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=best_k),
    X_train_scaled, y_train, cv=5
)
print("Cross-validation accuracy:", cv_scores.mean())

# -----------------------------
# 7. Distance Metrics Comparison
# -----------------------------
metrics = ["euclidean", "manhattan", "minkowski"]

print("\nDistance Metric Comparison:")

for metric in metrics:
    model = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"{metric.capitalize()} Accuracy: {acc:.4f}")

# -----------------------------
# 8. Final Notes
# -----------------------------
print("\nProject completed successfully!")
