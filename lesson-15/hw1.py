# ================================
# SVM on Digits Dataset - Full Homework Script
# ================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# ================================
# 1. Load Dataset
# ================================
digits = load_digits()
X = digits.data
y = digits.target

print("Data shape:", X.shape)
print("Target shape:", y.shape)

# Show sample images
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title(f"Label: {y[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# ================================
# 2. Data Exploration
# ================================
print("\nNumber of samples:", X.shape[0])
print("Number of features:", X.shape[1])

unique, counts = np.unique(y, return_counts=True)
print("\nClass distribution:")
for u, c in zip(unique, counts):
    print(f"Digit {u}: {c}")

# ================================
# 3. Train/Test Split + Scaling
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# 4. Linear SVM
# ================================
svm_linear = SVC(kernel="linear")
svm_linear.fit(X_train_scaled, y_train)

y_pred_linear = svm_linear.predict(X_test_scaled)

print("\n===== Linear Kernel Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

# ================================
# 5. Polynomial & RBF Kernels
# ================================
svm_poly = SVC(kernel="poly", degree=3)
svm_rbf = SVC(kernel="rbf")

svm_poly.fit(X_train_scaled, y_train)
svm_rbf.fit(X_train_scaled, y_train)

y_pred_poly = svm_poly.predict(X_test_scaled)
y_pred_rbf = svm_rbf.predict(X_test_scaled)

print("\n===== Polynomial Kernel Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred_poly))

print("\n===== RBF Kernel Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))

# ================================
# 6. Hyperparameter Tuning (GridSearch)
# ================================
param_grid = {
    "C": [0.1, 1, 10],
    "gamma": [0.001, 0.01, 0.1],
    "kernel": ["rbf", "poly"],
    "degree": [2, 3]
}

grid = GridSearchCV(SVC(), param_grid, cv=3, verbose=2, n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

print("\nBest parameters:", grid.best_params_)

y_best_pred = best_model.predict(X_test_scaled)

print("\n===== Best Model Results =====")
print("Accuracy:", accuracy_score(y_test, y_best_pred))
print(classification_report(y_test, y_best_pred))

# ================================
# 7. Confusion Matrix (Best Model)
# ================================
cm = confusion_matrix(y_test, y_best_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Best SVM Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ================================
# 8. PCA Visualization (2D)
# ================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap="tab10", s=15)
plt.legend(*scatter.legend_elements(), title="Digits")
plt.title("PCA Visualization of Digits Dataset (Training Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# ================================
# Bonus: Support Vectors Info
# ================================
print("\nNumber of support vectors:", best_model.n_support_)
