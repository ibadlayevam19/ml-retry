# ===============================
# Dimensionality Reduction Lab
# Breast Cancer Dataset
# ===============================

import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -------------------------------
# 1. Load dataset
# -------------------------------
data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 2. Logistic Regression (no scaling)
# -------------------------------
lr = LogisticRegression(max_iter=5000)

start = time.time()
lr.fit(X_train, y_train)
time_original = time.time() - start

y_pred = lr.predict(X_test)
acc_original = accuracy_score(y_test, y_pred)

print("\nLogistic Regression (No Scaling)")
print("Accuracy:", acc_original)
print("Training time:", time_original)

# -------------------------------
# 3. Logistic Regression (with scaling)
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_scaled = LogisticRegression(max_iter=5000)

start = time.time()
lr_scaled.fit(X_train_scaled, y_train)
time_scaled = time.time() - start

y_pred_scaled = lr_scaled.predict(X_test_scaled)
acc_scaled = accuracy_score(y_test, y_pred_scaled)

print("\nLogistic Regression (With Scaling)")
print("Accuracy:", acc_scaled)
print("Training time:", time_scaled)

# -------------------------------
# 4. PCA with 2 components
# -------------------------------
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_train_scaled)

# Plot original feature space (first 2 features)
plt.figure()
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap="coolwarm")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Original Feature Space")
plt.show()

# Plot PCA space
plt.figure()
plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=y_train, cmap="coolwarm")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA (2 Components)")
plt.show()

# -------------------------------
# 5. Explained variance analysis
# -------------------------------
pca_full = PCA(n_components=None)
pca_full.fit(X_train_scaled)

explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

# Explained variance plot
plt.figure()
plt.plot(explained_var, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance per Component")
plt.show()

# Cumulative variance plot
plt.figure()
plt.plot(cumulative_var, marker="o")
plt.axhline(y=0.9, color="r", linestyle="--")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.show()

# Components for 90% variance
n_components_90 = np.argmax(cumulative_var >= 0.9) + 1
print("\nComponents needed for 90% variance:", n_components_90)

# -------------------------------
# 6. Logistic Regression with PCA (~90%)
# -------------------------------
pca_90 = PCA(n_components=n_components_90)

X_train_pca = pca_90.fit_transform(X_train_scaled)
X_test_pca = pca_90.transform(X_test_scaled)

lr_pca = LogisticRegression(max_iter=5000)

start = time.time()
lr_pca.fit(X_train_pca, y_train)
time_pca = time.time() - start

y_pred_pca = lr_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print("\nLogistic Regression (PCA ~90%)")
print("Accuracy:", acc_pca)
print("Training time:", time_pca)

# -------------------------------
# 7. t-SNE visualization
# -------------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_train_scaled)

plt.figure()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, cmap="coolwarm")
plt.title("t-SNE Visualization")
plt.show()

# -------------------------------
# Final Comparison Summary
# -------------------------------
print("\n===== Summary =====")
print("Original Accuracy:", acc_original)
print("Scaled Accuracy:", acc_scaled)
print("PCA Accuracy:", acc_pca)

print("\nOriginal Time:", time_original)
print("Scaled Time:", time_scaled)
print("PCA Time:", time_pca)
