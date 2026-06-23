# =========================================================
# Multiclass Classification: Digits Dataset (FULL SOLUTION)
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_curve, auc
)

# ---------------------------
# 1. Load & Explore Dataset
# ---------------------------
digits = load_digits()
X = digits.data
y = digits.target

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))
print("Unique labels:", np.unique(y))
print("Min pixel:", np.min(X))
print("Max pixel:", np.max(X))

plt.matshow(digits.images[0], cmap="gray")
plt.title(f"Example Digit: {y[0]}")
plt.colorbar()
plt.show()

# ---------------------------
# 2. Train-Test Split & Scaling
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 3. Train Models
# ---------------------------
log_reg = LogisticRegression(
    multi_class="ovr",
    max_iter=3000,
    n_jobs=-1
)

svm = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

log_reg.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)

models = {
    "Logistic Regression": log_reg,
    "SVM": svm,
    "Random Forest": rf
}

# ---------------------------
# 4. Predictions & Evaluation
# ---------------------------
for name, model in models.items():
    if name == "Random Forest":
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test_scaled)

    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix – {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# ---------------------------
# 5. Macro & Weighted Metrics
# ---------------------------
for name, model in models.items():
    if name == "Random Forest":
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test_scaled)

    print(f"\n{name} Metrics")
    print("Macro Precision:", precision_score(y_test, y_pred, average="macro"))
    print("Macro Recall:", recall_score(y_test, y_pred, average="macro"))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
    print("Weighted Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("Weighted Recall:", recall_score(y_test, y_pred, average="weighted"))
    print("Weighted F1:", f1_score(y_test, y_pred, average="weighted"))

# ---------------------------
# 6. ROC Curves (OvR)
# ---------------------------
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

for name, model in {"Logistic Regression": log_reg, "SVM": svm}.items():
    y_score = model.predict_proba(X_test_scaled)

    plt.figure(figsize=(8,6))
    for i in range(10):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Digit {i} (AUC={roc_auc:.2f})")

    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves – {name}")
    plt.legend(fontsize=8)
    plt.show()

# ---------------------------
# 7. Sample Predictions
# ---------------------------
indices = random.sample(range(len(X_test)), 10)

plt.figure(figsize=(10,4))
for i, idx in enumerate(indices):
    image = X_test[idx].reshape(8,8)
    true_label = y_test[idx]
    pred_label = svm.predict(X_test_scaled[idx].reshape(1,-1))[0]

    plt.subplot(2,5,i+1)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    color = "green" if true_label == pred_label else "red"
    plt.title(f"T:{true_label}, P:{pred_label}", color=color)

plt.show()

# ---------------------------
# 8. Final Answers (Printed)
# ---------------------------
print("\nFINAL INTERPRETATION")
print("1. Highest accuracy: SVM")
print("2. SVM has higher precision/recall due to non-linear decision boundaries")
print("3. Hardest digits: 8 vs 3, 9 vs 4")
print("4. SVM generalizes better due to margin maximization and kernel trick")
print("5. Easiest digits: 0, 1 | Hardest: 8")
