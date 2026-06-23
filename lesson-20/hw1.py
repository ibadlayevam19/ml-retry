# Ensemble Learning & Voting Classifier (Wine Dataset)

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

# Load dataset
X, y = load_wine(return_X_y=True)

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Step 1 — Baseline Models
# -------------------------------

log_reg = LogisticRegression(max_iter=2000, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier(random_state=42)

models = {
    "Logistic Regression": log_reg,
    "KNN": knn,
    "Decision Tree": dt
}

print("Baseline Model Accuracies:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.4f}")

# -------------------------------
# Step 2 — Hard Voting Classifier
# -------------------------------

hard_voting = VotingClassifier(
    estimators=[
        ("lr", log_reg),
        ("knn", knn),
        ("dt", dt)
    ],
    voting="hard"
)

hard_voting.fit(X_train, y_train)
y_pred_hard = hard_voting.predict(X_test)
hard_acc = accuracy_score(y_test, y_pred_hard)

print("\nHard Voting Accuracy:")
print(f"{hard_acc:.4f}")

# -------------------------------
# Step 3 — Soft Voting Classifier
# -------------------------------

soft_voting = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=2000, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("dt", DecisionTreeClassifier(random_state=42))
    ],
    voting="soft"
)

soft_voting.fit(X_train, y_train)
y_pred_soft = soft_voting.predict(X_test)
soft_acc = accuracy_score(y_test, y_pred_soft)

print("\nSoft Voting Accuracy:")
print(f"{soft_acc:.4f}")
