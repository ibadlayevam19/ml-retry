# ============================================
# Decision Tree Classification on Iris Dataset
# ============================================

# 1. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

# Convert to DataFrame for easier exploration
df = pd.DataFrame(X, columns=feature_names)
df["species"] = y

# 3. Data Exploration
print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Pairplot
sns.pairplot(df, hue="species")
plt.show()

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Model Building
dt = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

dt.fit(X_train, y_train)

# 6. Model Evaluation
y_pred = dt.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# 7. Decision Tree Visualization
plt.figure(figsize=(14, 8))
plot_tree(
    dt,
    feature_names=feature_names,
    class_names=target_names,
    filled=True,
    rounded=True
)
plt.show()

# 8. Feature Importance
importances = pd.Series(dt.feature_importances_, index=feature_names)
print("\nFeature Importances:\n", importances.sort_values(ascending=False))

importances.sort_values().plot(kind="barh", title="Feature Importance")
plt.show()

# 9. Hyperparameter Tuning (Optional Bonus)
param_grid = {
    "max_depth": [2, 3, 4, 5, None],
    "criterion": ["gini", "entropy"]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# 10. Final Model Evaluation with Best Params
best_model = grid.best_estimator_
best_pred = best_model.predict(X_test)

print("\nFinal Accuracy:", accuracy_score(y_test, best_pred))
