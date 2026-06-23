import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/heart_disease.csv")  # adjust path if needed

# Basic exploration
df.head()

df.info()

# Check missing values
df.isnull().sum()

df.fillna(df.median(numeric_only=True), inplace=True)

df.hist(figsize=(15,10))
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

categorical_features = ['cp', 'slope', 'thal']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

from sklearn.preprocessing import StandardScaler

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    solver='liblinear',
    penalty='l2',
    max_iter=1000
)

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": log_reg.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

feature_importance

from sklearn.feature_selection import RFE

rfe = RFE(log_reg, n_features_to_select=10)
rfe.fit(X_train, y_train)

selected_features = X.columns[rfe.support_]
selected_features

X_train_rfe = X_train[:, rfe.support_]
X_test_rfe = X_test[:, rfe.support_]

log_reg_rfe = LogisticRegression(solver='liblinear')
log_reg_rfe.fit(X_train_rfe, y_train)

y_pred_rfe = log_reg_rfe.predict(X_test_rfe)
print("Accuracy after RFE:", accuracy_score(y_test, y_pred_rfe))

new_patient = np.array([[
    55, 1, 140, 240, 0, 1, 150, 0, 1.0, 0, 1,
    0, 1, 0, 1, 0   # example encoded categorical values
]])

new_patient_scaled = scaler.transform(new_patient)
prediction = log_reg.predict(new_patient_scaled)
prediction_prob = log_reg.predict_proba(new_patient_scaled)[0][1]

print("Prediction:", prediction)
print("Probability of Heart Disease:", prediction_prob)

