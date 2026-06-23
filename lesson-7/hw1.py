import pandas as pd

# Load dataset
df = pd.read_csv("data/california_housing.csv")

# Display first rows
df.head()

df.info()
df.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

mse_lr, r2_lr

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import numpy as np

alphas = np.logspace(-3, 3, 20)

ridge = Ridge()
ridge_cv = GridSearchCV(
    ridge,
    {"alpha": alphas},
    cv=5,
    scoring="r2"
)

ridge_cv.fit(X_train_scaled, y_train)
ridge_cv.best_params_

ridge_best = ridge_cv.best_estimator_
y_pred_ridge = ridge_best.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

mse_ridge, r2_ridge

coefs = []

for a in alphas:
    r = Ridge(alpha=a)
    r.fit(X_train_scaled, y_train)
    coefs.append(r.coef_)

plt.figure(figsize=(10,6))
plt.plot(alphas, coefs)
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficients")
plt.title("Ridge Coefficients vs Alpha")
plt.show()

from sklearn.linear_model import Lasso

lasso = Lasso(max_iter=10000)
lasso_cv = GridSearchCV(
    lasso,
    {"alpha": alphas},
    cv=5,
    scoring="r2"
)

lasso_cv.fit(X_train_scaled, y_train)
lasso_cv.best_params_

lasso_best = lasso_cv.best_estimator_
y_pred_lasso = lasso_best.predict(X_test_scaled)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

mse_lasso, r2_lasso

coefs_lasso = []

for a in alphas:
    l = Lasso(alpha=a, max_iter=10000)
    l.fit(X_train_scaled, y_train)
    coefs_lasso.append(l.coef_)

plt.figure(figsize=(10,6))
plt.plot(alphas, coefs_lasso)
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficients")
plt.title("Lasso Coefficients vs Alpha")
plt.show()

results = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression"],
    "MSE": [mse_lr, mse_ridge, mse_lasso],
    "R²": [r2_lr, r2_ridge, r2_lasso]
})

results

non_zero = sum(lasso_best.coef_ != 0)
non_zero

lasso_features = pd.Series(
    lasso_best.coef_,
    index=X.columns
)

lasso_features[lasso_features != 0].sort_values(key=abs, ascending=False)

