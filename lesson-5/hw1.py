import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes

data = load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("First 5 rows:")
X.head()

print("\nShape:", X.shape)

print("\nFeature names:")
print(X.columns.tolist())

print("\nDataset statistics:")
X.describe()

from sklearn.preprocessing import PolynomialFeatures

poly_features = ["bmi", "bp", "s5"]
X_poly_part = X[poly_features]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_expanded = poly.fit_transform(X_poly_part)

poly_feature_names = poly.get_feature_names_out(poly_features)

X_poly_df = pd.DataFrame(X_poly_expanded, columns=poly_feature_names)

remaining_features = X.drop(columns=poly_features)

X_final = pd.concat([remaining_features, X_poly_df], axis=1)

print("Number of polynomial features:", len(poly_feature_names))
print("\nExpanded feature names:")
print(poly_feature_names)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train_base, X_test_base, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_poly, X_test_poly, _, _ = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

lin_reg = LinearRegression()
lin_reg.fit(X_train_base, y_train)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

coef_df = pd.DataFrame({
    "Feature": X_train_poly.columns,
    "Coefficient": poly_reg.coef_
})

coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()

top_10 = coef_df.sort_values(by="Abs_Coefficient", ascending=False).head(10)
top_10

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred_base = lin_reg.predict(X_test_base)
y_pred_poly = poly_reg.predict(X_test_poly)

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

base_metrics = evaluate(y_test, y_pred_base)
poly_metrics = evaluate(y_test, y_pred_poly)

print("Baseline Model:")
print("MAE:", base_metrics[0])
print("RMSE:", base_metrics[1])
print("R²:", base_metrics[2])

print("\nPolynomial Model:")
print("MAE:", poly_metrics[0])
print("RMSE:", poly_metrics[1])
print("R²:", poly_metrics[2])

plt.figure(figsize=(12,5))

# Baseline
plt.subplot(1,2,1)
plt.scatter(y_test, y_pred_base, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Baseline Model")

# Polynomial
plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_poly, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial Model")

plt.tight_layout()
plt.show()

labels = ["MAE", "RMSE"]
baseline_vals = base_metrics[:2]
poly_vals = poly_metrics[:2]

x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, baseline_vals, width, label="Baseline")
plt.bar(x + width/2, poly_vals, width, label="Polynomial")

plt.xticks(x, labels)
plt.ylabel("Error")
plt.title("Error Comparison")
plt.legend()
plt.show()

# Single feature
X_bmi = X[["bmi"]]
y_target = y

poly_bmi = PolynomialFeatures(degree=2, include_bias=False)
X_bmi_poly = poly_bmi.fit_transform(X_bmi)

model_bmi = LinearRegression()
model_bmi.fit(X_bmi_poly, y_target)

# Smooth curve
bmi_sorted_idx = np.argsort(X_bmi["bmi"])
bmi_sorted = X_bmi.iloc[bmi_sorted_idx]
bmi_poly_sorted = poly_bmi.transform(bmi_sorted)

y_curve = model_bmi.predict(bmi_poly_sorted)

plt.scatter(X_bmi, y_target, alpha=0.5)
plt.plot(bmi_sorted, y_curve)
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Polynomial Relationship: BMI vs Target")
plt.show()

sample = pd.DataFrame([{
    "age": 0.05,
    "sex": 0.02,
    "bmi": 0.04,
    "bp": 0.03,
    "s1": -0.02,
    "s2": -0.01,
    "s3": 0.00,
    "s4": 0.02,
    "s5": 0.03,
    "s6": 0.01
}])

baseline_prediction = lin_reg.predict(sample)
print("Baseline Prediction:", baseline_prediction[0])

sample_poly_part = sample[poly_features]
sample_poly_expanded = poly.transform(sample_poly_part)
sample_poly_df = pd.DataFrame(sample_poly_expanded, columns=poly_feature_names)

sample_remaining = sample.drop(columns=poly_features)
sample_final = pd.concat([sample_remaining, sample_poly_df], axis=1)

poly_prediction = poly_reg.predict(sample_final)
print("Polynomial Prediction:", poly_prediction[0])
