import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/car_price.csv")

# Display first rows
df.head()

df.info()
df.describe()
df.isnull().sum()
df.fillna(df.mean(), inplace=True)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop("price", axis=1)
y = df["price"].values.reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]   # Add bias term
    
    theta = np.zeros((n + 1, 1))
    cost_history = []

    for _ in range(epochs):
        y_pred = X_b @ theta
        error = y_pred - y
        
        gradients = (2/m) * X_b.T @ error
        theta -= lr * gradients
        
        cost_history.append(mse(y, y_pred))
        
    return theta, cost_history

learning_rates = [0.001, 0.01, 0.1]
histories = {}

for lr in learning_rates:
    theta, history = gradient_descent(X_train_scaled, y_train, lr=lr, epochs=1000)
    histories[lr] = history

plt.figure()
for lr, history in histories.items():
    plt.plot(history, label=f"lr={lr}")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.legend()
plt.title("Gradient Descent Convergence")
plt.show()


theta, _ = gradient_descent(X_train_scaled, y_train, lr=0.01, epochs=1000)

# Predictions
X_test_b = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]
y_pred_test = X_test_b @ theta

from sklearn.metrics import r2_score

test_mse = mse(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print("Test MSE:", test_mse)
print("Test R²:", test_r2)

feature_name = "horsepower"
idx = X.columns.get_loc(feature_name)

plt.figure()
plt.scatter(X_test[feature_name], y_test, label="Actual")
plt.scatter(X_test[feature_name], y_pred_test, label="Predicted")
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.legend()
plt.title("Horsepower vs Car Price")
plt.show()

from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

sk_pred = lr_model.predict(X_test_scaled)

print("Sklearn MSE:", mse(y_test, sk_pred))
print("Sklearn R²:", r2_score(y_test, sk_pred))

