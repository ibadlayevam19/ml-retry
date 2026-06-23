from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train_scaled, y_train)

y_pred_baseline = baseline_model.predict(X_test_scaled)
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print("Baseline Model Accuracy:", baseline_accuracy)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred_pipeline = pipeline.predict(X_test)
pipeline_accuracy = accuracy_score(y_test, y_pred_pipeline)

print("Pipeline Model Accuracy:", pipeline_accuracy)


print("\nComparison:")
print("Accuracy difference:", pipeline_accuracy - baseline_accuracy)
print("\nWhy pipeline is preferred:")
print("- Prevents data leakage")
print("- Ensures correct preprocessing order")
print("- Cleaner and more maintainable code")
print("- Works seamlessly with cross-validation and grid search")
