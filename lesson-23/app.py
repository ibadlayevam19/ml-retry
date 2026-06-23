# train_model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "iris_model.joblib")
print("Model saved as iris_model.joblib")

#all above code is inside of train_model.py file and below code is inside of app.py file

# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib
import numpy as np

app = FastAPI(title="Iris Prediction API")

# Load model at startup
model = joblib.load("iris_model.joblib")

# Pydantic model for input validation
class IrisFeatures(BaseModel):
    features: conlist(float, min_items=4, max_items=4)  # 4 features for iris

@app.post("/predict")
def predict(iris: IrisFeatures):
    try:
        X = np.array([iris.features])
        prediction = model.predict(X)
        return {"prediction": int(prediction[0])}  # convert numpy int to Python int
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
