from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Initialize FastAPI app
app = FastAPI(title="Iris Classification API")

# Define input data model
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load and prepare the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "iris_model.joblib")

# Calculate accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

# Store iris target names
iris_target_names = iris.target_names.tolist()

# 

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classification API"}

@app.get("/model-info")
def model_info():
    return {
        "model_type": "K-Nearest Neighbors",
        "accuracy": accuracy,
        "iris_types": iris_target_names
    }

@app.post("/predict")
def predict(iris: IrisFeatures):
    try:
        # Load the model (in a real-world scenario, you'd want to load this once and reuse)
        model = joblib.load("iris_model.joblib")
        
        # Make prediction
        features = np.array([[
            iris.sepal_length,
            iris.sepal_width,
            iris.petal_length,
            iris.petal_width
        ]])
        prediction = model.predict(features)
        
        # Get the predicted iris type
        iris_type = iris_target_names[prediction[0]]
        
        return {
            "predicted_class": int(prediction[0]),
            "iris_type": iris_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)