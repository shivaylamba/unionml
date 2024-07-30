import requests
import json

# The URL where your FastAPI app is running
BASE_URL = "http://localhost:8000"

def get_welcome_message():
    response = requests.get(f"{BASE_URL}/")
    print("Welcome Message:")
    print(response.json())
    print()

def get_model_info():
    response = requests.get(f"{BASE_URL}/model-info")
    print("Model Info:")
    print(json.dumps(response.json(), indent=2))
    print()

def make_prediction(sepal_length, sepal_width, petal_length, petal_width):
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("Prediction:")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    get_welcome_message()
    get_model_info()
    
    # Example predictions
    print("Example 1:")
    make_prediction(5.1, 3.5, 1.4, 0.2)
    
    print("Example 2:")
    make_prediction(6.3, 3.3, 6.0, 2.5)
    
    print("Example 3:")
    make_prediction(4.9, 2.4, 3.3, 1.0)