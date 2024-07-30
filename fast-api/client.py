import requests
from sklearn.datasets import load_digits

# generate predictions
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": load_digits(as_frame=True).frame.sample(5, random_state=42).to_dict(orient="records")},
)

# Check if the request was successful
if response.status_code == 200:
    print("Prediction successful!")
    print("Predictions:", response.json())
else:
    print(f"Error: {response.status_code}")
    print("Response:", response.text)