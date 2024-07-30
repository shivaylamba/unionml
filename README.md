# Digits Classifier FastAPI Application

This project demonstrates a machine learning model for classifying digits using scikit-learn and FastAPI. It includes both the model training script and a FastAPI server for serving predictions.

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Installation

1. Clone the repository:git clone <repository-url> cd <repository-directory>


2. Install the required packages:pip install unionml fastapi sklearn pandas requests


pip install unionml fastapi sklearn pandas requests

Copy

## Running the Application

1. Train the model:
python app.py

Copy
This will train the model and save it to `/tmp/model_object.joblib`.

2. Start the FastAPI server:
unionml serve app:app --model-path /tmp/model_object.joblib --reload

Copy
The server will start running on `http://127.0.0.1:8000`.

3. In a separate terminal, you can run the client to test predictions:
python client.py

Copy

## API Endpoints

- `/predict`: POST request to get predictions for input features

## File Descriptions

- `app.py`: Contains the model training code and FastAPI app setup
- `client.py`: A sample client script to test the API

## Notes

- The model is trained on the digits dataset from scikit-learn
- The API uses FastAPI and is served using unionml
- The client script sends a POST request to the `/predict` endpoint with sample data
