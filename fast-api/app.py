from typing import List

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from unionml import Dataset, Model


dataset = Dataset(name="digits_dataset", test_size=0.2, shuffle=True, targets=["target"])
model = Model(name="digits_classifier", init=LogisticRegression, dataset=dataset)


@dataset.reader
def reader(sample_frac: float = 1.0, random_state: int = 12345) -> pd.DataFrame:
    data = load_digits(as_frame=True).frame
    return data.sample(frac=sample_frac, random_state=random_state)


@model.trainer
def trainer(estimator: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> LogisticRegression:
    return estimator.fit(features, target.squeeze())


@model.predictor
def predictor(estimator: LogisticRegression, features: pd.DataFrame) -> List[float]:
    return [float(x) for x in estimator.predict(features)]


@model.evaluator
def evaluator(estimator: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> float:
    return float(accuracy_score(target.squeeze(), predictor(estimator, features)))


if __name__ == "__main__":
    model_object, metrics = model.train(
        hyperparameters={"C": 1.0, "max_iter": 10000},
        sample_frac=1.0,
        random_state=12345,
    )

    predictions = model.predict(
        features=load_digits(as_frame=True).frame.sample(5, random_state=42)
    )

    print(f"model object: {model_object}")
    print(f"training metrics: {metrics}")
    print(f"predictions: {predictions}")

    # save model to a file, using joblib as the default serialization format
    model.save("/tmp/model_object.joblib")


from fastapi import FastAPI

app = FastAPI()

model.serve(app)

#unionml serve app:app --model-path /tmp/model_object.joblib --reload