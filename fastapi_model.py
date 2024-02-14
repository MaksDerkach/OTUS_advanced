from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from joblib import load
from pydantic import BaseModel, Field
from typing import Sequence
import pandas as pd

class PredictSample(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

class PredictVector(BaseModel):
    vector: list



app = FastAPI()

def load_model():
    model = load("HD_model.joblib")
    names = model.feature_names_in_
    return names, model


@app.get('/', response_class=HTMLResponse)
def index():
    return '''
    <html>
        <head>
            <title>Heart Disease</title>
        </head>
        <body>
            <h1>Hello, this is HeartDisease model!<h1>
        </body>
    </html>'''


@app.get("/feature_names")
def feature_names():
    return {'feature_names': list(col_names)}


@app.post("/predict")
def predict(sample: PredictVector):
    prediction = model.predict([sample.vector]).tolist()[0]
    return {'prediction': prediction}


@app.post("/predict_by_names")
def predict(sample: PredictSample):
    posted_data = {key: [value] for key, value in sample.model_dump().items()}
    prediction = model.predict(pd.DataFrame(posted_data)[col_names])
    return {'prediction': prediction.tolist()[0]}


col_names, model = load_model()

