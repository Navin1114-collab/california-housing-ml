from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create the FastAPI app
app = FastAPI(title='California Housing Price Predictor')

# Define what input data looks like
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    rooms_per_person: float
    bedrooms_ratio: float
    income_per_occupant: float

# Root endpoint — just to confirm API is alive
@app.get('/')
def home():
    return {'message': 'California Housing Price Predictor is running'}

# Prediction endpoint
@app.post('/predict')
def predict(features: HouseFeatures):
    data = np.array([[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude,
        features.rooms_per_person,
        features.bedrooms_ratio,
        features.income_per_occupant
    ]])
    prediction = model.predict(data)[0]
    return {
        'predicted_house_value': round(float(prediction), 4),
        'predicted_house_value_usd': f'${round(float(prediction) * 100000, 2):,.2f}'
    }