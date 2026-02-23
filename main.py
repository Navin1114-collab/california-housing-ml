from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Train the model on startup
print("Training model...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# Feature engineering
df = df[df['MedHouseVal'] < 5.0]
df['rooms_per_person'] = df['AveRooms'] / df['AveOccup']
df['bedrooms_ratio'] = df['AveBedrms'] / df['AveRooms']
df['income_per_occupant'] = df['MedInc'] / df['AveOccup']

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained. R2:", round(model.score(X_test, y_test), 4))

# Create the FastAPI app
app = FastAPI(title='California Housing Price Predictor')

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

@app.get('/')
def home():
    return {'message': 'California Housing Price Predictor is running'}

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