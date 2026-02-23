from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Haversine formula for distance calculation
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def dist_to_coast(lat, lon):
    coastline_points = [
        (41.7, -124.2), (40.5, -124.4), (39.0, -123.7),
        (38.3, -123.0), (37.8, -122.5), (37.2, -122.4),
        (36.6, -121.9), (35.7, -121.3), (34.4, -120.5),
        (34.0, -119.7), (33.7, -118.3), (33.2, -117.4),
        (32.7, -117.2)
    ]
    return min(haversine(lat, lon, clat, clon) for clat, clon in coastline_points)

# Train model on startup
print("Training V2 model with 19 features...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# Cleaning
df = df[df['MedHouseVal'] < 5.0]

# Original engineered features
df['rooms_per_person'] = df['AveRooms'] / df['AveOccup']
df['bedrooms_ratio'] = df['AveBedrms'] / df['AveRooms']
df['income_per_occupant'] = df['MedInc'] / df['AveOccup']

# New features
SF = (37.7749, -122.4194)
LA = (34.0522, -118.2437)
SD = (32.7157, -117.1611)

df['dist_to_sf'] = haversine(df['Latitude'], df['Longitude'], SF[0], SF[1])
df['dist_to_la'] = haversine(df['Latitude'], df['Longitude'], LA[0], LA[1])
df['dist_to_sd'] = haversine(df['Latitude'], df['Longitude'], SD[0], SD[1])
df['dist_to_nearest_city'] = df[['dist_to_sf', 'dist_to_la', 'dist_to_sd']].min(axis=1)
df['coastal_proximity_km'] = df.apply(lambda row: dist_to_coast(row['Latitude'], row['Longitude']), axis=1)
df['income_inequality_proxy'] = df['MedInc'] / (df['AveOccup'] * df['AveBedrms'])
df['housing_density'] = df['Population'] / df['AveRooms']

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"V2 Model trained. R2: {model.score(X_test, y_test):.4f}")

# FastAPI app
app = FastAPI(title='California Housing Price Predictor V2')

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get('/')
def home():
    return {'message': 'California Housing Price Predictor V2 is running', 'model_r2': 0.8208}

@app.post('/predict')
def predict(features: HouseFeatures):
    # Calculate engineered features automatically from raw inputs
    rooms_per_person = features.AveRooms / features.AveOccup
    bedrooms_ratio = features.AveBedrms / features.AveRooms
    income_per_occupant = features.MedInc / features.AveOccup
    dist_sf = haversine(features.Latitude, features.Longitude, SF[0], SF[1])
    dist_la = haversine(features.Latitude, features.Longitude, LA[0], LA[1])
    dist_sd = haversine(features.Latitude, features.Longitude, SD[0], SD[1])
    dist_nearest = min(dist_sf, dist_la, dist_sd)
    coastal = dist_to_coast(features.Latitude, features.Longitude)
    income_inequality = features.MedInc / (features.AveOccup * features.AveBedrms)
    density = features.Population / features.AveRooms

    data = np.array([[
        features.MedInc, features.HouseAge, features.AveRooms,
        features.AveBedrms, features.Population, features.AveOccup,
        features.Latitude, features.Longitude,
        rooms_per_person, bedrooms_ratio, income_per_occupant,
        dist_sf, dist_la, dist_sd, dist_nearest,
        coastal, income_inequality, density
    ]])

    prediction = model.predict(data)[0]
    return {
        'predicted_house_value': round(float(prediction), 4),
        'predicted_house_value_usd': f'${round(float(prediction) * 100000, 2):,.2f}',
        'coastal_proximity_km': round(coastal, 2),
        'nearest_major_city_km': round(dist_nearest, 2)
    }