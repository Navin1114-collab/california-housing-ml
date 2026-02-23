# 🏠 California Housing Price Predictor — Live API

> XGBoost ML model predicting California house prices, deployed as a live REST API on Render.com

[![Live API](https://img.shields.io/badge/Live%20API-00C851?style=for-the-badge&logo=render&logoColor=white)](https://california-housing-ml-ab74.onrender.com)
[![API Docs](https://img.shields.io/badge/API%20Docs-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://california-housing-ml-ab74.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)]()

---

## 🎯 What This Project Does

This project takes raw California housing data, engineers 8 custom features, trains an XGBoost model achieving **R² = 0.8208**, and serves predictions through a live REST API that anyone can call with a URL.

You send 8 inputs → the API calculates 19 features automatically → returns predicted house price in dollars.

**Try it live:**
```
POST https://california-housing-ml-ab74.onrender.com/predict
```

---

## 📊 Results

| Model | R² Score | Notes |
|---|---|---|
| Linear Regression | 0.6387 | Baseline |
| Ridge Regression | 0.6387 | No overfitting detected |
| Random Forest | 0.7939 | Big jump from linear |
| **XGBoost V2** | **0.8208** | **Winner — deployed** |

---

## 🔧 Feature Engineering

I engineered 8 custom features on top of the original 8 raw columns:

| Feature | Formula | Why It Matters |
|---|---|---|
| `coastal_proximity_km` | Haversine distance to coastline | **#1 predictor** — beach premium |
| `dist_to_nearest_city` | Min distance to SF/LA/SD | Urban vs rural pricing |
| `dist_to_sf` | Haversine to San Francisco | Tech hub premium |
| `dist_to_la` | Haversine to Los Angeles | Metro premium |
| `dist_to_sd` | Haversine to San Diego | Southern CA market |
| `rooms_per_person` | AveRooms / AveOccup | Real spaciousness signal |
| `bedrooms_ratio` | AveBedrms / AveRooms | House type signal |
| `income_per_occupant` | MedInc / AveOccup | Wealth per person |
| `income_inequality_proxy` | MedInc / (AveOccup × AveBedrms) | Overcrowding signal |
| `housing_density` | Population / AveRooms | Urban density signal |

**Key insight:** `coastal_proximity_km` — a feature that doesn't exist in the original dataset — became the single most important predictor at 0.54 importance score, beating raw income and location columns.

---

## 🗺️ K-Means Clustering

K-Means (k=4) segmented California into 4 distinct housing market types:

| Cluster | Area | Avg Income | Avg House Value |
|---|---|---|---|
| 0 | SF Bay Area | $50,600 | $290,000 |
| 1 | Inland Southern CA | $29,500 | $152,000 |
| 2 | LA Coastline | $53,700 | $300,000 |
| 3 | Northern Inland CA | $27,800 | $120,000 |

---

## 🚀 API Usage

**Base URL:** `https://california-housing-ml-ab74.onrender.com`

### Endpoints

`GET /` — Health check
```json
{"message": "California Housing Price Predictor V2 is running", "model_r2": 0.8208}
```

`POST /predict` — Predict house price

**Input (8 fields only):**
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.984127,
  "AveBedrms": 1.023810,
  "Population": 322.0,
  "AveOccup": 2.555556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

**Output:**
```json
{
  "predicted_house_value": 3.6867,
  "predicted_house_value_usd": "$368,671.18",
  "coastal_proximity_km": 25.32,
  "nearest_major_city_km": 20.33
}
```

---

## 🏗️ Project Structure
```
california-housing-api/
├── main.py                      # FastAPI app + model training on startup
├── EDA.ipynb                    # Exploratory data analysis + 4 Plotly charts
├── feature_engineering_v2.ipynb # Feature engineering + model training
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## ⚙️ Tech Stack

- **Python** — core language
- **Pandas** — data manipulation
- **Scikit-learn** — model training, cross-validation
- **XGBoost** — gradient boosting model
- **FastAPI** — REST API framework
- **Plotly** — interactive visualisations
- **Render.com** — free cloud deployment

---

## 🏃 Run Locally
```bash
git clone https://github.com/Navin1114-collab/california-housing-ml.git
cd california-housing-ml
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open: `http://127.0.0.1:8000/docs`

---

## 👤 Author

**Navin Kumar Nagisetty**
📧 navinnagisetty@gmail.com
💼 [LinkedIn](https://www.linkedin.com/in/navinnagisetty/)
🐙 [GitHub](https://github.com/Navin1114-collab)