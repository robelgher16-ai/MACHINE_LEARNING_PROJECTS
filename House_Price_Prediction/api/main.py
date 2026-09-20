from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_price


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using a trained machine learning model.",
    version="1.0.0"
)


# ============================================================
# INPUT SCHEMA
# ============================================================

class HouseInput(BaseModel):
    area: int = Field(..., ge=501, le=4999)
    bedrooms: int = Field(..., ge=1, le=5)
    bathrooms: int = Field(..., ge=1, le=4)
    floors: int = Field(..., ge=1, le=3)
    year_built: int = Field(..., ge=1900, le=2023)

    location: Literal[
        "Downtown",
        "Urban",
        "Suburban",
        "Rural"
    ]

    condition: Literal[
        "Excellent",
        "Good",
        "Fair",
        "Poor"
    ]

    garage: Literal[
        "Yes",
        "No"
    ]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_house_price(data: HouseInput):

    price = predict_price(
        area=data.area,
        bedrooms=data.bedrooms,
        bathrooms=data.bathrooms,
        floors=data.floors,
        year_built=data.year_built,
        location=data.location,
        condition=data.condition,
        garage=data.garage
    )

    return {
        "predicted_price": round(price, 2)
    }