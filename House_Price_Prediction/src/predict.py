import joblib
import pandas as pd

from .config import MODEL_PATH

# Load trained model
model = joblib.load(MODEL_PATH)
def predict_price(
    area,
    bedrooms,
    bathrooms,
    floors,
    year_built,
    location,
    condition,
    garage
):
    house = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "YearBuilt": year_built,
        "Location": location,
        "Condition": condition,
        "Garage": garage
    }])

    prediction = model.predict(house)

    return float(prediction[0])

if __name__ == "__main__":

    price = predict_price(
        area=2500,
        bedrooms=4,
        bathrooms=3,
        floors=2,
        year_built=2018,
        location="Urban",
        condition="Good",
        garage="Yes"
    )

    print(f"Predicted Price: ${price:,.2f}")
