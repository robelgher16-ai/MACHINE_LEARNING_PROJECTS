import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .config import DATA_PATH


def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.drop(columns=["Id"])

    return df


def build_preprocessor():
    numerical_features = [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Floors",
        "YearBuilt"
    ]

    categorical_features = [
        "Location",
        "Condition",
        "Garage"
    ]

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor