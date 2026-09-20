import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from .preprocessing import load_data, build_preprocessor
from .config import MODEL_PATH


df = load_data()

X = df.drop(columns=["Price"])

y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

preprocessor = build_preprocessor()

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

joblib.dump(model, MODEL_PATH)

print("Model saved successfully.")