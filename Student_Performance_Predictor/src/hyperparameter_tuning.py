# ==========================================
# File: src/hyperparameter_tuning.py
# Purpose: Optimize Random Forest
# ==========================================

from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data_cleaning import clean_dataset
from src.preprocessing import create_preprocessor

warnings.filterwarnings("ignore")

# ------------------------------------------
# Configuration
# ------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ------------------------------------------
# Paths
# ------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT /
    "data/raw/Students_Performance.csv"
)

MODEL_PATH = (
    PROJECT_ROOT /
    "models/best_random_forest_tuned.pkl"
)

REPORT_PATH = (
    PROJECT_ROOT /
    "reports/random_forest_tuning.json"
)

# ------------------------------------------
# Load Data
# ------------------------------------------

print("=" * 70)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df = clean_dataset(df)

X = df.drop(columns=["Student_ID", "Grade"])
y = df["Grade"].astype(int)

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

preprocessor = create_preprocessor(
    numerical_features,
    categorical_features
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        random_state=RANDOM_STATE
    ))
])

# ------------------------------------------
# Search Space
# ------------------------------------------

param_grid = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [5, 10, 15, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", None]
}

print("\nSearching best parameters...")

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=25,
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

# ------------------------------------------
# Evaluation
# ------------------------------------------

pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

# ------------------------------------------
# Save
# ------------------------------------------

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, MODEL_PATH)

report = {
    "best_parameters": search.best_params_,
    "cv_rmse": float(-search.best_score_),
    "test_mae": float(mae),
    "test_rmse": float(rmse),
    "test_r2": float(r2)
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=4)

# ------------------------------------------
# Display
# ------------------------------------------

print("\n" + "=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

for k, v in search.best_params_.items():
    print(f"{k}: {v}")

print("\nPerformance")
print("-" * 30)
print(f"CV RMSE : {-search.best_score_:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R² : {r2:.4f}")

print("\nSaved:")
print(MODEL_PATH)
print(REPORT_PATH)
print("=" * 70)