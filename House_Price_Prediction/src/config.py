from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_PATH = BASE_DIR / "data" / "House Price Prediction Dataset.csv"

# Models
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "final_model.pkl"