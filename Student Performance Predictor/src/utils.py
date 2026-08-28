# ==========================================
# File: src/utils.py
# Purpose: Reusable helper functions
# ==========================================

from pathlib import Path
import pandas as pd
import joblib


def load_dataset(path: str):
    """
    Load a CSV dataset.

    Parameters
    ----------
    path : str
        CSV file path.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(path)


def save_model(model, path: str):
    """
    Save trained model.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    """
    Load trained model.
    """
    return joblib.load(path)


def create_folder(path: str):
    """
    Create folder if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)