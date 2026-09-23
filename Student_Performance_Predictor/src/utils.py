
# ==========================================
# File: src/utils.py
# Purpose: Reusable project helper functions
# ==========================================

from pathlib import Path
import json
from typing import Any

import joblib
import pandas as pd


# ==========================================
# File / Directory Helpers
# ==========================================

def create_folder(path: str | Path) -> Path:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path : str or Path
        Directory path.

    Returns
    -------
    Path
        Created/existing directory path.
    """

    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)

    return folder


def check_file_exists(path: str | Path) -> Path:
    """
    Check whether a file exists.

    Parameters
    ----------
    path : str or Path
        File path.

    Returns
    -------
    Path
        Validated file path.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """

    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"File not found: {file_path.resolve()}"
        )

    return file_path


# ==========================================
# Dataset Helpers
# ==========================================

def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV dataset.

    Parameters
    ----------
    path : str or Path
        CSV file path.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    file_path = check_file_exists(path)

    return pd.read_csv(file_path)


# ==========================================
# Model Helpers
# ==========================================

def save_model(model: Any, path: str | Path) -> Path:
    """
    Save a trained model using joblib.

    Parameters
    ----------
    model : Any
        Trained machine learning model or pipeline.

    path : str or Path
        Destination path.

    Returns
    -------
    Path
        Saved model path.
    """

    model_path = Path(path)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, model_path)

    return model_path


def load_model(path: str | Path) -> Any:
    """
    Load a trained model from disk.

    Parameters
    ----------
    path : str or Path
        Saved model path.

    Returns
    -------
    Any
        Loaded model.
    """

    model_path = check_file_exists(path)

    return joblib.load(model_path)


# ==========================================
# JSON Helpers
# ==========================================

def save_json(data: dict, path: str | Path) -> Path:
    """
    Save dictionary data as a JSON file.

    Parameters
    ----------
    data : dict
        Data to save.

    path : str or Path
        JSON destination path.

    Returns
    -------
    Path
        Saved JSON path.
    """

    json_path = Path(path)

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with json_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    return json_path


def load_json(path: str | Path) -> dict:
    """
    Load a JSON file.

    Parameters
    ----------
    path : str or Path
        JSON file path.

    Returns
    -------
    dict
        Loaded JSON data.
    """

    json_path = check_file_exists(path)

    with json_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)
