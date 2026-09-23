# ==========================================
# File: src/classification_train.py
# Purpose: Train student grade classification
# ==========================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_cleaning import clean_dataset
from src.preprocessing import create_preprocessor


warnings.filterwarnings("ignore")


# ==========================================
# Configuration
# ==========================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Students_Performance.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_classifier.pkl"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

METRICS_PATH = (
    REPORTS_DIR
    / "classification_metrics.csv"
)

CV_RESULTS_PATH = (
    REPORTS_DIR
    / "classification_cv_results.csv"
)

CONFUSION_MATRIX_PATH = (
    REPORTS_DIR
    / "classification_confusion_matrix.csv"
)

REPORT_PATH = (
    REPORTS_DIR
    / "classification_report.csv"
)

METADATA_PATH = (
    REPORTS_DIR
    / "classification_metadata.json"
)


# ==========================================
# Grade Mapping
# ==========================================

GRADE_MAPPING = {
    "Fail": 0,
    "DD": 1,
    "DC": 2,
    "CC": 3,
    "CB": 4,
    "BB": 5,
    "BA": 6,
    "AA": 7
}

REVERSE_GRADE_MAPPING = {
    value: key
    for key, value in GRADE_MAPPING.items()
}


# ==========================================
# Header
# ==========================================

print("=" * 75)
print("STUDENT PERFORMANCE CLASSIFICATION")
print("MODEL TRAINING PIPELINE")
print("=" * 75)


# ==========================================
# 1. Load Dataset
# ==========================================

print("\n[1/12] Loading dataset...")

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"""
Dataset not found.

Expected path:
{DATA_PATH}
"""
    )


df_raw = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset shape: {df_raw.shape}"
)

print(
    f"Dataset path: {DATA_PATH}"
)


# ==========================================
# 2. Clean Dataset
# ==========================================

print(
    "\n[2/12] Cleaning and validating dataset..."
)

df_clean = clean_dataset(
    df_raw
)

print(
    "Dataset cleaning: OK"
)

print(
    f"Cleaned dataset shape: "
    f"{df_clean.shape}"
)


# ==========================================
# 3. Prepare Features and Target
# ==========================================

print(
    "\n[3/12] Preparing features and target..."
)


X = df_clean.drop(
    columns=[
        "Grade",
        "Student_ID"
    ]
)


y = df_clean["Grade"]


print(
    f"Features: {X.shape}"
)

print(
    f"Target:   {y.shape}"
)


# ==========================================
# 3.1 Validate Target
# ==========================================

print(
    "\n[3.1/12] Validating target..."
)


y = pd.to_numeric(
    y,
    errors="raise"
)

y = y.astype(int)


if y.isna().any():

    raise ValueError(
        "Target contains missing values."
    )


valid_classes = set(
    GRADE_MAPPING.values()
)


actual_classes = set(
    y.unique()
)


if not actual_classes.issubset(
    valid_classes
):

    raise ValueError(
        f"Unexpected target classes: "
        f"{actual_classes}"
    )


print(
    "Target validation: OK"
)


print(
    "\nGrade distribution:"
)

print(
    y.value_counts()
    .sort_index()
)


# ==========================================
# 4. Identify Feature Types
# ==========================================

print(
    "\n[4/12] Identifying feature types..."
)


numerical_features = (
    X
    .select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    )
    .columns
    .tolist()
)


categorical_features = (
    X
    .select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    )
    .columns
    .tolist()
)


print("\nNumerical features:")

for feature in numerical_features:

    print(
        f"  - {feature}"
    )


print("\nCategorical features:")

for feature in categorical_features:

    print(
        f"  - {feature}"
    )


# ==========================================
# 5. Train/Test Split
# ==========================================

print(
    "\n[5/12] Creating stratified train/test split..."
)


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples:  {len(X_test)}"
)


# ==========================================
# 6. Cross Validation
# ==========================================

print(
    "\n[6/12] Creating cross-validation strategy..."
)


cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)


print(
    f"Cross-validation folds: {CV_FOLDS}"
)

print(
    "Cross-validation strategy: OK"
)


# ==========================================
# 7. Create Models
# ==========================================

print(
    "\n[7/12] Creating classification models..."
)


models = {

    # --------------------------------------
    # Linear Models
    # --------------------------------------

    "Logistic Regression":
        LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE
        ),

    "Ridge Classifier":
        RidgeClassifier(
            alpha=1.0
        ),

    # --------------------------------------
    # Tree Models
    # --------------------------------------

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=5
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        ),

    # --------------------------------------
    # Modern Boosting
    # --------------------------------------

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="multi:softprob",
            num_class=8,
            eval_metric="mlogloss"
        ),

    "LightGBM":
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
            class_weight="balanced"
        )
}


print(
    f"Models to train: {len(models)}"
)


# ==========================================
# 8. Train and Evaluate
# ==========================================

print(
    "\n[8/12] Training and evaluating models..."
)

print("=" * 75)


results = []

cv_results = []

trained_models = {}


for name, model in models.items():

    print(
        f"\nTraining: {name}"
    )

    # --------------------------------------
    # Create fresh preprocessor
    # --------------------------------------

    model_preprocessor = (
        create_preprocessor(
            numerical_features=
                numerical_features,
            categorical_features=
                categorical_features
        )
    )


    pipeline = Pipeline([
        (
            "preprocessor",
            model_preprocessor
        ),
        (
            "model",
            model
        )
    ])


    # --------------------------------------
    # Cross Validation
    # --------------------------------------

    print(
        "  Running cross-validation..."
    )


    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy":
            "balanced_accuracy",
        "precision":
            "precision_weighted",
        "recall":
            "recall_weighted",
        "f1":
            "f1_weighted"
    }


    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )


    cv_accuracy = (
        cv_scores[
            "test_accuracy"
        ]
    )

    cv_balanced_accuracy = (
        cv_scores[
            "test_balanced_accuracy"
        ]
    )

    cv_precision = (
        cv_scores[
            "test_precision"
        ]
    )

    cv_recall = (
        cv_scores[
            "test_recall"
        ]
    )

    cv_f1 = (
        cv_scores[
            "test_f1"
        ]
    )


    print(
        f"  CV Accuracy          : "
        f"{cv_accuracy.mean():.4f} "
        f"± {cv_accuracy.std():.4f}"
    )

    print(
        f"  CV Balanced Accuracy : "
        f"{cv_balanced_accuracy.mean():.4f} "
        f"± {cv_balanced_accuracy.std():.4f}"
    )

    print(
        f"  CV F1                : "
        f"{cv_f1.mean():.4f} "
        f"± {cv_f1.std():.4f}"
    )


    # --------------------------------------
    # Train Full Training Split
    # --------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------
    # Test Prediction
    # --------------------------------------

    predictions = (
        pipeline.predict(
            X_test
        )
    )


    # --------------------------------------
    # Test Metrics
    # --------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )


    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    print(
        f"  Test Accuracy          : "
        f"{accuracy:.4f}"
    )

    print(
        f"  Test Balanced Accuracy : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"  Test Precision         : "
        f"{precision:.4f}"
    )

    print(
        f"  Test Recall            : "
        f"{recall:.4f}"
    )

    print(
        f"  Test F1                : "
        f"{f1:.4f}"
    )


    # --------------------------------------
    # Store Test Results
    # --------------------------------------

    results.append({

        "Model": name,

        "Accuracy":
            round(
                accuracy,
                4
            ),

        "Balanced_Accuracy":
            round(
                balanced_accuracy,
                4
            ),

        "Precision":
            round(
                precision,
                4
            ),

        "Recall":
            round(
                recall,
                4
            ),

        "F1":
            round(
                f1,
                4
            )
    })


    # --------------------------------------
    # Store CV Results
    # --------------------------------------

    cv_results.append({

        "Model": name,

        "CV_Accuracy":
            round(
                cv_accuracy.mean(),
                4
            ),

        "CV_Accuracy_STD":
            round(
                cv_accuracy.std(),
                4
            ),

        "CV_Balanced_Accuracy":
            round(
                cv_balanced_accuracy.mean(),
                4
            ),

        "CV_Balanced_Accuracy_STD":
            round(
                cv_balanced_accuracy.std(),
                4
            ),

        "CV_Precision":
            round(
                cv_precision.mean(),
                4
            ),

        "CV_Recall":
            round(
                cv_recall.mean(),
                4
            ),

        "CV_F1":
            round(
                cv_f1.mean(),
                4
            ),

        "CV_F1_STD":
            round(
                cv_f1.std(),
                4
            )
    })


    trained_models[name] = pipeline


# ==========================================
# 9. Model Comparison
# ==========================================

print("\n")

print("=" * 75)

print(
    "MODEL COMPARISON"
)

print("=" * 75)


results_df = pd.DataFrame(
    results
)


cv_results_df = pd.DataFrame(
    cv_results
)


# ------------------------------------------
# Merge test and CV results
# ------------------------------------------

comparison_df = results_df.merge(
    cv_results_df,
    on="Model"
)


# ------------------------------------------
# Select based on CV F1
# ------------------------------------------

comparison_df = (
    comparison_df
    .sort_values(
        by="CV_F1",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    comparison_df.to_string(
        index=False
    )
)


# ==========================================
# 10. Select Best Model
# ==========================================

print(
    "\n[9/12] Selecting best classification model..."
)


best_model_name = (
    comparison_df
    .iloc[0]["Model"]
)


best_model = (
    trained_models[
        best_model_name
    ]
)


best_row = (
    comparison_df.iloc[0]
)


print(
    f"Selected model: "
    f"{best_model_name}"
)


print(
    f"CV F1: "
    f"{best_row['CV_F1']}"
)


print(
    f"Test F1: "
    f"{best_row['F1']}"
)


print(
    f"Test Accuracy: "
    f"{best_row['Accuracy']}"
)


# ==========================================
# 11. Detailed Evaluation
# ==========================================

print(
    "\n[10/12] Creating detailed evaluation..."
)


final_predictions = (
    best_model.predict(
        X_test
    )
)


# ------------------------------------------
# Confusion Matrix
# ------------------------------------------

cm = confusion_matrix(
    y_test,
    final_predictions,
    labels=list(
        range(8)
    )
)


cm_df = pd.DataFrame(
    cm,
    index=[
        REVERSE_GRADE_MAPPING[i]
        for i in range(8)
    ],
    columns=[
        REVERSE_GRADE_MAPPING[i]
        for i in range(8)
    ]
)


print("\nConfusion Matrix:")

print(
    cm_df.to_string()
)


# ------------------------------------------
# Classification Report
# ------------------------------------------

classification_report_dict = (
    classification_report(
        y_test,
        final_predictions,
        labels=list(range(8)),
        target_names=[
            REVERSE_GRADE_MAPPING[i]
            for i in range(8)
        ],
        output_dict=True,
        zero_division=0
    )
)


classification_report_df = (
    pd.DataFrame(
        classification_report_dict
    ).transpose()
)


print(
    "\nClassification Report:"
)

print(
    classification_report_df.to_string()
)


# ==========================================
# 12. Save Everything
# ==========================================

print(
    "\n[11/12] Saving model and reports..."
)


MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------
# Save model
# ------------------------------------------

joblib.dump(
    best_model,
    MODEL_PATH
)


# ------------------------------------------
# Save metrics
# ------------------------------------------

comparison_df.to_csv(
    METRICS_PATH,
    index=False
)


# ------------------------------------------
# Save CV results
# ------------------------------------------

cv_results_df.to_csv(
    CV_RESULTS_PATH,
    index=False
)


# ------------------------------------------
# Save confusion matrix
# ------------------------------------------

cm_df.to_csv(
    CONFUSION_MATRIX_PATH
)


# ------------------------------------------
# Save classification report
# ------------------------------------------

classification_report_df.to_csv(
    REPORT_PATH
)


# ------------------------------------------
# Metadata
# ------------------------------------------

metadata = {

    "project":
        "Student Performance Predictor",

    "task":
        "Multi-class classification",

    "dataset": {

        "path":
            str(DATA_PATH),

        "rows":
            int(df_clean.shape[0]),

        "columns":
            int(df_clean.shape[1])
    },

    "features": {

        "count":
            int(X.shape[1]),

        "numerical":
            numerical_features,

        "categorical":
            categorical_features
    },

    "target": {

        "name":
            "Grade",

        "type":
            "multiclass_ordinal",

        "classes":
            {
                str(key): value
                for key, value
                in GRADE_MAPPING.items()
            }
    },

    "training": {

        "test_size":
            TEST_SIZE,

        "random_state":
            RANDOM_STATE,

        "cv_folds":
            CV_FOLDS,

        "training_samples":
            int(len(X_train)),

        "testing_samples":
            int(len(X_test))
    },

    "best_model": {

        "name":
            best_model_name,

        "CV_F1":
            float(best_row["CV_F1"]),

        "CV_Accuracy":
            float(
                best_row[
                    "CV_Accuracy"
                ]
            ),

        "CV_Balanced_Accuracy":
            float(
                best_row[
                    "CV_Balanced_Accuracy"
                ]
            ),

        "Test_Accuracy":
            float(
                best_row["Accuracy"]
            ),

        "Test_Balanced_Accuracy":
            float(
                best_row[
                    "Balanced_Accuracy"
                ]
            ),

        "Test_Precision":
            float(
                best_row["Precision"]
            ),

        "Test_Recall":
            float(
                best_row["Recall"]
            ),

        "Test_F1":
            float(
                best_row["F1"]
            )
    },

    "artifacts": {

        "model":
            str(MODEL_PATH),

        "metrics":
            str(METRICS_PATH),

        "cv_results":
            str(CV_RESULTS_PATH),

        "confusion_matrix":
            str(CONFUSION_MATRIX_PATH),

        "classification_report":
            str(REPORT_PATH)
    }
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ==========================================
# Final Report
# ==========================================

print(
    "\n[12/12] Training complete"
)


print("\n")

print("=" * 75)

print(
    "CLASSIFICATION TRAINING COMPLETE"
)

print("=" * 75)


print(
    f"Best Model : "
    f"{best_model_name}"
)


print(
    f"CV Accuracy : "
    f"{best_row['CV_Accuracy']}"
)


print(
    f"CV F1       : "
    f"{best_row['CV_F1']}"
)


print(
    f"Test Accuracy : "
    f"{best_row['Accuracy']}"
)


print(
    f"Test F1       : "
    f"{best_row['F1']}"
)


print(
    "\nSaved files:"
)


print(
    f"Model              : "
    f"{MODEL_PATH}"
)


print(
    f"Metrics            : "
    f"{METRICS_PATH}"
)


print(
    f"CV Results         : "
    f"{CV_RESULTS_PATH}"
)


print(
    f"Confusion Matrix   : "
    f"{CONFUSION_MATRIX_PATH}"
)


print(
    f"Classification     : "
    f"{REPORT_PATH}"
)


print(
    f"Metadata           : "
    f"{METADATA_PATH}"
)


print("=" * 75)