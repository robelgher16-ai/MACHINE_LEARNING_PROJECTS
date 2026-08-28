<<<<<<< HEAD
# 🎓 Student Performance Predictor

An end-to-end Machine Learning application that predicts a student's academic grade from demographic, academic, and behavioral features.

## 🚀 Project Overview

This project demonstrates a complete Machine Learning workflow:

- Data loading and exploration
- Data preprocessing
- Missing-value handling
- Categorical feature encoding
- Numerical feature scaling
- Train/test splitting
- Multiple regression models
- Model comparison
- Model evaluation
- Feature analysis
- Model persistence
- FastAPI REST API
- Streamlit web application

The final system allows a user to enter student information and receive a predicted academic grade.

---

## 🧠 Machine Learning Pipeline

```text
Student Dataset
      ↓
Data Cleaning
      ↓
Feature / Target Separation
      ↓
Train / Test Split
      ↓
Preprocessing
      ├── Numerical → Imputation → StandardScaler
      └── Categorical → Imputation → OneHotEncoder
      ↓
Multiple Regression Models
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Model Persistence
      ↓
FastAPI
      ↓
Streamlit
      ↓
Student Grade Prediction
```

## 🤖 Models Tested

The project compares several regression algorithms:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree Regressor
5. Random Forest Regressor
6. Gradient Boosting Regressor
7. Extra Trees Regressor
8. XGBoost
9. LightGBM

## 🏆 Model Selection

Models are evaluated using:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

The model with the lowest RMSE was selected as the final model.

The trained pipeline is saved as:

```text
models/best_model.pkl
```

## 📊 Grade Mapping

The original categorical grades are converted into numerical values for regression:

| Grade | Value |
| ----- | ----: |
| Fail  |     0 |
| DD    |     1 |
| DC    |     2 |
| CC    |     3 |
| CB    |     4 |
| BB    |     5 |
| BA    |     6 |
| AA    |     7 |

The prediction is converted back to the corresponding grade before being displayed to the user.

## 🔌 FastAPI

The project includes a REST API for making predictions.

Start the API with:

```bash
python -m uvicorn api.app:app --reload
```

API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Available endpoints

```text
GET  /
GET  /health
POST /predict
POST /batch_predict
```

## 🖥️ Streamlit Application

The project also contains a Streamlit interface.

Run:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The user enters student information and receives the predicted grade through the web interface.

## 📁 Project Structure

```text
Student Performance Predictor/
│
├── api/
│   └── app.py
│
├── app.py
│
├── data/
│   └── raw/
│       └── Students_Performance.csv
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── reports/
│   └── metrics.json
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Application

### Start FastAPI

```bash
python -m uvicorn api.app:app --reload
```

### Start Streamlit

Open another terminal, activate the environment, and run:

```bash
streamlit run app.py
```

## 📈 Evaluation

The project stores evaluation metrics in:

```text
reports/metrics.json
```

The evaluation includes:

```text
MAE
RMSE
R²
```

## 🔮 Future Improvements

Possible future improvements include:

- Hyperparameter optimization
- Cross-validation
- Better feature engineering
- Explainable AI with SHAP
- Model monitoring
- Prediction confidence/uncertainty
- Docker deployment
- Cloud deployment
- Authentication for the API
- Database integration
- CI/CD pipeline
- Automated model retraining

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib
- Seaborn
- Joblib
- FastAPI
- Pydantic
- Uvicorn
- Streamlit

## 🎯 Project Goal

The goal of this project is to demonstrate how a Machine Learning model can move from **raw data → model training → evaluation → deployment → user-facing application**.

This project is designed as an end-to-end Machine Learning portfolio project.
=======
robity you did not missed today

as you know everythong will by reson
robi


donefor today
>>>>>>> d83e872bd704e4a86fb26fe14f30655dacc1016b
