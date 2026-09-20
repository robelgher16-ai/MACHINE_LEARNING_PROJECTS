# House Price Prediction

A complete end-to-end Machine Learning project that predicts house prices from property characteristics using Python, Scikit-learn, Streamlit, and FastAPI.

The project covers the complete machine learning workflow from data analysis and model training to a web interface and REST API.

---

## Project Overview

The goal of this project is to build a machine learning system that predicts the estimated price of a house based on:

- Area
- Bedrooms
- Bathrooms
- Floors
- Year Built
- Location
- Condition
- Garage availability

The project uses a Kaggle House Price Prediction Dataset containing 2,000 records.

---

## Application Screenshots

### Home Page

![Home Page](images/home.png)

### Prediction Page

![Prediction Page](images/predict.png)

### Prediction Result

![Prediction Result](images/result.png)

### About Page

![About Page](images/about.png)

---

## Features

- House price prediction
- Data preprocessing
- Categorical feature encoding
- Train/test splitting
- Multiple machine learning models
- Model evaluation
- Model comparison
- Final model serialization
- Streamlit web application
- FastAPI REST API
- Input validation with Pydantic
- Local prediction pipeline
- Professional project structure

---

## Machine Learning Workflow

The project follows an end-to-end machine learning workflow:

```text
Problem Definition
        ↓
Dataset Understanding
        ↓
Data Quality Audit
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Preprocessing Pipeline
        ↓
Baseline Model
        ↓
Multiple ML Models
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Final Model
        ↓
Model Serialization
        ↓
Prediction Pipeline
        ↓
Streamlit Application
        ↓
FastAPI API
        ↓
Deployment
```

---

## Dataset

**Dataset:** House Price Prediction Dataset

**Number of records:** 2,000

### Features

| Feature   | Description              |
| --------- | ------------------------ |
| Area      | House area               |
| Bedrooms  | Number of bedrooms       |
| Bathrooms | Number of bathrooms      |
| Floors    | Number of floors         |
| YearBuilt | Year the house was built |
| Location  | Property location        |
| Condition | Property condition       |
| Garage    | Garage availability      |
| Price     | Target variable          |

The `Id` column was removed because it does not provide meaningful predictive information for the model.

---

## Data Preprocessing

Categorical variables were converted into numerical features using one-hot encoding.

The encoded feature set contains:

```text
Area
Bedrooms
Bathrooms
Floors
YearBuilt
Location_Rural
Location_Suburban
Location_Urban
Condition_Fair
Condition_Good
Condition_Poor
Garage_Yes
```

Reference categories were excluded to avoid redundant dummy variables.

The final feature matrix contains:

```text
2000 samples
12 features
```

The dataset was divided into:

```text
Training set: 80% → 1600 samples
Testing set: 20% → 400 samples
```

---

## Models Evaluated

Several regression models were evaluated:

- Baseline model
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

### Model Evaluation

The main evaluation metrics were:

- MAE — Mean Absolute Error
- MSE — Mean Squared Error
- RMSE — Root Mean Squared Error
- R² — Coefficient of Determination

### Current Results

| Model             |        MAE |       RMSE |      R² |
| ----------------- | ---------: | ---------: | ------: |
| Baseline          | 252,671.95 | 292,329.63 | -0.0984 |
| Linear Regression | 243,241.98 | 279,859.73 | -0.0067 |
| Decision Tree     | 329,752.94 |          — | -1.0805 |
| Random Forest     | 252,671.95 |          — | -0.0984 |
| Gradient Boosting | 245,284.36 |          — | -0.0355 |
| XGBoost           | 248,001.52 |          — |       — |
| LightGBM          | 247,034.45 | 285,070.30 | -0.0446 |
| CatBoost          | 244,376.78 |          — |       — |

The current final model used by the application is **Linear Regression**.

---

## Important Dataset Limitation

The dataset appears to contain relatively weak relationships between the available input features and the target price.

For example, the exploratory analysis showed very weak correlations between several numerical features and `Price`.

This affects predictive performance and explains why the evaluated models do not achieve a strong positive R² score.

This is an important machine learning lesson: **model complexity cannot compensate for weak or uninformative data relationships.**

---

## Project Structure

```text
House_Price_Prediction/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── app/
│   └── app.py
│
├── data/
│   └── House Price Prediction Dataset.csv
│
├── images/
│   ├── home.png
│   ├── predict.png
│   ├── result.png
│   └── about.png
│
├── models/
│   └── final_model.pkl
│
├── notebooks/
│   ├── House_Price_Prediction.ipynb
│   └── all_in_one.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
│
├── README.md
└── requirements.txt
```

---

## Technology Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

### Application

- Streamlit

### API

- FastAPI
- Pydantic
- Uvicorn

### Development

- Jupyter Notebook
- Git
- GitHub

---

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/robelgher16-ai/MACHINE_LEARNING_PROJECTS.git
```

### 2. Navigate to the project

```bash
cd MACHINE_LEARNING_PROJECTS/House_Price_Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Streamlit Application

From the `House_Price_Prediction` directory:

```bash
streamlit run app/app.py
```

The application provides:

- Home page
- Prediction form
- Prediction result
- About page

---

## Run the FastAPI Application

From the project directory:

```bash
uvicorn api.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
GET /health
```

Prediction endpoint:

```text
POST /predict
```

Example request:

```json
{
  "area": 2500,
  "bedrooms": 4,
  "bathrooms": 3,
  "floors": 2,
  "year_built": 2018,
  "location": "Urban",
  "condition": "Good",
  "garage": "Yes"
}
```

Example response:

```json
{
  "predicted_price": 550000.0
}
```

---

## Model Pipeline

The application uses a saved Scikit-learn pipeline containing:

```text
Input Data
    ↓
Preprocessing
    ↓
Categorical Encoding
    ↓
Linear Regression
    ↓
Predicted House Price
```

The trained pipeline is stored in:

```text
models/final_model.pkl
```

---

## Learning Objectives

This project demonstrates practical understanding of:

- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Categorical encoding
- Train/test splitting
- Regression
- Model evaluation
- Model comparison
- Machine learning pipelines
- Model serialization
- Prediction systems
- Streamlit development
- REST API development
- Input validation
- Git/GitHub project management

---

## Future Improvements

Possible improvements include:

- Hyperparameter optimization
- More informative real-world housing data
- Additional feature engineering
- Cross-validation
- Error analysis
- SHAP explainability
- Improved model selection
- Better feature collection
- Cloud deployment
- API authentication
- Automated model retraining
- Model monitoring

---

## Project Status

**Machine Learning pipeline:** Completed

**Model comparison:** Completed

**Prediction pipeline:** Completed

**Streamlit interface:** Completed

**FastAPI interface:** Completed

**Screenshots:** Completed

**GitHub preparation:** In progress

**Deployment:** Next step

---

## Author

**Robel Gebregziabher**

Information Technology Student
Machine Learning | Deep Learning | Generative AI | AI Engineering

GitHub:

`https://github.com/robelgher16-ai`
