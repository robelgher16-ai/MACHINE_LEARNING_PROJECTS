# 🎓 Student Performance Predictor

An end-to-end Machine Learning project that predicts a student's final academic grade using demographic information, attendance, study habits, scholarship, and classroom behavior.

---

## 🚀 Project Features

- Data Cleaning & Validation
- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Feature Engineering
- Machine Learning Pipeline
- Cross Validation
- Hyperparameter Tuning
- SHAP Explainable AI
- Streamlit Web Application

---

## 📊 Dataset

- **Rows:** 145 students
- **Features:** 13
- **Target:** Grade (Fail → AA)

### Grade Mapping

| Grade | Score |
| ----- | ----: |
| Fail  |     0 |
| DD    |     1 |
| DC    |     2 |
| CC    |     3 |
| CB    |     4 |
| BB    |     5 |
| BA    |     6 |
| AA    |     7 |

---

## 🤖 Models Trained

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Extra Trees
- XGBoost
- LightGBM

### Best Model

**Random Forest Regressor**

| Metric | Value |
| ------ | ----: |
| MAE    |  1.83 |
| RMSE   |  2.23 |
| R²     | 0.087 |

---

## 📁 Project Structure

```text
Student Performance Predictor/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
├── models/
├── reports/
└── src/
```

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Explainable AI

The application includes SHAP visualizations showing which features contributed most to each prediction.

---

## 👨‍💻 Author

**Robel Gebregziabher**

AI Engineering • Machine Learning • Deep Learning
