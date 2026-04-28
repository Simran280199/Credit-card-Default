# 💳 Credit Card Default Prediction

This project predicts whether a customer will default on their credit card payment using various Machine Learning and Deep Learning models. It includes data preprocessing, feature scaling, hyperparameter tuning, model evaluation, and comparison of performance metrics.

---

## 📌 Project Overview

Credit card default prediction helps financial institutions identify customers who are likely to miss future payments. By analyzing customer demographic details, bill statements, and payment history, this project builds predictive models to minimize financial risk.

---

## 🚀 Features

✅ Data Cleaning & Preprocessing  
✅ Exploratory Data Analysis (EDA)  
✅ Feature Engineering  
✅ Standardization / Scaling  
✅ Hyperparameter Tuning using GridSearchCV  
✅ Multiple Models Comparison  
✅ Streamlit Web App Deployment  

---

## 🛠 Tech Stack

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- TensorFlow / Keras  
- XGBoost  
- Streamlit  

---

## 📂 Dataset

The dataset contains customer information such as:

- LIMIT_BAL – Amount of given credit  
- SEX – Gender  
- EDUCATION – Education level  
- MARRIAGE – Marital status  
- AGE – Age  
- PAY_0 to PAY_6 – Repayment status  
- BILL_AMT1 to BILL_AMT6 – Bill statement amounts  
- PAY_AMT1 to PAY_AMT6 – Previous payment amounts  

### 🎯 Target Variable:
- `default.payment.next.month`
  - 1 → Default
  - 0 → No Default

---

## 🤖 Models Used

### 1. :contentReference[oaicite:0]{index=0}
- Hyperparameter tuned using GridSearchCV

### 2. :contentReference[oaicite:1]{index=1}
- Hyperparameter tuned using GridSearchCV

### 3. :contentReference[oaicite:2]{index=2}
- Gradient boosting based model

### 4. :contentReference[oaicite:3]{index=3}
- Deep Learning approach using Keras

---

## 📊 Model Performance

The models were evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Classification Report  

Example results:

| Model | Test Accuracy |
|-------|--------------|
| SVM | ~81.89% |
| Random Forest | ~81.97% |
| ANN | ~82.4% |
| XGBoost | ~81.48% |

---

## 📈 Workflow

1. Data Loading  
2. Data Cleaning  
3. EDA & Visualization  
4. Feature Scaling  
5. Model Training  
6. Hyperparameter Tuning  
7. Model Evaluation  
8. Deployment using Streamlit  

---

## 🌐 Streamlit App:-https://credit-card-default-sihag.streamlit.app/

```bash
streamlit run app.py
