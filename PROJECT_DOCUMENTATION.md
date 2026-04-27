# 📄 Credit Card Default Prediction — Complete Project Documentation

> **Document Type:** Model Design Document (MDD)  
> **Project:** Credit Card Default Risk Classification  
> **Dataset:** UCI Taiwan Credit Card Default (April–September 2005)  
> **Best Model:** Random Forest Classifier  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Environment Setup](#3-environment-setup)
4. [Data Loading & Initial Inspection](#4-data-loading--initial-inspection)
5. [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis)
6. [Data Cleaning & Preprocessing](#6-data-cleaning--preprocessing)
7. [Distribution Analysis & Transformation](#7-distribution-analysis--transformation)
8. [Feature Engineering](#8-feature-engineering)
9. [Class Imbalance Treatment — SMOTE](#9-class-imbalance-treatment--smote)
10. [Model Building](#10-model-building)
11. [Model Evaluation & Comparison](#11-model-evaluation--comparison)
12. [Model Saving & Serialization](#12-model-saving--serialization)
13. [Streamlit Application](#13-streamlit-application)
14. [Project File Structure](#14-project-file-structure)
15. [How to Run](#15-how-to-run)

---

## 1. Project Overview

### Objective
Build an end-to-end machine learning classification system that predicts whether a credit card client will **default on their payment next month**, in order to:
- Minimize financial risk for the bank
- Maximize lending efficiency
- Provide an interactive tool for risk officers

### Business Problem
Banks face significant losses when credit card holders fail to make payments. Identifying high-risk clients **before** default occurs allows proactive intervention — adjusting credit limits, sending payment reminders, or triggering collections.

### Success Metrics
| Metric | Target |
|--------|--------|
| ROC-AUC | ≥ 0.74 |
| F1 Score | ≥ 0.48 |
| Recall (Default class) | ≥ 0.47 |

---

## 2. Dataset Description

| Property | Detail |
|---|---|
| Source | UCI Machine Learning Repository |
| Records | 30,000 customers |
| Features | 25 columns (24 features + 1 target) |
| Time Period | April 2005 – September 2005 |
| Region | Taiwan |
| Target Variable | `default.payment.next.month` (binary: 1=Yes, 0=No) |

### Feature Glossary

| Feature | Type | Description |
|---|---|---|
| `ID` | Integer | Unique customer identifier (dropped before modeling) |
| `LIMIT_BAL` | Numeric | Credit limit in NT dollars |
| `SEX` | Categorical | 1=Male, 2=Female |
| `EDUCATION` | Categorical | 1=Graduate, 2=University, 3=High School, 4=Others |
| `MARRIAGE` | Categorical | 1=Married, 2=Single, 3=Others |
| `AGE` | Numeric | Customer age in years |
| `PAY_0` | Categorical | Repayment status Sep 2005 |
| `PAY_2` to `PAY_6` | Categorical | Repayment status Aug–Apr 2005 |
| `BILL_AMT1–6` | Numeric | Bill statement amount Sep–Apr 2005 (NT$) |
| `PAY_AMT1–6` | Numeric | Previous payment amount Sep–Apr 2005 (NT$) |
| `default.payment.next.month` | Binary | **Target** — 1=Default, 0=No Default |

### Repayment Status Scale
- `-2` = No consumption; `-1` = Pay duly; `0` = Use of revolving credit  
- `1` = 1 month delay; `2` = 2 months delay; ... `9` = 9+ months delay

---

## 3. Environment Setup

### Prerequisites
- Python 3.10+
- pip / conda
- VS Code with Jupyter extension

### Installation
```bash
# Clone or create project folder
mkdir credit_card_default_project && cd credit_card_default_project

# Install dependencies
pip install -r requirements.txt
```

### Key Libraries Used
| Library | Version | Purpose |
|---|---|---|
| `pandas` | 2.2.2 | Data manipulation |
| `numpy` | 1.26.4 | Numerical computation |
| `scikit-learn` | 1.4.2 | ML models, preprocessing, metrics |
| `imbalanced-learn` | 0.12.3 | SMOTE oversampling |
| `matplotlib / seaborn` | 3.8.4 / 0.13.2 | Visualization |
| `plotly` | 5.22.0 | Interactive charts in Streamlit |
| `streamlit` | 1.35.0 | Web app framework |
| `joblib` | 1.4.2 | Model serialization |

---

## 4. Data Loading & Initial Inspection

### Steps Performed
1. Loaded CSV using `pd.read_csv()`
2. Renamed target column: `default.payment.next.month` → `default`
3. Inspected shape, dtypes, head/tail
4. Checked for missing values (none found)
5. Examined descriptive statistics

### Findings
```
Shape: (30000, 25)
Missing Values: 0 in all columns
Data Types: All int64
Target Distribution:
  0 (No Default) : 23,364 (77.88%)
  1 (Default)    :  6,636 (22.12%)
```

### Key Observations
- **No missing values** — dataset is complete
- **Class imbalance** — ~78:22 split confirms the need for SMOTE
- `LIMIT_BAL` ranges from 10,000 to 1,000,000 NT$ — wide range suggests scaling needed
- `BILL_AMT` values can be negative (credit notes / refunds)

---

## 5. Exploratory Data Analysis

### 5.1 Target Variable Distribution
**Finding:** 22.12% default rate — significant imbalance. A naïve model always predicting "No Default" would achieve 77.88% accuracy but be useless.

### 5.2 Categorical Feature Analysis
| Feature | Key Insight |
|---|---|
| SEX | Females (56%) outnumber males; males have slightly higher default rate |
| EDUCATION | University graduates form the largest group; higher education correlates with lower default |
| MARRIAGE | Single customers slightly more likely to default than married |

### 5.3 Age Distribution
- Mean age: ~35 years
- No major difference in age between defaulters and non-defaulters
- Slight uptick in defaults for customers aged 25–35

### 5.4 Credit Limit vs Default
- Non-defaulters have significantly **higher credit limits** on average
- Lower credit limits correlate with higher default probability

### 5.5 Bill & Payment Trends (6 months)
- Defaulters consistently have **higher bill amounts** and **lower payment amounts**
- Payment amount drops sharply for defaulters in the months preceding default

### 5.6 Correlation Analysis
- `BILL_AMT` columns are highly correlated with each other (r > 0.90) — collinearity exists
- `PAY_0` (latest repayment status) has the **highest correlation** with the target (~0.32)
- Payment history features (`PAY_*`) are most predictive

---

## 6. Data Cleaning & Preprocessing

### Steps
1. **Dropped `ID` column** — non-informative primary key
2. **Fixed EDUCATION anomalies:**
   - Categories `{0, 5, 6}` are undocumented → mapped to `4` (Others)
3. **Fixed MARRIAGE anomalies:**
   - Category `{0}` is undocumented → mapped to `3` (Others)
4. **No imputation required** — no missing values

### Code Snippet
```python
df.drop(columns=['ID'], inplace=True)
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
df['MARRIAGE']  = df['MARRIAGE'].replace({0: 3})
```

---

## 7. Distribution Analysis & Transformation

### 7.1 Skewness Assessment
Skewness was computed for all numeric features using `df.skew()`.

| Feature | Skewness (Before) | Action |
|---|---|---|
| `BILL_AMT1–6` | 1.2 – 2.8 | Log1p transform |
| `PAY_AMT1–6` | 5.0 – 9.0 | Log1p transform |
| `LIMIT_BAL` | 0.9 | Log1p transform |
| `AGE` | 0.6 | No transform |
| `PAY_0` to `PAY_6` | Categorical-like | No transform |

### 7.2 Transformation Applied
For all features with `|skew| > 1` and `min ≥ 0`, **Log1p transformation** was applied:

```python
df[col] = np.log1p(df[col])
```

**Why Log1p?**
- Handles zero values (unlike pure log)
- Compresses large outliers
- Reduces right skewness in bill/payment distributions

### 7.3 Effect
After transformation, skewness dropped significantly (most below 0.5), improving model performance, especially for distance-based or regularized models like Logistic Regression.

---

## 8. Feature Engineering

Four new features were created to capture latent customer behavior:

| New Feature | Formula | Rationale |
|---|---|---|
| `UTIL_RATIO` | `BILL_AMT1 / (LIMIT_BAL + 1)` | Credit utilization rate — high ratio = high risk |
| `AVG_BILL` | Mean of `BILL_AMT1–6` | Average spending across 6 months |
| `AVG_PAY_AMT` | Mean of `PAY_AMT1–6` | Average repayment capability |
| `TOTAL_DELAY` | Sum of `PAY_0` to `PAY_6` | Cumulative delay months — key risk indicator |

---

## 9. Class Imbalance Treatment — SMOTE

### Problem
The original dataset is imbalanced:
- Class 0 (No Default): 77.88%
- Class 1 (Default): 22.12%

Without treatment, models are biased toward the majority class, severely hurting recall on defaulters — which is the most critical metric for the bank.

### Solution: SMOTE (Synthetic Minority Oversampling Technique)
SMOTE generates **synthetic samples** for the minority class by interpolating between existing minority examples in feature space.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

### Critical Rule ⚠️
**SMOTE is applied ONLY to the training set** — never the test set. Applying it to the test set would cause data leakage and optimistically biased evaluation.

### Result
| | No Default | Default |
|---|---|---|
| Before SMOTE | 18,691 | 5,309 |
| After SMOTE | 18,691 | 18,691 |

---

## 10. Model Building

### Pipeline
```
Raw Data → Clean → Feature Engineer → Train/Test Split → 
SMOTE (train only) → StandardScaler → Model Training → Evaluation
```

### Models Trained
1. **Logistic Regression** — Baseline linear classifier; interpretable
2. **Decision Tree** — Non-linear, rule-based; max_depth=6 to prevent overfitting
3. **Random Forest** — Ensemble of 100 trees; best overall performance ⭐
4. **Gradient Boosting** — Sequential ensemble; high precision/recall balance

### Hyperparameters
| Model | Key Params |
|---|---|
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| Decision Tree | `max_depth=6`, `random_state=42` |
| Random Forest | `n_estimators=100`, `n_jobs=-1`, `random_state=42` |
| Gradient Boosting | `n_estimators=100`, `random_state=42` |

---

## 11. Model Evaluation & Comparison

### Metrics Used
| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted defaults, how many are actual defaults |
| **Recall** | Of actual defaults, how many were caught |
| **F1 Score** | Harmonic mean of Precision & Recall |
| **ROC-AUC** | Discriminative ability across all thresholds |

> 📌 **Recall** is the most important metric here — missing a real defaulter (false negative) is costlier than a false alarm.

### Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.6898 | 0.3714 | 0.5810 | 0.4531 | 0.6914 |
| Decision Tree | 0.7323 | 0.4211 | 0.5614 | 0.4813 | 0.7188 |
| **Random Forest** ⭐ | **0.7807** | **0.5044** | 0.4763 | **0.4899** | **0.7442** |
| Gradient Boosting | 0.7555 | 0.4568 | **0.5576** | **0.5022** | 0.7442 |

### Best Model: Random Forest
**Why chosen:**
- Highest accuracy (78.07%)
- Highest precision (50.44%) — fewer false alarms
- Highest ROC-AUC (tied with Gradient Boosting at 0.7442)
- More stable and less prone to overfitting vs individual trees
- Provides built-in feature importance

---

## 12. Model Saving & Serialization

### Files Saved
| File | Description |
|---|---|
| `models/model.pkl` | Trained Random Forest model |
| `models/scaler.pkl` | StandardScaler fitted on SMOTE-balanced training data |
| `models/meta.json` | Feature names, model name, all evaluation metrics |

### Code
```python
import joblib, json

joblib.dump(best_model, 'models/model.pkl')
joblib.dump(scaler,     'models/scaler.pkl')

meta = {
    'feature_names': X.columns.tolist(),
    'model_name': 'Random Forest',
    'metrics': results['Random Forest'],
    'all_results': results
}
with open('models/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
```

### Loading in App
```python
model  = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')
```

---

## 13. Streamlit Application

### Overview
A professional multi-page web application for credit risk analysts.

### Pages

#### 🏠 Home / Predict
- Input form for all 24 features (grouped by category)
- Real-time prediction with probability
- Risk gauge chart (Plotly)
- Probability breakdown bar chart
- Color-coded result (green=safe, red=default)

#### 📊 Model Dashboard
- Full model comparison table (all 4 models)
- Radar chart for multi-metric visual comparison
- Grouped bar chart by metric
- KPI cards for best model metrics
- All EDA/evaluation chart images

#### ℹ️ About Dataset
- Full dataset description and feature glossary
- Preprocessing steps summary
- Model list

### Running the App
```bash
streamlit run app.py
```

### App Architecture
```
app.py
├── @st.cache_resource → load_artifacts()  [model, scaler, meta]
├── Sidebar Navigation
├── Page 1: Predict
│   ├── Input Form (4 sections)
│   ├── Prediction → gauge + bar chart
│   └── Summary table
├── Page 2: Dashboard
│   ├── Metrics dataframe (highlighted)
│   ├── Radar chart
│   ├── Bar chart comparison
│   └── Saved chart images
└── Page 3: About
    └── Static documentation
```

---

## 14. Project File Structure

```
credit_card_default_project/
│
├── data/
│   └── Credit_Card_Default.csv           ← Original dataset
│
├── models/
│   ├── model.pkl                          ← Trained Random Forest
│   ├── scaler.pkl                         ← StandardScaler
│   ├── meta.json                          ← Feature names + metrics
│   ├── target_distribution.png
│   ├── categorical_analysis.png
│   ├── age_limit_analysis.png
│   ├── correlation_heatmap.png
│   ├── bill_payment_trends.png
│   ├── distribution_comparison.png
│   ├── smote_balance.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
├── Credit_Card_Default_Analysis.ipynb    ← Full Jupyter Notebook
├── app.py                                 ← Streamlit Web Application
├── requirements.txt                       ← Python dependencies
└── PROJECT_DOCUMENTATION.md              ← This file
```

---

## 15. How to Run

### Step 1: Install Dependencies
```bash
cd credit_card_default_project
pip install -r requirements.txt
```

### Step 2: Run Jupyter Notebook (EDA + Model Training)
```bash
# In VS Code
# Open Credit_Card_Default_Analysis.ipynb
# Run all cells top to bottom (generates model.pkl, scaler.pkl, charts)
```

Or from terminal:
```bash
jupyter nbconvert --to notebook --execute Credit_Card_Default_Analysis.ipynb
```

### Step 3: Launch Streamlit App
```bash
streamlit run app.py
```

Open browser at: `http://localhost:8501`

### Step 4: Test a Prediction
1. Go to **Home / Predict**
2. Enter customer details
3. Click **Predict Default Risk**
4. View gauge chart + probability

---

## Appendix: Glossary

| Term | Definition |
|---|---|
| SMOTE | Synthetic Minority Oversampling Technique — generates synthetic samples for minority class |
| ROC-AUC | Receiver Operating Characteristic — Area Under Curve; model's ability to discriminate classes |
| Recall | Sensitivity — fraction of actual positives correctly identified |
| Log1p | `log(1 + x)` transformation to reduce skewness while handling zeros |
| StandardScaler | Transforms features to zero mean and unit variance |
| Stratified Split | Train/test split that preserves the class ratio in both sets |
| Feature Importance | Random Forest's measure of how much each feature reduces impurity |

---

*Document generated for Credit Card Default Prediction ML Project | VS Code + Jupyter + Streamlit*
