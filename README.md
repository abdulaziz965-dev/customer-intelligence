# Customer Intelligence — Telecom Customer Churn Prediction

A machine learning project that analyzes customer behavior and predicts telecom customer churn using Python and scikit-learn.

## Project Overview

Customer churn occurs when customers stop using a company's services. Identifying customers who are likely to leave can help businesses develop targeted customer-retention strategies.

This project explores customer churn patterns, performs statistical analysis, builds classification models and evaluates their ability to identify customers at risk of leaving.

## Objectives

- Understand customer churn patterns through exploratory data analysis.
- Identify features associated with customer churn.
- Apply statistical hypothesis testing.
- Perform feature engineering and preprocessing.
- Train and compare machine learning models.
- Optimize the classification threshold.
- Interpret model predictions using feature importance techniques.

## Dataset

**Dataset:** IBM Telco Customer Churn

- Records: 7,043 customers
- Features: 21 original columns
- Target variable: `Churn`
- Target classes: Yes and No

The dataset contains customer demographics, subscribed services, contract information, payment methods and billing details.

## Technology Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Statistical analysis | SciPy, Statsmodels |
| Machine learning | Scikit-learn |
| Development | Jupyter Notebook, VS Code |
| Version control | Git, GitHub |

## Project Structure

```text
customer-intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_logistic_regression.ipynb
│   └── 05_random_forest.ipynb
│
├── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Project Workflow

### 1. Data Understanding and Cleaning
- Inspected dataset dimensions, data types and distributions.
- Checked missing values and duplicate records.
- Converted `TotalCharges` into a numerical feature.
- Handled 11 missing `TotalCharges` values.

### 2. Exploratory Data Analysis
- Examined overall churn distribution.
- Analyzed churn by contract type, tenure and payment method.
- Investigated demographic and service-related patterns.
- Visualized relationships between customer attributes and churn.

### 3. Statistical Analysis
- Applied Chi-square tests of independence.
- Measured association strength using Cramér's V.
- Applied Benjamini-Hochberg correction for multiple comparisons.

### 4. Feature Engineering
- Removed the customer identifier.
- Separated predictors and target variable.
- Applied StandardScaler to numerical features.
- Applied OneHotEncoder to categorical features.
- Used an 80:20 stratified train-test split.

### 5. Model Development

The following configurations were evaluated:

- Logistic Regression
- Tuned Logistic Regression using a lower classification threshold
- Random Forest
- Balanced Random Forest

### 6. Model Explainability
- Examined Logistic Regression coefficients.
- Calculated odds ratios.
- Analyzed Random Forest feature importance.
- Used permutation importance to evaluate feature contributions.

### 7. Model Validation
- Performed five-fold stratified cross-validation.
- Compared default and tuned classification thresholds.
- Evaluated the final model using a confusion matrix and classification metrics.

## Model Performance

The following results were obtained on the test dataset:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 84.21% |
| Logistic Regression (Tuned) | 76.93% | 55.16% | 70.05% | 61.72% | 84.21% |
| Random Forest | 78.35% | 61.86% | 48.13% | 54.14% | 82.06% |
| Balanced Random Forest | 75.23% | 52.47% | 71.12% | 60.39% | 83.43% |

**Selected model:** Tuned Logistic Regression

**Classification threshold:** 0.3645

The tuned model achieved a recall of 70.05% and an F1 score of 61.72% on the test dataset.

Five-fold cross-validation of default Logistic Regression achieved an average ROC-AUC of 84.62%.

*Note: The test dataset was used repeatedly during model comparison and threshold experimentation. These results should therefore be treated as exploratory rather than as a completely independent final performance estimate.*

## Key Findings

- Month-to-month customers exhibited substantially higher churn rates than customers with longer-term contracts.
- Customer tenure showed a strong relationship with churn.
- Electronic check users had a higher observed churn rate than customers using automatic payment methods.
- Fiber optic customers showed higher observed churn than DSL customers.
- Contract type was the most influential feature according to permutation importance for the