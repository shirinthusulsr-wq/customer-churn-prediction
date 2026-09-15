# Customer Churn Prediction & Retention Analytics
## Live Application

**[Launch the Customer Churn Predictor](https://shirin-thusu-churnanalytics.streamlit.app/)**

Use the interactive application to enter customer characteristics, estimate churn probability, classify customer risk, and determine retention priority.
An end-to-end machine learning project that predicts customer churn and translates predictions into actionable customer retention priorities using Customer Lifetime Value (CLTV).

## Business Problem

Customer churn is a major challenge for subscription-based businesses. Predicting which customers are at risk of leaving can allow companies to intervene before churn occurs.

This project addresses two key questions:

1. Which customers are most likely to churn?
2. Which at-risk customers should be prioritized for retention?

Rather than stopping at churn prediction, the project combines predicted churn probability with Customer Lifetime Value (CLTV) to create a business-oriented retention prioritization framework.

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Logistic Regression
- Random Forest
- Google Colab
- Streamlit
- Git & GitHub

## Exploratory Data Analysis

Several important churn patterns emerged from the analysis:

- **Fiber optic customers:** 41.89% churn rate, compared with 18.96% for DSL customers.
- **Electronic check:** 45.29% churn rate, the highest among payment methods.
- **No tech support:** 41.64% churn rate, compared with 15.17% for customers with tech support.
- Major reported churn reasons included support-person attitude, competitor download speeds, competitor data allowances, competitive offers, network reliability, product dissatisfaction, and price.

These relationships are associations and should not automatically be interpreted as causal effects.

## Machine Learning

Two classification models were evaluated.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.802 | 0.643 | 0.572 | 0.605 | 0.849 |
| Random Forest | 0.796 | 0.644 | 0.521 | 0.576 | 0.841 |

Logistic Regression produced the stronger overall baseline performance.

## Classification Threshold Optimization

At the default 0.50 threshold, Logistic Regression achieved **57.2% recall**.

Because failing to identify an actual churner can limit the opportunity for proactive retention, alternative operating thresholds were evaluated.

Reducing the threshold to **0.40** resulted in:

- Accuracy: **78.4%**
- Precision: **58.0%**
- Recall: **67.1%**
- F1 Score: **62.2%**
- ROC-AUC: **84.9%**

The lower threshold identified **251 of 374 actual churners**, compared with 214 at the default threshold.

The 0.40 threshold is an illustrative business-oriented choice. An actual production threshold should depend on the cost of retention interventions relative to the expected cost of churn.

## Customer Retention Prioritization

Predicted churn probability was combined with CLTV to classify customers into four actionable segments:

| Segment | Meaning |
|---|---|
| Priority Retention | Elevated churn risk + high customer value |
| Retention Candidate | Elevated churn risk + lower customer value |
| Relationship Maintenance | Lower churn risk + high customer value |
| Lower Priority | Lower churn risk + lower customer value |

Using a CLTV median of **4,527** and the selected churn threshold:

- **162 customers** were classified as Priority Retention.
- Their average CLTV was **5,278.75**.
- Their average predicted churn probability was approximately **61%**.

This allows retention resources to be focused on customers where both predicted churn risk and customer value are elevated.

## Interactive Streamlit Application

The project includes an interactive Streamlit application where a user can enter customer characteristics and receive:

- Churn probability
- Churn risk level
- Customer value classification
- Retention priority
- Suggested retention action

## Project Structure

```text
customer-churn-prediction/
|
|-- Model/
|   |-- logistic_churn_model.pkl
|   |-- scaler.pkl
|   |-- model_features.pkl
|   `-- business_parameters.pkl
|
|-- Notebook/
|   `-- Customer_Churn_Analysis.ipynb
|
|-- app.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Key Skills Demonstrated

**Data Cleaning | Exploratory Data Analysis | Machine Learning | Classification | Model Evaluation | Threshold Optimization | Customer Segmentation | Business Analytics | Streamlit | Git | GitHub**

## Limitations

The model is designed as a portfolio demonstration rather than a production decision system. Model relationships are predictive rather than causal, and the retention thresholds would require economic validation before real-world deployment.