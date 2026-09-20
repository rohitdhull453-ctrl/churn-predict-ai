# Customer Churn Prediction

A Machine Learning web application that predicts whether a customer is likely to churn.

The application is built using Python, Scikit-learn and Streamlit.

## Features

- Customer churn prediction
- Multiple Machine Learning models
- Logistic Regression
- Decision Tree
- KNN
- Probability estimation
- Interactive Streamlit UI
- Saved ML pipelines
- Ready for GitHub
- Ready for Streamlit deployment

## Machine Learning Models

The project contains three trained models:

1. Logistic Regression
2. Decision Tree
3. KNN

Each model contains the preprocessing pipeline.

## Input Features

### Numerical Features

- age
- income
- orders
- total_spent
- rating
- discount
- website_visits
- support_calls

### Categorical Features

- gender
- city
- education
- category
- payment
- membership

## Project Structure

```text
churn-prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── logistic_churn_prediction_model.pkl
├── DT_churn_prediction_model.pkl
├── KNN_churn_prediction_model.pkl
└── columns (2).pkl