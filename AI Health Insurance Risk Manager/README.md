# AI HealthGuard — AI Health Insurance Risk Manager

A Streamlit-based Machine Learning application for predicting health insurance costs and classifying customer risk.

## Hackathon Track

**Track 2 — AI Risk Manager**

## Features

- Insurance cost prediction
- Low / Medium / High risk classification
- Risk factor analysis
- Interactive Streamlit dashboard
- Business insights
- Model performance metrics
- AI-style risk explanations and recommendations

## Expected CSV Columns

The application works with a CSV containing:

- age
- sex
- bmi
- children
- smoker
- region
- charges

Place the CSV in the project folder as `health_insurance.csv` or `insurance.csv`.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application can also run without a CSV because it includes sample data for demonstration.
