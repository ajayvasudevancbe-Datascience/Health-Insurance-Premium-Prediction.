# Health Insurance Premium Prediction — Streamlit

Interactive public web app for the **Health Insurance Premium Prediction** case study.

## Model
- Linear Regression
- One-hot encoding using `pd.get_dummies(..., drop_first=True)`
- Train/test split: 70/30
- `random_state=42`
- Input features: Age, Sex, BMI, Children, Smoker, Region
- Target: Insurance Charges

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy publicly with Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Go to Streamlit Community Cloud.
4. Connect your GitHub account.
5. Select the repository.
6. Set the main file to `app.py`.
7. Click **Deploy**.
8. Your app will receive a public `streamlit.app` URL.

The app uses a public copy of the standard insurance dataset by default. You can also upload your own `insurance_prediction.csv` from the sidebar.

## Expected CSV columns

`age, sex, bmi, children, smoker, region, charges`
