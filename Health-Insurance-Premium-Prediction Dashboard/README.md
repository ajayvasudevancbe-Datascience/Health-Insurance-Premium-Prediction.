# AI HealthGuard - Health Insurance Premium Prediction

## Files

- `app.py` - Streamlit application
- `requirements.txt` - Python dependencies
- `health_insurance_model.pkl` - trained machine learning model (you must add this file)

## GitHub / Streamlit Cloud structure

Put these files in the root of your repository:

health-insurance-premium-prediction/
- app.py
- requirements.txt
- health_insurance_model.pkl
- README.md

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Important

The model file must be named exactly:

`health_insurance_model.pkl`

The saved model should accept these columns:

- age
- sex
- bmi
- children
- smoker
- region

If Streamlit Cloud reports `ModuleNotFoundError: No module named 'joblib'`,
make sure `requirements.txt` is committed to GitHub and redeploy/reboot the app.
