from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AI HealthGuard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AI HealthGuard")
st.subheader("AI Health Insurance Risk Manager")
st.markdown("---")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "health_insurance_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("❌ health_insurance_model.pkl file not found.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

with col2:
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox(
        "Region",
        ["southwest", "southeast", "northwest", "northeast"]
    )

st.markdown("---")

if st.button("🔮 Predict Insurance Risk", use_container_width=True):
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    prediction = model.predict(input_data)[0]

    st.success("Prediction Completed Successfully!")
    st.metric("Estimated Insurance Premium", f"${prediction:,.2f}")

    st.subheader("Risk Category")

    if prediction < 5000:
        st.success("🟢 Low Risk Customer")
    elif prediction < 15000:
        st.warning("🟡 Medium Risk Customer")
    else:
        st.error("🔴 High Risk Customer")

    st.markdown("---")
    st.subheader("Customer Details")
    st.dataframe(input_data, use_container_width=True)

    st.subheader("AI HealthGuard Risk Summary")
    st.write(f"**Age:** {age}")
    st.write(f"**BMI:** {bmi}")
    st.write(f"**Smoker:** {smoker}")
    st.write(f"**Region:** {region}")
    st.write(f"**Predicted Premium:** ${prediction:,.2f}")

st.markdown("---")
st.caption("Built with Python, Scikit-learn and Streamlit")
