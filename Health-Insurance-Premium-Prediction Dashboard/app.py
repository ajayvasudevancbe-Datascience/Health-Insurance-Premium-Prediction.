import os
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
st.write("Predict estimated health insurance cost using a machine learning model.")

MODEL_FILE = "health_insurance_model.pkl"

if not os.path.exists(MODEL_FILE):
    st.error(
        f"Model file '{MODEL_FILE}' was not found. "
        "Upload health_insurance_model.pkl to the same folder as app.py."
    )
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    st.error("Unable to load the machine learning model.")
    st.code(str(e))
    st.stop()

st.divider()
st.header("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

with col2:
    children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox(
        "Region",
        ["southwest", "southeast", "northwest", "northeast"]
    )

st.divider()

if st.button("🔮 Predict Insurance Premium", use_container_width=True):
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    try:
        prediction = float(model.predict(input_data)[0])

        st.subheader("💰 Predicted Insurance Cost")
        st.metric("Estimated Premium", f"${prediction:,.2f}")

        if prediction < 5000:
            st.success("🟢 Low Risk")
        elif prediction < 15000:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

        st.subheader("📋 Customer Information")
        st.dataframe(input_data, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
        st.info(
            "Make sure the saved model was trained using the same six input "
            "features: age, sex, bmi, children, smoker, region."
        )

st.divider()
st.caption("AI HealthGuard | Health Insurance Premium Prediction")
