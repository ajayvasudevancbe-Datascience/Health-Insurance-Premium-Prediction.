import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AI HealthGuard", page_icon="🏥", layout="wide")

st.title("🏥 AI HealthGuard")
st.subheader("AI Health Insurance Risk Manager")

model = joblib.load("health_insurance_model.pkl")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

with col2:
    children = st.number_input("Children", 0, 10, 0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox(
        "Region",
        ["southwest", "southeast", "northwest", "northeast"]
    )

if st.button("🔮 Predict Risk", use_container_width=True):
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    prediction = model.predict(input_data)[0]

    st.metric("Predicted Insurance Cost", f"${prediction:,.2f}")

    if prediction < 5000:
        st.success("🟢 Low Risk")
    elif prediction < 15000:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    st.subheader("Customer Information")
    st.dataframe(input_data, use_container_width=True)
