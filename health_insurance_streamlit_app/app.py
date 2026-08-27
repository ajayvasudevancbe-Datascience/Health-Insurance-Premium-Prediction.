import streamlit as st
import pandas as pd
import joblib

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AI HealthGuard",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🏥 AI HealthGuard")
st.subheader("AI Health Insurance Risk Manager")

st.write(
    "Enter customer information to predict the insurance risk."
)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("health_insurance_model.pkl")

# ============================================================
# USER INPUT
# ============================================================

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )


with col2:

    children = st.number_input(
        "Children",
        min_value=0,
        max_value=10,
        value=0
    )

    smoker = st.selectbox(
        "Smoker",
        ["yes", "no"]
    )

    region = st.selectbox(
        "Region",
        [
            "southwest",
            "southeast",
            "northwest",
            "northeast"
        ]
    )


# ============================================================
# CREATE INPUT DATA
# ============================================================

input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Risk",
    use_container_width=True
):

    prediction = model.predict(input_data)

    insurance_cost = prediction[0]

    # ========================================================
    # RISK LEVEL
    # ========================================================

    if insurance_cost < 5000:

        risk = "Low Risk"

    elif insurance_cost < 15000:

        risk = "Medium Risk"

    else:

        risk = "High Risk"


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown("---")

    st.header("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Insurance Cost",
            f"${insurance_cost:,.2f}"
        )

    with col2:

        if risk == "High Risk":

            st.error("🔴 HIGH RISK")

        elif risk == "Medium Risk":

            st.warning("🟡 MEDIUM RISK")

        else:

            st.success("🟢 LOW RISK")


    # ========================================================
    # CUSTOMER DETAILS
    # ========================================================

    st.subheader("Customer Information")

    st.dataframe(
        input_data,
        use_container_width=True
    )

