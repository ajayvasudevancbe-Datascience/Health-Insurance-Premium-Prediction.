from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="AI HealthGuard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "health_insurance_model.pkl"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🏥 AI HealthGuard")
st.subheader("AI Health Insurance Risk Manager")

st.markdown(
    """
    Predict an estimated health insurance premium based on
    customer demographic and lifestyle information.
    """
)

st.divider()


# --------------------------------------------------
# MODEL CHECK
# --------------------------------------------------
if model is None:
    st.error(
        "❌ Model file not found.\n\n"
        "Please make sure `health_insurance_model.pkl` "
        "is inside the same folder as `app.py`."
    )

    st.info(
        "Expected location:\n\n"
        "`AI Health Insurance Risk Manager/"
        "health_insurance_model.pkl`"
    )

    st.stop()


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
st.header("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"],
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
    )


with col2:
    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
    )

    smoker = st.selectbox(
        "Smoking Status",
        ["yes", "no"],
    )

    region = st.selectbox(
        "Region",
        [
            "southwest",
            "southeast",
            "northwest",
            "northeast",
        ],
    )


st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button(
    "🔮 Predict Insurance Premium",
    use_container_width=True,
    type="primary",
):

    input_data = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region],
        }
    )

    try:

        prediction = model.predict(input_data)[0]

        prediction = float(prediction)

        # ------------------------------------------
        # PREMIUM
        # ------------------------------------------
        st.success("✅ Prediction completed successfully!")

        st.metric(
            label="Estimated Insurance Premium",
            value=f"${prediction:,.2f}",
        )

        # ------------------------------------------
        # RISK CLASSIFICATION
        # ------------------------------------------
        st.header("⚠️ Risk Category")

        if prediction < 5000:

            risk = "Low Risk"
            st.success("🟢 Low Risk Customer")

        elif prediction < 15000:

            risk = "Medium Risk"
            st.warning("🟡 Medium Risk Customer")

        else:

            risk = "High Risk"
            st.error("🔴 High Risk Customer")

        # ------------------------------------------
        # CUSTOMER DETAILS
        # ------------------------------------------
        st.divider()

        st.header("📋 Customer Details")

        result_data = pd.DataFrame(
            {
                "Feature": [
                    "Age",
                    "Sex",
                    "BMI",
                    "Children",
                    "Smoker",
                    "Region",
                    "Estimated Premium",
                    "Risk Category",
                ],
                "Value": [
                    age,
                    sex,
                    bmi,
                    children,
                    smoker,
                    region,
                    f"${prediction:,.2f}",
                    risk,
                ],
            }
        )

        st.dataframe(
            result_data,
            use_container_width=True,
            hide_index=True,
        )

        # ------------------------------------------
        # SUMMARY
        # ------------------------------------------
        st.header("🧠 AI HealthGuard Summary")

        st.write(
            f"""
            Based on the provided customer information, the estimated
            insurance premium is **${prediction:,.2f}**.

            The customer is classified as **{risk}** based on the
            predicted premium threshold.
            """
        )

    except Exception as error:

        st.error("❌ Prediction failed.")

        st.code(str(error))

        st.info(
            "Make sure the saved model was trained using these "
            "same input features: age, sex, bmi, children, "
            "smoker and region."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "AI HealthGuard | Built with Python, "
    "Scikit-learn and Streamlit"
)
