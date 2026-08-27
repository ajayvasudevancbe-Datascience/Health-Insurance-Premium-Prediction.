from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI HealthGuard",
    page_icon="🏥",
    layout="wide"
)


# --------------------------------------------------
# MODEL PATH
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

    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

        return model

    except Exception as error:
        st.error("❌ Unable to load the machine learning model.")
        st.code(str(error))
        return None


model = load_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏥 AI HealthGuard")

st.subheader(
    "AI Health Insurance Risk Manager"
)

st.markdown(
    """
    ### 💡 Insurance Premium Prediction

    Enter customer information below to estimate
    the health insurance premium and identify the
    associated risk category.
    """
)

st.divider()


# --------------------------------------------------
# MODEL CHECK
# --------------------------------------------------

if model is None:

    st.error(
        "❌ Machine learning model is not available."
    )

    st.info(
        "Make sure this file exists beside app.py:"
    )

    st.code(
        "health_insurance_model.pkl"
    )

    st.stop()


# --------------------------------------------------
# CUSTOMER INPUT
# --------------------------------------------------

st.header("👤 Customer Information")


col1, col2 = st.columns(2)


# --------------------------------------------------
# LEFT COLUMN
# --------------------------------------------------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    sex = st.selectbox(
        "Sex",
        [
            "male",
            "female"
        ]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )


# --------------------------------------------------
# RIGHT COLUMN
# --------------------------------------------------

with col2:

    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    smoker = st.selectbox(
        "Smoking Status",
        [
            "yes",
            "no"
        ]
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


st.divider()


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button(
    "🔮 Predict Insurance Premium",
    use_container_width=True
):

    # ----------------------------------------------
    # CREATE INPUT DATAFRAME
    # ----------------------------------------------

    input_data = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region]
        }
    )


    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------

    try:

        prediction = model.predict(
            input_data
        )[0]

        prediction = float(prediction)


    except Exception as error:

        st.error(
            "❌ Prediction failed."
        )

        st.code(
            str(error)
        )

        st.info(
            """
            The model must be trained using these
            six features:

            age
            sex
            bmi
            children
            smoker
            region
            """
        )

        st.stop()


    # ----------------------------------------------
    # PREMIUM
    # ----------------------------------------------

    st.success(
        "✅ Prediction completed successfully!"
    )

    st.metric(
        "Estimated Insurance Premium",
        f"${prediction:,.2f}"
    )


    # ----------------------------------------------
    # RISK CATEGORY
    # ----------------------------------------------

    st.header("⚠️ Risk Category")


    if prediction < 5000:

        risk = "Low Risk"

        st.success(
            "🟢 Low Risk Customer"
        )


    elif prediction < 15000:

        risk = "Medium Risk"

        st.warning(
            "🟡 Medium Risk Customer"
        )


    else:

        risk = "High Risk"

        st.error(
            "🔴 High Risk Customer"
        )


    # ----------------------------------------------
    # CUSTOMER SUMMARY
    # ----------------------------------------------

    st.divider()

    st.header(
        "📋 Customer Summary"
    )


    result = pd.DataFrame(
        {
            "Feature": [
                "Age",
                "Sex",
                "BMI",
                "Children",
                "Smoker",
                "Region",
                "Estimated Premium",
                "Risk Category"
            ],

            "Value": [
                age,
                sex,
                bmi,
                children,
                smoker,
                region,
                f"${prediction:,.2f}",
                risk
            ]
        }
    )


    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )


    # ----------------------------------------------
    # AI SUMMARY
    # ----------------------------------------------

    st.header(
        "🧠 AI HealthGuard Summary"
    )

    st.write(
        f"""
        Based on the provided customer information,
        the estimated health insurance premium is
        **${prediction:,.2f}**.

        The calculated risk category is
        **{risk}**.
        """
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI HealthGuard | Python | Scikit-learn | Streamlit"
)