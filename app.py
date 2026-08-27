import os
import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI HealthGuard",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: bold;
            text-align: center;
        }

        .sub-title {
            font-size: 20px;
            text-align: center;
            margin-bottom: 30px;
        }

        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🏥 AI HealthGuard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI Health Insurance Risk Manager</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# MODEL FILE
# ============================================================

MODEL_FILE = "health_insurance_model.pkl"


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not os.path.exists(MODEL_FILE):

    st.error(
        f"""
        ❌ Model file not found!

        Expected file:

        `{MODEL_FILE}`

        Please make sure `health_insurance_model.pkl`
        is uploaded to the same folder as `app.py`.
        """
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(MODEL_FILE)

except Exception as e:

    st.error("❌ Error while loading the machine learning model.")

    st.code(str(e))

    st.stop()


# ============================================================
# USER INPUT SECTION
# ============================================================

st.header("👤 Customer Information")

col1, col2 = st.columns(2)


# ============================================================
# COLUMN 1
# ============================================================

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


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    smoker = st.selectbox(
        "Smoker",
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


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Insurance Premium",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # DISPLAY INPUT DATA
    # --------------------------------------------------------

    st.subheader("📋 Customer Input")

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(input_data)

        prediction = float(prediction[0])


        # ----------------------------------------------------
        # DISPLAY PREMIUM
        # ----------------------------------------------------

        st.subheader("💰 Predicted Insurance Premium")

        st.metric(
            label="Estimated Insurance Cost",
            value=f"${prediction:,.2f}"
        )


        # ----------------------------------------------------
        # RISK CLASSIFICATION
        # ----------------------------------------------------

        st.subheader("🚦 Risk Level")


        if prediction < 5000:

            st.success(
                "🟢 LOW RISK\n\n"
                "The predicted insurance cost is relatively low."
            )


        elif prediction < 15000:

            st.warning(
                "🟡 MEDIUM RISK\n\n"
                "The predicted insurance cost is moderate."
            )


        else:

            st.error(
                "🔴 HIGH RISK\n\n"
                "The predicted insurance cost is relatively high."
            )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.subheader("📊 Prediction Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)


        with summary_col1:

            st.metric(
                "Age",
                age
            )


        with summary_col2:

            st.metric(
                "BMI",
                f"{bmi:.1f}"
            )


        with summary_col3:

            st.metric(
                "Children",
                children
            )


        summary_col4, summary_col5, summary_col6 = st.columns(3)


        with summary_col4:

            st.metric(
                "Smoker",
                smoker
            )


        with summary_col5:

            st.metric(
                "Sex",
                sex
            )


        with summary_col6:

            st.metric(
                "Region",
                region
            )


    # --------------------------------------------------------
    # PREDICTION ERROR
    # --------------------------------------------------------

    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.warning(
            """
            Your model may have been trained with different
            input columns or preprocessing.
            """
        )

        st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏥 AI HealthGuard | Health Insurance Premium Prediction "
    "using Machine Learning"
)