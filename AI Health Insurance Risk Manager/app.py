
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="AI HealthGuard",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------------------------
# SAMPLE / DATA LOADING
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Replace with your own CSV path if required.
    possible_files = [
        "health_insurance.csv",
        "insurance.csv",
        "data/health_insurance.csv",
        "data/insurance.csv"
    ]

    for file in possible_files:
        try:
            df = pd.read_csv(file)
            return df
        except FileNotFoundError:
            continue

    # Built-in sample data so the app can run immediately.
    rng = np.random.default_rng(42)
    n = 1200

    age = rng.integers(18, 65, n)
    sex = rng.choice(["male", "female"], n)
    bmi = np.round(rng.normal(28, 5, n).clip(16, 48), 1)
    children = rng.integers(0, 5, n)
    smoker = rng.choice(["yes", "no"], n, p=[0.20, 0.80])
    region = rng.choice(
        ["southwest", "southeast", "northwest", "northeast"], n
    )

    charges = (
        1500
        + age * 240
        + bmi * 120
        + children * 350
        + np.where(smoker == "yes", 18000, 0)
        + np.where(sex == "male", 300, 0)
        + rng.normal(0, 1800, n)
    ).clip(500)

    return pd.DataFrame({
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "charges": np.round(charges, 2)
    })


df = load_data()

# ---------------------------------------------------------
# COLUMN STANDARDIZATION
# ---------------------------------------------------------
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Common alternative names
rename_map = {}
for col in df.columns:
    if col in ["medical_cost", "insurance_cost", "expenses"]:
        rename_map[col] = "charges"

df = df.rename(columns=rename_map)

required = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]

missing = [c for c in required if c not in df.columns]

if missing:
    st.error(
        "Your CSV is missing these required columns: "
        + ", ".join(missing)
    )
    st.stop()

# Numeric cleaning
for col in ["age", "bmi", "children", "charges"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required).copy()

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
features = ["age", "sex", "bmi", "children", "smoker", "region"]
target = "charges"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

categorical_features = ["sex", "smoker", "region"]
numeric_features = ["age", "bmi", "children"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=250,
                random_state=42,
                max_depth=12,
                min_samples_leaf=2
            )
        )
    ]
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# ---------------------------------------------------------
# RISK LOGIC
# ---------------------------------------------------------
q1 = df["charges"].quantile(0.33)
q2 = df["charges"].quantile(0.66)

def get_risk(cost):
    if cost <= q1:
        return "Low Risk"
    elif cost <= q2:
        return "Medium Risk"
    return "High Risk"

df["risk_level"] = df["charges"].apply(get_risk)

def risk_color(risk):
    if risk == "High Risk":
        return "🔴"
    if risk == "Medium Risk":
        return "🟡"
    return "🟢"

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("🏥 AI HealthGuard")
st.sidebar.caption("AI Health Insurance Risk Manager")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🔮 Risk Prediction",
        "📊 Risk Dashboard",
        "🔍 Risk Analysis",
        "🤖 AI Risk Assistant"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("### Model Performance")
st.sidebar.metric("R² Score", f"{r2:.2f}")
st.sidebar.metric("MAE", f"${mae:,.0f}")

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🏥 AI HealthGuard")
st.subheader("AI-Powered Health Insurance Risk Manager")
st.markdown(
    "Predict insurance cost, classify customer risk, "
    "identify important risk factors, and generate actionable insights."
)

# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------
if page == "🏠 Overview":

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Average Cost", f"${df['charges'].mean():,.0f}")
    c3.metric(
        "High-Risk Customers",
        f"{(df['risk_level'] == 'High Risk').sum():,}"
    )
    c4.metric(
        "High-Risk %",
        f"{(df['risk_level'] == 'High Risk').mean() * 100:.1f}%"
    )

    st.markdown("### 🎯 Business Problem")

    st.write(
        "Insurance companies need to understand customer risk, "
        "estimate expected medical costs, and identify customers "
        "who may require additional risk assessment."
    )

    st.markdown("### 💡 Solution")

    st.write(
        "AI HealthGuard uses a Machine Learning model to estimate "
        "insurance charges from customer information and converts "
        "the prediction into Low, Medium, or High Risk categories."
    )

    st.markdown("### 🔄 Project Workflow")

    st.code(
        """Health Insurance Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Random Forest ML Model
        ↓
Insurance Cost Prediction
        ↓
Risk Classification
        ↓
AI Explanation & Recommendations
        ↓
Streamlit Dashboard""",
        language="text"
    )

# ---------------------------------------------------------
# RISK PREDICTION
# ---------------------------------------------------------
elif page == "🔮 Risk Prediction":

    st.header("🔮 Customer Risk Prediction")

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
            sorted(df["sex"].astype(str).unique())
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=27.0,
            step=0.1
        )

    with col2:
        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0
        )

        smoker = st.selectbox(
            "Smoker",
            sorted(df["smoker"].astype(str).unique())
        )

        region = st.selectbox(
            "Region",
            sorted(df["region"].astype(str).unique())
        )

    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    if st.button("🚀 Predict Risk", use_container_width=True):

        predicted_cost = float(model.predict(input_df)[0])
        risk = get_risk(predicted_cost)

        st.markdown("---")
        st.subheader("Prediction Result")

        c1, c2 = st.columns(2)

        c1.metric(
            "Estimated Insurance Cost",
            f"${predicted_cost:,.0f}"
        )

        c2.metric(
            "Risk Level",
            f"{risk_color(risk)} {risk}"
        )

        st.markdown("### 🔍 Risk Factors")

        factors = []

        if age >= 50:
            factors.append("Age is relatively high.")
        if bmi >= 30:
            factors.append("BMI is in the obese range.")
        elif bmi >= 25:
            factors.append("BMI is above the healthy range.")
        if smoker.lower() == "yes":
            factors.append("Smoking status is a major risk factor.")
        if children >= 3:
            factors.append("Number of dependents may increase expected cost.")

        if not factors:
            factors.append(
                "No major manually identified risk factor was detected."
            )

        for factor in factors:
            st.write("•", factor)

        st.markdown("### 💡 Recommendation")

        if risk == "High Risk":
            st.warning(
                "High-risk case. Consider additional underwriting review "
                "and closer risk monitoring."
            )
        elif risk == "Medium Risk":
            st.info(
                "Medium-risk case. Consider routine monitoring and "
                "preventive-risk assessment."
            )
        else:
            st.success(
                "Lower-risk case based on the current model prediction."
            )

# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------
elif page == "📊 Risk Dashboard":

    st.header("📊 Insurance Risk Dashboard")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Low Risk",
        f"{(df['risk_level'] == 'Low Risk').sum():,}"
    )

    c2.metric(
        "Medium Risk",
        f"{(df['risk_level'] == 'Medium Risk').sum():,}"
    )

    c3.metric(
        "High Risk",
        f"{(df['risk_level'] == 'High Risk').sum():,}"
    )

    col1, col2 = st.columns(2)

    with col1:
        risk_counts = (
            df["risk_level"]
            .value_counts()
            .reset_index()
        )
        risk_counts.columns = ["risk_level", "count"]

        fig = px.pie(
            risk_counts,
            names="risk_level",
            values="count",
            title="Risk Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df,
            x="charges",
            nbins=40,
            title="Insurance Cost Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            df,
            x="smoker",
            y="charges",
            title="Insurance Cost by Smoking Status"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df,
            x="age",
            y="charges",
            color="risk_level",
            hover_data=["bmi", "smoker"],
            title="Age vs Insurance Cost"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# RISK ANALYSIS
# ---------------------------------------------------------
elif page == "🔍 Risk Analysis":

    st.header("🔍 Risk Factor Analysis")

    col1, col2 = st.columns(2)

    with col1:
        smoker_summary = (
            df.groupby("smoker")["charges"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            smoker_summary,
            x="smoker",
            y="charges",
            title="Average Cost by Smoking Status",
            text_auto=".2f"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        region_summary = (
            df.groupby("region")["charges"]
            .mean()
            .reset_index()
            .sort_values("charges", ascending=False)
        )

        fig = px.bar(
            region_summary,
            x="region",
            y="charges",
            title="Average Cost by Region",
            text_auto=".2f"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Key Business Insights")

    smoker_avg = df.groupby("smoker")["charges"].mean()

    if "yes" in smoker_avg.index and "no" in smoker_avg.index:
        difference = smoker_avg["yes"] - smoker_avg["no"]
        st.write(
            f"• Average insurance cost difference between smokers "
            f"and non-smokers: **${difference:,.0f}**"
        )

    high_risk_bmi = df.loc[
        df["risk_level"] == "High Risk",
        "bmi"
    ].mean()

    st.write(
        f"• Average BMI among high-risk customers: **{high_risk_bmi:.1f}**"
    )

    high_risk_age = df.loc[
        df["risk_level"] == "High Risk",
        "age"
    ].mean()

    st.write(
        f"• Average age among high-risk customers: **{high_risk_age:.1f} years**"
    )

# ---------------------------------------------------------
# AI ASSISTANT
# ---------------------------------------------------------
elif page == "🤖 AI Risk Assistant":

    st.header("🤖 AI Risk Assistant")

    st.write(
        "Select a customer profile and receive a model-based "
        "risk explanation."
    )

    selected_age = st.slider(
        "Age",
        18,
        80,
        35
    )

    selected_bmi = st.slider(
        "BMI",
        15.0,
        50.0,
        27.0
    )

    selected_smoker = st.selectbox(
        "Smoking Status",
        sorted(df["smoker"].astype(str).unique()),
        key="assistant_smoker"
    )

    if st.button("🔎 Analyze Risk", use_container_width=True):

        assistant_input = pd.DataFrame([{
            "age": selected_age,
            "sex": df["sex"].mode()[0],
            "bmi": selected_bmi,
            "children": int(df["children"].median()),
            "smoker": selected_smoker,
            "region": df["region"].mode()[0]
        }])

        predicted = float(model.predict(assistant_input)[0])
        risk = get_risk(predicted)

        st.success(
            f"Predicted Risk: {risk_color(risk)} **{risk}**"
        )

        st.metric(
            "Estimated Insurance Cost",
            f"${predicted:,.0f}"
        )

        st.markdown("### AI Risk Explanation")

        explanations = []

        if selected_smoker.lower() == "yes":
            explanations.append(
                "Smoking is likely to be a major contributor to predicted cost."
            )

        if selected_bmi >= 30:
            explanations.append(
                "BMI is high and may contribute to increased expected medical cost."
            )
        elif selected_bmi >= 25:
            explanations.append(
                "BMI is above the healthy range and may increase risk."
            )

        if selected_age >= 50:
            explanations.append(
                "Higher age can be associated with higher expected healthcare costs."
            )

        if not explanations:
            explanations.append(
                "The selected profile does not contain the main manually "
                "identified risk factors used in this explanation."
            )

        for item in explanations:
            st.write("•", item)

        st.markdown("### Recommended Action")

        if risk == "High Risk":
            st.warning(
                "Perform additional risk assessment and monitor this customer closely."
            )
        elif risk == "Medium Risk":
            st.info(
                "Consider preventive health guidance and periodic risk monitoring."
            )
        else:
            st.success(
                "Maintain routine monitoring based on the current model assessment."
            )

st.markdown("---")
st.caption(
    "AI HealthGuard | Machine Learning + Streamlit | "
    "For demonstration and educational purposes"
)
