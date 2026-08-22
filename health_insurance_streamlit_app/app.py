import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="Health Insurance Premium Prediction",
    page_icon="🏥",
    layout="wide",
)

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/stedy/"
    "Machine-Learning-with-R-datasets/master/insurance.csv"
)

REQUIRED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]


@st.cache_data
def load_data(source):
    if source == "default":
        df = pd.read_csv(DEFAULT_DATA_URL)
    else:
        df = pd.read_csv(source)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def prepare_data(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    work = df[REQUIRED_COLUMNS].copy()
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    work["bmi"] = pd.to_numeric(work["bmi"], errors="coerce")
    work["children"] = pd.to_numeric(work["children"], errors="coerce")
    work["charges"] = pd.to_numeric(work["charges"], errors="coerce")
    work = work.dropna()

    encoded = pd.get_dummies(
        work,
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=int,
    )
    X = encoded.drop("charges", axis=1)
    y = encoded["charges"]
    return work, encoded, X, y


st.title("🏥 Health Insurance Premium Prediction")
st.caption("Linear Regression • Streamlit • Interactive ML Prediction App")

with st.sidebar:
    st.header("⚙️ Data Source")
    uploaded = st.file_uploader("Upload insurance CSV", type=["csv"])

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
    else:
        raw = load_data("default")

    st.info(
        "The model follows the notebook workflow: "
        "one-hot encoding with drop_first=True and a 70/30 train-test split."
    )

try:
    data, encoded, X, y = prepare_data(raw)
except Exception as e:
    st.error(f"Could not prepare the dataset: {e}")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "🔎 Data Explorer"])

with tab1:
    st.subheader("Predict an Individual's Insurance Charge")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
    with c2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=2)
        sex = st.selectbox("Sex", ["female", "male"])
    with c3:
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox(
            "Region",
            ["northeast", "northwest", "southeast", "southwest"]
        )

    input_df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region,
    }])

    input_encoded = pd.get_dummies(
        input_df,
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=int,
    )
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    if st.button("💰 Predict Insurance Charge", type="primary", use_container_width=True):
        prediction = float(model.predict(input_encoded)[0])
        st.success(f"### Estimated Insurance Charge: ${prediction:,.2f}")
        st.caption(
            "This is a machine-learning estimate, not a quotation or medical/financial advice."
        )

with tab2:
    st.subheader("Model Evaluation")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test R²", f"{test_r2:.3f}")
    m2.metric("Test RMSE", f"${test_rmse:,.0f}")
    m3.metric("Train R²", f"{train_r2:.3f}")
    m4.metric("Train RMSE", f"${train_rmse:,.0f}")

    st.markdown("### Actual vs Predicted Charges")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, test_pred, alpha=0.65)
    line_min = min(y_test.min(), test_pred.min())
    line_max = max(y_test.max(), test_pred.max())
    ax.plot([line_min, line_max], [line_min, line_max])
    ax.set_xlabel("Actual Charges")
    ax.set_ylabel("Predicted Charges")
    ax.set_title("Actual vs Predicted — Test Set")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Feature Coefficients")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=lambda s: s.abs(), ascending=False)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Dataset Overview")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(data):,}")
    b.metric("Features", "6 input features")
    c.metric("Target", "charges")

    st.dataframe(data.head(20), use_container_width=True, hide_index=True)

    st.markdown("### Charges Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data["charges"], bins=30)
    ax.set_xlabel("Insurance Charges")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Insurance Charges")
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.caption("Developed by Ajay Vasudevan • B.Sc. Data Science • Streamlit ML Project")
