from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR.parent / "insurance_prediction.csv"
MODEL_PATH = BASE_DIR / "health_insurance_model.pkl"


# --------------------------------------------------
# CHECK DATASET
# --------------------------------------------------
if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Rows: {data.shape[0]}")
print(f"Columns: {data.shape[1]}")


# --------------------------------------------------
# CHECK TARGET
# --------------------------------------------------
if "charges" not in data.columns:

    raise ValueError(
        "Target column 'charges' was not found "
        "in the dataset."
    )


# --------------------------------------------------
# FEATURES / TARGET
# --------------------------------------------------
X = data[
    [
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region",
    ]
]

y = data["charges"]


# --------------------------------------------------
# FEATURES
# --------------------------------------------------
categorical_features = [
    "sex",
    "smoker",
    "region",
]

numeric_features = [
    "age",
    "bmi",
    "children",
]


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features,
        ),
        (
            "numeric",
            "passthrough",
            numeric_features,
        ),
    ]
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# --------------------------------------------------
# TRAIN
# --------------------------------------------------
print("Training Random Forest model...")

model.fit(X, y)

print("Model training completed.")


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------
joblib.dump(
    model,
    MODEL_PATH,
)

print()
print("====================================")
print("✅ MODEL CREATED SUCCESSFULLY")
print("====================================")
print(f"Saved to:")
print(MODEL_PATH)