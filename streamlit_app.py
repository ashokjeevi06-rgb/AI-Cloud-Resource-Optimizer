
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="AI Cloud Resource Optimizer",
    page_icon="☁️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "cloud_workload_dataset.csv"
MODEL_PATH = BASE_DIR / "model" / "cloud_optimizer.pkl"

# --------------------------------------------------
# STYLE
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f3f6fb;
}
.header {
    background-color: #1e2d47;
    padding: 24px;
    border-radius: 12px;
    color: white;
}
.header h1 {
    color: #61c7ff;
}
.section {
    background: white;
    padding: 18px;
    border-radius: 12px;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


df = pd.DataFrame()
dataset_error = None

try:
    if DATASET_PATH.exists():
        df = load_dataset(str(DATASET_PATH))
    else:
        dataset_error = (
            f"Dataset file not found: {DATASET_PATH}"
        )
except Exception as exc:
    dataset_error = str(exc)

# --------------------------------------------------
# LOAD SAVED OBJECT
# --------------------------------------------------
@st.cache_resource
def load_saved_object(path):
    return joblib.load(path)


saved_object = None
model_load_error = None

try:
    if MODEL_PATH.exists():
        saved_object = load_saved_object(str(MODEL_PATH))
    else:
        model_load_error = f"Model file not found: {MODEL_PATH}"
except Exception as exc:
    model_load_error = str(exc)


# --------------------------------------------------
# FIND A PREDICTION MODEL INSIDE A DICTIONARY
# --------------------------------------------------
def find_estimator(obj, depth=0):
    """
    Find an object exposing predict() inside the saved object.
    Returns (estimator, dictionary_path).
    """
    if depth > 5:
        return None, None

    if callable(getattr(obj, "predict", None)):
        return obj, "saved object"

    if isinstance(obj, dict):
        # Check common model keys first.
        preferred_keys = [
            "model",
            "best_model",
            "classifier",
            "regressor",
            "estimator",
            "pipeline",
            "trained_model",
            "random_forest",
            "random_forest_model"
        ]

        ordered_keys = (
            [key for key in preferred_keys if key in obj]
            + [key for key in obj if key not in preferred_keys]
        )

        for key in ordered_keys:
            value = obj[key]
            result, location = find_estimator(value, depth + 1)

            if result is not None:
                return result, f"{key} -> {location}"

    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            result, location = find_estimator(value, depth + 1)

            if result is not None:
                return result, f"[{index}] -> {location}"

    return None, None


model, model_location = find_estimator(saved_object)


# --------------------------------------------------
# FIND FEATURE NAMES IN SAVED METADATA
# --------------------------------------------------
def find_saved_feature_names(obj, depth=0):
    if depth > 5:
        return None

    if isinstance(obj, dict):
        name_keys = {
            "feature_names",
            "features",
            "input_features",
            "feature_columns",
            "input_columns",
            "x_columns",
            "columns"
        }

        for key, value in obj.items():
            if (
                str(key).lower() in name_keys
                and isinstance(value, (list, tuple))
                and all(isinstance(item, str) for item in value)
            ):
                return list(value)

        for value in obj.values():
            result = find_saved_feature_names(value, depth + 1)

            if result is not None:
                return result

    elif isinstance(obj, (list, tuple)):
        for value in obj:
            result = find_saved_feature_names(value, depth + 1)

            if result is not None:
                return result

    return None


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="header">
    <h1>☁️ AI Cloud Resource Optimizer</h1>
    <p>Workload Prediction and Smart Resource Planning</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Dataset Records", len(df))
col2.metric("Dataset Columns", len(df.columns))
col3.metric(
    "AI Model Status",
    "Loaded" if model is not None else "Check Model"
)

if dataset_error:
    st.error(f"Dataset error: {dataset_error}")

if model_load_error:
    st.error(f"Model loading error: {model_load_error}")

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------
st.subheader("📊 Dataset Preview")

if not df.empty:
    st.dataframe(df.head(10), use_container_width=True)
else:
    st.warning("Dataset is empty or unavailable.")

# --------------------------------------------------
# HISTORICAL WORKLOAD CHART
# --------------------------------------------------
st.subheader("📈 Historical Workload Trend")

if not df.empty:
    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        preferred = [
            column for column in numeric_df.columns
            if any(
                term in str(column).lower()
                for term in ("cpu", "memory", "usage", "workload", "load")
            )
        ]

        chart_columns = preferred[:3] or list(numeric_df.columns[:3])

        st.line_chart(numeric_df[chart_columns].head(100))
        st.caption("Historical dataset values, not a future forecast.")
    else:
        st.info("No numeric columns are available for charting.")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.divider()
st.subheader("🤖 AI Workload Prediction")

features = []
feature_problem = None

if model is not None:
    model_names = getattr(model, "feature_names_in_", None)
    model_feature_count = getattr(model, "n_features_in_", None)

    if model_names is not None:
        features = list(model_names)

    elif model_feature_count is not None:
        metadata_features = find_saved_feature_names(saved_object)

        if (
            metadata_features is not None
            and len(metadata_features) == model_feature_count
        ):
            features = metadata_features
        else:
            # Numeric columns are a candidate only if their count matches.
            numeric_columns = list(
                df.select_dtypes(include="number").columns
            )

            if len(numeric_columns) == model_feature_count:
                features = numeric_columns
            else:
                feature_problem = (
                    f"The estimator expects {model_feature_count} input "
                    f"features, but the dataset contains "
                    f"{len(numeric_columns)} numeric columns. "
                    "The original training feature list is needed."
                )

    else:
        metadata_features = find_saved_feature_names(saved_object)

        if metadata_features:
            features = metadata_features
        else:
            feature_problem = (
                "A prediction model was found inside the saved file, "
                "but its expected input features are not recorded. "
                "The original training code is needed to identify them."
            )

if model is None:
    st.error(
        "No object with a predict() method was found inside "
        "cloud_optimizer.pkl."
    )

    if isinstance(saved_object, dict):
        st.write("Keys inside your saved file:", list(saved_object.keys()))
    elif saved_object is not None:
        st.write("Saved object type:", type(saved_object).__name__)

elif feature_problem:
    st.warning(feature_problem)

    st.write("Model type:", type(model).__name__)
    st.write("Model stored at:", model_location)
    st.write(
        "Expected feature count:",
        getattr(model, "n_features_in_", "Not recorded")
    )

    if isinstance(saved_object, dict):
        st.write("Saved file keys:", list(saved_object.keys()))

    st.info(
        "The dashboard is running, but no prediction is made until "
        "the correct training inputs are identified."
    )

elif not features:
    st.warning("No model input features could be identified.")

else:
    st.caption(f"Model located at: {model_location}")

    # Start with typical dataset values for each feature.
    defaults = []

    for feature in features:
        if feature in df.columns:
            values = pd.to_numeric(df[feature], errors="coerce").dropna()
            defaults.append(
                float(values.median()) if not values.empty else 0.0
            )
        else:
            defaults.append(0.0)

    with st.form("prediction_form"):
        st.write("Enter the values required by your trained model.")

        input_values = []
        input_columns = st.columns(2)

        for index, feature in enumerate(features):
            with input_columns[index % 2]:
                value = st.number_input(
                    str(feature),
                    value=float(defaults[index]),
                    key=f"feature_{index}"
                )
                input_values.append(value)

        submitted = st.form_submit_button(
            "🔍 Analyze Workload",
            use_container_width=True
        )

    if submitted:
        try:
            if getattr(model, "feature_names_in_", None) is not None:
                model_input = pd.DataFrame(
                    [input_values],
                    columns=features
                )
            else:
                model_input = [input_values]

            prediction = model.predict(model_input)[0]

            st.success(f"Predicted result: {prediction}")

            # Interpret the result only if it is numeric.
            try:
                number = float(prediction)

                st.caption(
                    "The following ranges are examples only. "
                    "Adjust them to match your trained target."
                )

                if number < 20:
                    st.info("Illustrative result: LOW")
                elif number < 60:
                    st.info("Illustrative result: NORMAL")
                else:
                    st.warning("Illustrative result: HIGH")

            except (ValueError, TypeError):
                st.info(
                    "This is the model's predicted category. "
                    "Interpret it using your training labels."
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# --------------------------------------------------
# CLOUD COST ESTIMATOR
# --------------------------------------------------
st.divider()
st.subheader("💰 Cloud Cost Estimator")

cost_col1, cost_col2 = st.columns(2)

with cost_col1:
    hourly_cost = st.number_input(
        "Estimated hourly cost (₹)",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

with cost_col2:
    savings_percent = st.slider(
        "Assumed savings (%)",
        min_value=0,
        max_value=100,
        value=10
    )

hourly_savings = hourly_cost * savings_percent / 100
adjusted_cost = hourly_cost - hourly_savings

metric1, metric2 = st.columns(2)

metric1.metric(
    "Estimated Hourly Savings",
    f"₹{hourly_savings:.2f}"
)

metric2.metric(
    "Estimated Hourly Cost After Savings",
    f"₹{adjusted_cost:.2f}"
)

st.caption(
    "These are illustrative calculations, not actual cloud billing "
    "or guaranteed savings. This application does not automatically "
    "change cloud resources."
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption("AI-Based Smart Cloud Resource Optimizer | Student Project")
