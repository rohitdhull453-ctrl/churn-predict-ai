import json
import os
import sqlite3
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

LOGISTIC_MODEL_FILE = BASE_DIR / "logistic_churn_prediction_model.pkl"
DT_MODEL_FILE = BASE_DIR / "DT_churn_prediction_model.pkl"
KNN_MODEL_FILE = BASE_DIR / "KNN_churn_prediction_model.pkl"

COLUMNS_FILE = BASE_DIR / "columns.pkl"
SCALER_FILE = BASE_DIR / "standard_scaler.pkl"
DATA_FILE = BASE_DIR / "clean_data_100k.csv"

SQLITE_DATABASE_FILE = BASE_DIR / "churn_database.db"

MODEL_FILES = {
    "Logistic Regression": LOGISTIC_MODEL_FILE,
    "Decision Tree": DT_MODEL_FILE,
    "KNN": KNN_MODEL_FILE,
}


# ============================================================
# LOAD MODELS, SCALER, AND FEATURE COLUMNS
# ============================================================

@st.cache_resource
def load_assets():
    errors = []

    # Load columns.pkl
    try:
        saved_columns = joblib.load(COLUMNS_FILE)

        # columns.pkl was saved as X.columns, which is pandas.Index
        if isinstance(saved_columns, pd.Index):
            features = saved_columns.tolist()

        elif isinstance(
            saved_columns,
            (list, tuple, np.ndarray, pd.Series),
        ):
            features = list(saved_columns)

        else:
            features = []

        features = [str(feature) for feature in features]

        if not features:
            errors.append(
                "columns.pkl does not contain readable feature columns."
            )

    except Exception as error:
        features = []
        errors.append(
            f"Could not load columns.pkl: {error}"
        )

    # Load standard scaler
    try:
        scaler = joblib.load(SCALER_FILE)

    except Exception as error:
        scaler = None
        errors.append(
            f"Could not load standard_scaler.pkl: {error}"
        )

    return features, scaler, errors


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


@st.cache_data
def load_reference_data():
    dataframe = pd.read_csv(DATA_FILE)

    dataframe["income"] = pd.to_numeric(
        dataframe["income"],
        errors="coerce",
    )

    dataframe["total_spent"] = pd.to_numeric(
        dataframe["total_spent"],
        errors="coerce",
    )

    dataframe["last_purchase"] = pd.to_datetime(
        dataframe["last_purchase"],
        errors="coerce",
    )

    return dataframe


FEATURES, SCALER, ASSET_ERRORS = load_assets()


# ============================================================
# DATABASE
# ============================================================

def get_database_url():
    """
    Reads DATABASE_URL from Streamlit secrets after deployment.
    Falls back to environment variable for local use.
    """

    try:
        database_url = st.secrets.get(
            "DATABASE_URL",
            "",
        )
    except Exception:
        database_url = ""

    return str(
        database_url
        or os.getenv("DATABASE_URL", "")
    ).strip()


def get_database_connection():
    """
    PostgreSQL is used when DATABASE_URL exists.
    SQLite is used locally when no DATABASE_URL is configured.
    """

    database_url = get_database_url()

    if database_url:

        try:
            import psycopg2

            connection = psycopg2.connect(
                database_url
            )

            return "postgres", connection

        except Exception as error:
            raise RuntimeError(
                f"Could not connect to PostgreSQL database: {error}"
            )

    connection = sqlite3.connect(
        SQLITE_DATABASE_FILE,
        check_same_thread=False,
    )

    return "sqlite", connection


def initialize_database():
    database_type, connection = get_database_connection()
    cursor = connection.cursor()

    if database_type == "postgres":

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                model_name TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probability DOUBLE PRECISION NOT NULL,
                input_data JSONB NOT NULL
            )
            """
        )

    else:

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL,
                input_data TEXT NOT NULL
            )
            """
        )

    connection.commit()

    return database_type, connection


def save_prediction(
    model_name,
    prediction,
    probability,
    customer_data,
):
    database_type, connection = initialize_database()
    cursor = connection.cursor()

    input_json = json.dumps(
        customer_data,
        default=str,
    )

    try:
        if database_type == "postgres":

            cursor.execute(
                """
                INSERT INTO predictions
                (
                    model_name,
                    prediction,
                    probability,
                    input_data
                )
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (
                    model_name,
                    int(prediction),
                    float(probability),
                    input_json,
                ),
            )

        else:

            cursor.execute(
                """
                INSERT INTO predictions
                (
                    model_name,
                    prediction,
                    probability,
                    input_data
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    model_name,
                    int(prediction),
                    float(probability),
                    input_json,
                ),
            )

        connection.commit()

    finally:
        connection.close()


def get_prediction_history():
    _, connection = initialize_database()

    try:
        dataframe = pd.read_sql_query(
            """
            SELECT
                id,
                created_at,
                model_name,
                prediction,
                probability
            FROM predictions
            ORDER BY id DESC
            """,
            connection,
        )

        return dataframe

    finally:
        connection.close()


# ============================================================
# DATA PREPARATION
# ============================================================

def get_options(dataframe, column_name, fallback):
    if column_name not in dataframe.columns:
        return fallback

    values = (
        dataframe[column_name]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if values:
        return sorted(values)

    return fallback


def set_one_hot_value(
    input_dataframe,
    column_prefix,
    selected_value,
):
    """
    Example:
    gender + male = gender_male
    """

    feature_name = (
        f"{column_prefix}_{selected_value}"
    )

    if feature_name in input_dataframe.columns:
        input_dataframe.loc[0, feature_name] = 1.0


def prepare_model_input(customer_data):
    """
    Your models were trained using:
    1. pd.get_dummies()
    2. standard_scaler.pkl

    This function recreates the same input format.
    """

    input_dataframe = pd.DataFrame(
        0.0,
        index=[0],
        columns=FEATURES,
    )

    numeric_values = {
        "Unnamed: 0": 0,
        "age": customer_data["age"],
        "income": customer_data["income"],
        "orders": customer_data["orders"],
        "total_spent": customer_data["total_spent"],
        "discount": customer_data["discount"],
        "rating": customer_data["rating"],
        "website_visits": customer_data["website_visits"],
        "support_calls": customer_data["support_calls"],
        "spending_per_order": (
            customer_data["total_spent"]
            / customer_data["orders"]
        ),
        "purchase_year": (
            customer_data["last_purchase"].year
        ),
        "purchase_month": (
            customer_data["last_purchase"].month
        ),
    }

    for column_name, value in numeric_values.items():

        if column_name in input_dataframe.columns:

            input_dataframe.loc[
                0,
                column_name,
            ] = float(value)

    # Set one-hot encoded values
    set_one_hot_value(
        input_dataframe,
        "customer_id",
        customer_data["customer_id"],
    )

    set_one_hot_value(
        input_dataframe,
        "gender",
        customer_data["gender"],
    )

    set_one_hot_value(
        input_dataframe,
        "city",
        customer_data["city"],
    )

    set_one_hot_value(
        input_dataframe,
        "education",
        customer_data["education"],
    )

    set_one_hot_value(
        input_dataframe,
        "category",
        customer_data["category"],
    )

    set_one_hot_value(
        input_dataframe,
        "payment",
        customer_data["payment"],
    )

    set_one_hot_value(
        input_dataframe,
        "membership",
        customer_data["membership"],
    )

    set_one_hot_value(
        input_dataframe,
        "age_group",
        customer_data["age_group"],
    )

    set_one_hot_value(
        input_dataframe,
        "last_purchase",
        customer_data["last_purchase"].strftime(
            "%Y-%m-%d"
        ),
    )

    # Apply the scaler used while training
    scaled_data = SCALER.transform(
        input_dataframe
    )

    return scaled_data


def make_prediction(model, prepared_input):
    prediction = int(
        np.asarray(
            model.predict(prepared_input)
        ).reshape(-1)[0]
    )

    probability = float(prediction)

    if hasattr(model, "predict_proba"):

        probabilities = np.asarray(
            model.predict_proba(prepared_input)
        )

        classes = list(
            getattr(model, "classes_", [0, 1])
        )

        if 1 in classes:

            probability = float(
                probabilities[0, classes.index(1)]
            )

        else:

            probability = float(
                probabilities[0, -1]
            )

    return prediction, probability


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ ChurnGuard AI")
    st.caption("Customer Churn Prediction")

    page = st.radio(
        "Navigation",
        [
            "🔮 Predict Churn",
            "📊 Prediction History",
            "⚙️ System Status",
        ],
    )

    st.divider()

    available_models = [
        model_name
        for model_name, model_path in MODEL_FILES.items()
        if model_path.exists()
    ]

    if available_models:

        selected_model_name = st.selectbox(
            "Select Model",
            available_models,
        )

    else:

        selected_model_name = None

    st.divider()

    for model_name, model_path in MODEL_FILES.items():

        if model_path.exists():
            st.success(f"✓ {model_name}")

        else:
            st.error(f"✗ {model_name}")


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ Customer Churn Prediction")

st.caption(
    "Predict customer churn using your trained machine-learning models."
)

st.divider()


# ============================================================
# PREDICTION PAGE
# ============================================================

if page == "🔮 Predict Churn":

    if ASSET_ERRORS or not FEATURES or SCALER is None:

        st.error(
            "Required model files could not be loaded."
        )

        for error in ASSET_ERRORS:
            st.warning(error)

        st.stop()

    if not selected_model_name:

        st.error(
            "No model file was found."
        )

        st.stop()

    if not DATA_FILE.exists():

        st.error(
            "Missing clean_data_100k.csv."
        )

        st.stop()

    reference_data = load_reference_data()

    customer_ids = get_options(
        reference_data,
        "customer_id",
        ["C197414"],
    )

    genders = get_options(
        reference_data,
        "gender",
        ["male", "female"],
    )

    cities = get_options(
        reference_data,
        "city",
        ["Hyderabad"],
    )

    educations = get_options(
        reference_data,
        "education",
        ["Bachelor"],
    )

    categories = get_options(
        reference_data,
        "category",
        ["Sports"],
    )

    payments = get_options(
        reference_data,
        "payment",
        ["Credit Card"],
    )

    memberships = get_options(
        reference_data,
        "membership",
        ["Basic"],
    )

    st.subheader("Customer Information")

    with st.form("churn_prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            customer_id = st.selectbox(
                "Customer ID",
                customer_ids,
            )

            age = st.number_input(
                "Age",
                min_value=18.0,
                max_value=100.0,
                value=30.0,
            )

            gender = st.selectbox(
                "Gender",
                genders,
            )

            city = st.selectbox(
                "City",
                cities,
            )

            education = st.selectbox(
                "Education",
                educations,
            )

        with col2:

            income = st.number_input(
                "Income",
                min_value=0.0,
                value=44227.0,
            )

            orders = st.number_input(
                "Orders",
                min_value=1.0,
                value=8.0,
                step=1.0,
            )

            total_spent = st.number_input(
                "Total Spent",
                min_value=0.0,
                value=32936.78,
            )

            category = st.selectbox(
                "Product Category",
                categories,
            )

            payment = st.selectbox(
                "Payment Method",
                payments,
            )

        with col3:

            discount = st.number_input(
                "Discount",
                min_value=0.0,
                value=9.2,
            )

            rating = st.number_input(
                "Rating",
                min_value=0.0,
                max_value=5.0,
                value=3.9,
            )

            website_visits = st.number_input(
                "Website Visits",
                min_value=0.0,
                value=13.0,
                step=1.0,
            )

            support_calls = st.number_input(
                "Support Calls",
                min_value=0.0,
                value=1.0,
                step=1.0,
            )

            membership = st.selectbox(
                "Membership",
                memberships,
            )

        last_purchase = st.date_input(
            "Last Purchase Date",
            value=date(2024, 9, 14),
        )

        age_group = st.selectbox(
            "Age Group",
            [
                "young",
                "adult",
                "middle_aged",
            ],
        )

        predict_button = st.form_submit_button(
            "🚀 Predict Customer Churn",
            type="primary",
            use_container_width=True,
        )

    if predict_button:

        customer_data = {
            "customer_id": customer_id,
            "age": age,
            "gender": gender,
            "city": city,
            "education": education,
            "income": income,
            "orders": orders,
            "total_spent": total_spent,
            "category": category,
            "payment": payment,
            "discount": discount,
            "rating": rating,
            "website_visits": website_visits,
            "support_calls": support_calls,
            "last_purchase": last_purchase,
            "membership": membership,
            "age_group": age_group,
        }

        try:

            model = load_model(
                str(
                    MODEL_FILES[
                        selected_model_name
                    ]
                )
            )

            prepared_input = prepare_model_input(
                customer_data
            )

            prediction, probability = make_prediction(
                model,
                prepared_input,
            )

            # Save result in database
            save_prediction(
                selected_model_name,
                prediction,
                probability,
                customer_data,
            )

            if prediction == 1:

                st.error(
                    "⚠️ High Churn Risk — customer is predicted to CHURN."
                )

            else:

                st.success(
                    "✅ Low Churn Risk — customer is predicted to STAY."
                )

            result_1, result_2, result_3 = st.columns(3)

            result_1.metric(
                "Churn Probability",
                f"{probability:.1%}",
            )

            result_2.metric(
                "Prediction",
                "CHURN"
                if prediction == 1
                else "STAY",
            )

            result_3.metric(
                "Model",
                selected_model_name,
            )

        except Exception as error:

            st.error(
                "Prediction or database save failed."
            )

            with st.expander(
                "Show Technical Error"
            ):
                st.exception(error)


# ============================================================
# PREDICTION HISTORY PAGE
# ============================================================

elif page == "📊 Prediction History":

    st.subheader("📊 Saved Prediction History")

    try:

        history = get_prediction_history()

        if history.empty:

            st.info(
                "No predictions have been saved yet."
            )

        else:

            total_predictions = len(history)

            churn_predictions = int(
                (
                    history["prediction"] == 1
                ).sum()
            )

            average_probability = float(
                history["probability"].mean()
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Predictions",
                total_predictions,
            )

            col2.metric(
                "Predicted Churn",
                churn_predictions,
            )

            col3.metric(
                "Average Probability",
                f"{average_probability:.1%}",
            )

            display_history = history.copy()

            display_history["prediction"] = (
                display_history["prediction"]
                .map({
                    0: "Stay",
                    1: "Churn",
                })
            )

            display_history["probability"] = (
                display_history["probability"]
                .apply(
                    lambda value:
                    f"{float(value):.1%}"
                )
            )

            st.dataframe(
                display_history,
                use_container_width=True,
                hide_index=True,
            )

    except Exception as error:

        st.error(
            f"Could not read database: {error}"
        )


# ============================================================
# SYSTEM STATUS PAGE
# ============================================================

else:

    st.subheader("⚙️ System Status")

    status_data = []

    for model_name, model_path in MODEL_FILES.items():

        status_data.append(
            {
                "Component": model_name,
                "File": model_path.name,
                "File Exists": model_path.exists(),
            }
        )

    status_data.extend(
        [
            {
                "Component": "Feature Columns",
                "File": COLUMNS_FILE.name,
                "File Exists": COLUMNS_FILE.exists(),
            },
            {
                "Component": "Standard Scaler",
                "File": SCALER_FILE.name,
                "File Exists": SCALER_FILE.exists(),
            },
            {
                "Component": "Dataset",
                "File": DATA_FILE.name,
                "File Exists": DATA_FILE.exists(),
            },
        ]
    )

    st.dataframe(
        pd.DataFrame(status_data),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        f"Loaded {len(FEATURES):,} features from columns.pkl."
    )

    if get_database_url():

        st.success(
            "PostgreSQL database is configured."
        )

    else:

        st.warning(
            "Using local SQLite. Add DATABASE_URL before deployment for permanent prediction history."
        )

    for error in ASSET_ERRORS:
        st.warning(error)