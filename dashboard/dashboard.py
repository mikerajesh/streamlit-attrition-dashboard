import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("Employee Attrition Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

df = load_data()

# =========================
# Interactive Feature 1
# =========================

department = st.sidebar.selectbox(
    "Select Department",
    ["All"] + sorted(df["Department"].unique())
)

if department != "All":
    df = df[df["Department"] == department]

# =========================
# Interactive Feature 2
# =========================

job_role = st.sidebar.selectbox(
    "Select Job Role",
    ["All"] + sorted(df["JobRole"].unique())
)

if job_role != "All":
    df = df[df["JobRole"] == job_role]

# =========================
# Visualization 1
# =========================

st.subheader("Attrition Distribution")

st.bar_chart(
    df["Attrition"].value_counts()
)

# =========================
# Visualization 2
# =========================

st.subheader("Overtime vs Attrition")

overtime = pd.crosstab(
    df["OverTime"],
    df["Attrition"]
)

st.bar_chart(overtime)

# =========================
# Visualization 3
# =========================

st.subheader("Job Satisfaction Distribution")

job_sat = (
    df["JobSatisfaction"]
    .value_counts()
    .sort_index()
)

st.bar_chart(job_sat)

# =========================
# Analytical Output
# =========================

attrition_rate = (
    (df["Attrition"] == "Yes")
    .mean()
    * 100
)

st.subheader("Current Attrition Rate")

st.metric(
    "Attrition Rate",
    f"{attrition_rate:.2f}%"
)

# =========================
# Data Table
# =========================

st.subheader("Filtered Employee Records")

st.dataframe(df)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

st.title("👨‍💼 Employee Attrition Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

df = load_data()

# =========================
# SAFE MODEL LOADING (IMPORTANT FIX)
# =========================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "models/improved_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
selected_features = joblib.load(os.path.join(BASE_DIR, "models/selected_features.pkl"))

feature_list = (
    selected_features.tolist()
    if hasattr(selected_features, "tolist")
    else list(selected_features)
)

if "Attrition" in feature_list:
    feature_list.remove("Attrition")

# =========================
# ENCODING FUNCTION
# =========================
cat_cols = df.select_dtypes(include="object").columns

def encode_input(input_df):
    df_encoded = input_df.copy()

    for col in cat_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype("category").cat.codes

    return df_encoded

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Filters")

department = st.sidebar.selectbox(
    "Select Department",
    ["All"] + sorted(df["Department"].unique())
)

if department != "All":
    df = df[df["Department"] == department]

job_role = st.sidebar.selectbox(
    "Select Job Role",
    ["All"] + sorted(df["JobRole"].unique())
)

if job_role != "All":
    df = df[df["JobRole"] == job_role]

# =========================
# VISUALIZATIONS
# =========================
st.subheader("📊 Attrition Distribution")
st.bar_chart(df["Attrition"].value_counts())

st.subheader("⏱ OverTime vs Attrition")
st.bar_chart(pd.crosstab(df["OverTime"], df["Attrition"]))

st.subheader("🙂 Job Satisfaction Distribution")
st.bar_chart(df["JobSatisfaction"].value_counts().sort_index())

# =========================
# METRIC
# =========================
attrition_rate = (df["Attrition"] == "Yes").mean() * 100
st.metric("Attrition Rate", f"{attrition_rate:.2f}%")

# =========================
# PREDICTION SECTION
# =========================
st.markdown("---")
st.subheader("🤖 Predict Employee Attrition Risk")
st.write("Fill employee details below to predict likelihood of leaving.")

input_data = {}

col1, col2 = st.columns(2)

for i, col in enumerate(feature_list):

    target_col = col1 if i % 2 == 0 else col2

    with target_col:

        if col in df.columns and df[col].dtype == "object":
            input_data[col] = st.selectbox(f"{col}", sorted(df[col].unique()))

        elif col in df.columns and df[col].nunique() <= 5:
            input_data[col] = st.slider(
                f"{col}",
                int(df[col].min()),
                int(df[col].max()),
                int(df[col].median())
            )

        elif col in df.columns:
            input_data[col] = st.number_input(
                f"{col}",
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )

# =========================
# PREDICTION BUTTON
# =========================
if st.button("Run Attrition Prediction"):

    input_df = pd.DataFrame([input_data])

    # ensure correct order
    input_df = input_df[feature_list]

    # encode categorical columns safely
    input_encoded = encode_input(input_df)

    # scale input
    input_scaled = scaler.transform(input_encoded)

    # prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk: Employee likely to leave ({probability:.1f}% probability)")
    else:
        st.success(f"✅ Low Risk: Employee likely to stay ({probability:.1f}% probability)")

# =========================
# DATA VIEW
# =========================
st.markdown("---")
st.subheader("📄 Filtered Dataset")
st.dataframe(df)
