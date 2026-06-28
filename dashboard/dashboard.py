import streamlit as st
import pandas as pd

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
# Monitoring Metrics (Q5)
# =========================

st.subheader("Monitoring Metrics")

col1, col2 = st.columns(2)

# Business Metric
with col1:
    st.metric(
        "Attrition Rate",
        f"{attrition_rate:.2f}%"
    )

# Data Quality Metric
missing_values = df.isnull().sum().sum()

with col2:
    st.metric(
        "Missing Values",
        int(missing_values)
    )

st.caption(
    "These monitoring metrics help HR monitor employee turnover and dataset quality after dashboard deployment."
)

# =========================
# Simple Data Drift Analysis
# =========================

st.subheader("Simple Data Drift Analysis")

original_df = load_data()

overall_income = original_df["MonthlyIncome"].mean()
filtered_income = df["MonthlyIncome"].mean()

difference = abs(overall_income - filtered_income)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Overall Mean Income",
        f"${overall_income:,.0f}"
    )

with col2:
    st.metric(
        "Filtered Mean Income",
        f"${filtered_income:,.0f}"
    )

with col3:
    st.metric(
        "Difference",
        f"${difference:,.0f}"
    )

st.caption(
    "A large difference between the original dataset and filtered data may indicate data drift."
)
