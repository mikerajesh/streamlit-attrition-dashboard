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

st.subheader("Predict Employee Resignation Risk")
    st.write("Fill in the employee details below to predict their probability of leaving.")

    try:
        # File paths updated to include 'dashboard/models/' based on your GitHub link
        selected_features = joblib.load('dashboard/models/selected_features.pkl')
        scaler = joblib.load('dashboard/models/scaler.pkl') 
        model = joblib.load('dashboard/models/improved_model.pkl') 
        
        # Strip out the target index 'Attrition' if it accidentally got saved in your features list
        if hasattr(selected_features, 'tolist'):
            feature_list = selected_features.tolist()
        else:
            feature_list = list(selected_features)
            
        if 'Attrition' in feature_list:
            feature_list.remove('Attrition')

        # 1. Dynamically generate inputs only for features required by your model
        st.write("### Employee Profile Inputs")
        input_data = {}
        
        # Split inputs into clean columns
        col1, col2 = st.columns(2)
        
        for idx, col in enumerate(feature_list):
            # Place inputs alternatingly in column 1 and column 2
            target_col = col1 if idx % 2 == 0 else col2
            
            with target_col:
                if df[col].dtype == 'object':
                    # Automatically generate a dropdown matching original category values
                    input_data[col] = st.selectbox(f"Select {col}", sorted(df[col].unique()))
                elif df[col].nunique() <= 5:
                    # Use a slider for low-range rankings (like JobSatisfaction)
                    input_data[col] = st.slider(f"{col} Rating", int(df[col].min()), int(df[col].max()), int(df[col].median()))
                else:
                    # Use numeric input boxes for ages/salary lines
                    input_data[col] = st.number_input(f"Enter {col}", int(df[col].min()), int(df[col].max()), int(df[col].median()))

        # 2. Predict Button Action
        if st.button("Run Attrition Risk Analysis"):
            input_df = pd.DataFrame([input_data])
            
            # Reorder fields to match training layout
            input_df = input_df[feature_list]
            
            # Safe Fallback Encoder Transformation (Uses live mappings directly from dataset)
            for col in input_df.columns:
                if df[col].dtype == 'object':
                    # Create a quick dictionary mapping text options to alphabetical index order
                    mapping = {val: i for i, val in enumerate(sorted(df[col].unique()))}
                    input_df[col] = input_df[col].map(mapping)
            
            # Apply your saved scaling model array
            input_scaled = scaler.transform(input_df)
            
            # Run the Balanced Logistic Regression Prediction
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0][1] * 100
            
            # Output Predictive Results
            st.write("---")
            st.subheader("🎯 Predictive Output Results")
            
            if prediction[0] == 1 or probability > 50:
                st.error(f"⚠️ High Risk: Employee has a **{probability:.1f}%** probability of leaving.")
            else:
                st.success(f"✅ Low Risk: Employee has a **{probability:.1f}%** probability of staying (Leaving Risk: {probability:.1f}%).")
                
    except FileNotFoundError as e:
        st.warning("⚠️ File path configuration error. Streamlit cannot find your model assets.")
        st.info(f"Looking inside directory path: {str(e)}")

