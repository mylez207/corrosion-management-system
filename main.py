import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    with open("xgboost_corrosion_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["label_encoder"], data["scaler"]

model, label_encoder, scaler = load_model()

# Define feature columns
FEATURE_COLS = [
    "Fe2O3_percent", "Fe_percent", "SiO2_percent", "SO3_percent",
    "Na2O_percent", "CaO_percent", "MnO_percent", "Al2O3_percent",
    "Cl_percent", "MgO_percent", "Cr2O3_percent", "Co2O3_percent"
]

# ---------------------------
# Helper Functions
# ---------------------------
def predict_sample(sample_df):
    sample_scaled = scaler.transform(sample_df)
    preds = model.predict(sample_scaled)
    labels = label_encoder.inverse_transform(preds)
    return labels

# ---------------------------
# Session State
# ---------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last_pred" not in st.session_state:
    st.session_state.last_pred = "Unknown"
if "last_generated_at" not in st.session_state:
    st.session_state.last_generated_at = None

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Model Testing"])

# ---------------------------
# Page 1: Dashboard
# ---------------------------
if page == "Data Overview":
    st.title("Corrosion Maintenance Management System")

    # Color coding for status card
    color_map = {
        "Low": "#4CAF50",      # Green
        "Moderate": "#FFC107", # Amber
        "High": "#FF5722",     # Orange
        "Severe": "#900F06"    # Red
    }
    bg_color = color_map.get(st.session_state.last_pred, "#9E9E9E")

    # Status card
    st.subheader("Live Corrosion Status")
    st.markdown(
        f"""
        <div style='padding:20px; border-radius:15px; background:{bg_color}; color:white; text-align:center;'>
            <h2>Corrosion Severity: {st.session_state.last_pred}</h2>
            <p style='margin:0;'>Last update: {st.session_state.last_generated_at if st.session_state.last_generated_at else '-'} </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    
    # If severe or high, prompt action
    if st.session_state.last_pred in ["Severe", "High"]:
        st.markdown("  ")
        if st.button("⚠️ Take Action Now", key="action_btn"):
            action_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.logs.append({
                "time": action_time,
                "severity": st.session_state.last_pred,
                "action": "User acknowledged and initiated maintenance"
            })
            st.success("Action logged successfully.")

    st.markdown(" ")

    # Logs section
    st.subheader("Logs")
    if st.session_state.logs:
        logs_df = pd.DataFrame(st.session_state.logs)
        st.dataframe(logs_df)
    else:
        st.info("No actions logged yet.")

# ---------------------------
# Page 2: Model Testing
# ---------------------------
elif page == "Model Testing":
    st.title("Test The  Model")
    st.write("Enter values manually or upload a file for predictions.")

    # Manual input
    st.subheader("Manual Input")
    manual_data = {}
    for col in FEATURE_COLS:
        manual_data[col] = st.number_input(col, value=0.0)

    if st.button("Predict from Manual Input"):
        manual_df = pd.DataFrame([manual_data])
        preds = predict_sample(manual_df)
        pred_label = preds[0]

        # Update dashboard state
        st.session_state.last_pred = pred_label
        st.session_state.last_generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.logs.append({
            "time": st.session_state.last_generated_at,
            "severity": pred_label,
            "action": "Manual input tested"
        })

        st.success(f"Predicted corrosion severity: {pred_label}")

   # File upload
    st.subheader("Upload File for Batch Testing")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        preds = predict_sample(user_df[FEATURE_COLS])
        user_df["Predicted_Severity"] = preds

        # Apply color styling
        def highlight_severity(val):
            color_map = {
                "Low": "background-color: #4CAF50; color: white;",
                "Moderate": "background-color: #FFC107; color: black;",
                "High": "background-color: #FF5722; color: white;",
                "Severe": "background-color: #F44336; color: white;"
            }
            return color_map.get(val, "")

        styled_df = user_df.style.applymap(highlight_severity, subset=["Predicted_Severity"])
        st.dataframe(styled_df)

        st.download_button("Download Predictions", user_df.to_csv(index=False), "predictions.csv")