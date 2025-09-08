import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import os
from email_config import send_email_notification as send_email_notification_real

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    try:
        with open("logistic_model_scaled.pkl", "rb") as f:
            data = pickle.load(f)
        return data["model"], data["label_encoder"], data["scaler"]
    except FileNotFoundError:
        st.error("Model file 'logistic_model_scaled.pkl' not found. Please ensure the file exists in the current directory.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Try to load model
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
def generate_random_sample(df, n_samples=1, feature_cols=FEATURE_COLS, random_state=None):
    """Generate random samples within the range of the dataset"""
    rng = np.random.default_rng(random_state)
    samples = {}
    for col in feature_cols:
        if col in df.columns:
            col_min, col_max = df[col].min(), df[col].max()
            samples[col] = rng.uniform(col_min, col_max, n_samples)
        else:
            # Default values if column not found
            samples[col] = rng.uniform(0, 10, n_samples)
    return pd.DataFrame(samples)

def predict_sample(sample_df):
    """Make predictions on sample data"""
    if model is None or scaler is None or label_encoder is None:
        return ["Unknown"] * len(sample_df)
    
    try:
        sample_scaled = scaler.transform(sample_df)
        preds = model.predict(sample_scaled)
        labels = label_encoder.inverse_transform(preds)
        return labels
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return ["Error"] * len(sample_df)

# Send email notification
def send_email_notification(to_email, severity):
    """Send email notification using email_config.py"""
    try:
        success = send_email_notification_real(to_email, severity)
        if success:
            st.success(f"📧 Email sent to: {to_email}")
            st.info(f"Subject: Corrosion Alert: {severity} Severity Detected")
        return success
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# ---------------------------
# Session State Initialization
# ---------------------------
def init_session_state():
    """Initialize session state variables"""
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "last_pred" not in st.session_state:
        st.session_state.last_pred = "Unknown"
    if "last_generated_at" not in st.session_state:
        st.session_state.last_generated_at = None
    if "engineer_email" not in st.session_state:
        st.session_state.engineer_email = ""
    if "action_taken" not in st.session_state:
        st.session_state.action_taken = False
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False

init_session_state()

# ---------------------------
# Main App Configuration
# ---------------------------
st.set_page_config(
    page_title="Corrosion Monitoring System",
    page_icon="🔧",
    layout="wide"
)

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Model Testing"])

# Auto-refresh toggle
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh (every 10s)", value=st.session_state.auto_refresh)
st.session_state.auto_refresh = auto_refresh

if st.sidebar.button("Clear Logs"):
    st.session_state.logs = []
    st.session_state.action_taken = False
    st.experimental_rerun()

# ---------------------------
# Page 1: Dashboard
# ---------------------------
if page == "Data Overview":
    st.title("🏭 Corrosion Maintenance Management System")

    # Try to load dataset
    df = None
    dataset_paths = [
        "data/corrosion_dataset_20250907_234135.csv",
        "corrosion_dataset_20250907_234135.csv",
        "data/corrosion_dataset.csv"
    ]
    
    for path in dataset_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Dataset loaded: {path}")
                break
        except Exception as e:
            continue
    
    if df is None:
        st.error("❌ Dataset not found. Please place your CSV file in one of these locations:")
        for path in dataset_paths:
            st.write(f"- {path}")
        
        # Create a mock dataset for demo purposes
        st.warning("Using mock data for demonstration...")
        np.random.seed(42)
        mock_data = {}
        for col in FEATURE_COLS:
            mock_data[col] = np.random.uniform(0, 10, 1000)
        df = pd.DataFrame(mock_data)
        df['severity'] = np.random.choice(['Low', 'Moderate', 'High', 'Severe'], 1000)

    if df is not None:
        # Generate a new reading
        sample = generate_random_sample(df)
        pred_label = predict_sample(sample)[0]
        
        # Only update if prediction changed or if it's the first run
        if (st.session_state.last_pred != pred_label or 
            st.session_state.last_generated_at is None):
            
            st.session_state.last_pred = pred_label
            st.session_state.last_generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Log incoming data
            st.session_state.logs.append({
                "time": st.session_state.last_generated_at,
                "severity": pred_label,
                "action": "Data received"
            })
            
            # Reset action taken if severity changed
            st.session_state.action_taken = False

    # Color coding for status card
    color_map = {
        "Low": "#4CAF50",      # Green
        "Moderate": "#FFC107", # Amber
        "High": "#FF5722",     # Orange
        "Severe": "#F44336",   # Red
        "Unknown": "#9E9E9E",  # Grey
        "Error": "#9E9E9E"     # Grey
    }
    bg_color = color_map.get(st.session_state.last_pred, "#9E9E9E")

    # Status card
    st.subheader("📊 Live Corrosion Status")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(
            f"""
            <div style='padding:20px; border-radius:15px; background:{bg_color}; color:white; text-align:center; margin-bottom:20px;'>
                <h2>🔍 Corrosion Severity: {st.session_state.last_pred}</h2>
                <p style='margin:0;'>Last update: {st.session_state.last_generated_at or 'Never'}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        if st.button("🔄 Refresh Now"):
            st.experimental_rerun()
    
    with col3:
        total_logs = len(st.session_state.logs)
        st.metric("Total Readings", total_logs)

    # Show sample data if available
    if df is not None and len(sample) > 0:
        with st.expander("📋 Current Sensor Readings"):
            sample_display = sample.round(3)
            st.dataframe(sample_display, use_container_width=True)

    # If severe or high, prompt action
    if st.session_state.last_pred in ["Severe", "High"]:
        if not st.session_state.action_taken:
            st.markdown("### ⚠️ Maintenance Action Required")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                engineer_email = st.text_input(
                    "Responsible Engineer Email", 
                    value=st.session_state.engineer_email,
                    placeholder="engineer@company.com"
                )
            
            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("📧 Send Maintenance Alert", type="primary"):
                    if engineer_email and "@" in engineer_email:
                        st.session_state.engineer_email = engineer_email
                        if send_email_notification(engineer_email, st.session_state.last_pred):
                            action_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.logs.append({
                                "time": action_time,
                                "severity": st.session_state.last_pred,
                                "action": f"Email alert sent to {engineer_email}"
                            })
                            st.session_state.action_taken = True
                            st.experimental_rerun()
                    else:
                        st.error("Please enter a valid email address.")
        else:
            st.success("✅ Maintenance action already taken. Email notification sent.")
    
    elif st.session_state.last_pred == "Moderate":
        st.info("ℹ️ Moderate corrosion detected. Monitor closely.")
    
    elif st.session_state.last_pred == "Low":
        st.success("✅ Corrosion levels are within acceptable range.")

    # Logs section
    st.markdown("---")
    st.subheader("📝 System Logs")
    
    if st.session_state.logs:
        # Show recent logs first
        recent_logs = st.session_state.logs[-10:]  # Last 10 entries
        logs_df = pd.DataFrame(recent_logs[::-1])  # Reverse to show newest first
        
        # Style the dataframe
        st.dataframe(
            logs_df,
            use_container_width=True,
            hide_index=True
        )
        
        if len(st.session_state.logs) > 10:
            st.caption(f"Showing latest 10 of {len(st.session_state.logs)} total logs")
    else:
        st.info("No system activities logged yet.")

    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(10)
        st.experimental_rerun()

# ---------------------------
# Page 2: Model Testing
# ---------------------------
elif page == "Model Testing":
    st.title("🧪 Test Logistic Regression Model")
    st.write("Generate random samples, enter values manually, or upload a file for predictions.")

    if model is None:
        st.error("❌ Model not loaded. Cannot perform testing.")
        st.stop()

    # Manual input
    st.subheader("✏️ Manual Input")
    
    with st.form("manual_input_form"):
        cols = st.columns(3)
        manual_data = {}
        
        for i, col in enumerate(FEATURE_COLS):
            with cols[i % 3]:
                manual_data[col] = st.number_input(
                    col.replace("_", " ").title(), 
                    value=0.0, 
                    step=0.01,
                    key=f"manual_{col}"
                )
        
        submitted = st.form_submit_button("🔍 Predict from Manual Input", type="primary")
        
        if submitted:
            manual_df = pd.DataFrame([manual_data])
            preds = predict_sample(manual_df)
            severity = preds[0]
            
            # Show prediction with color
            color = color_map.get(severity, "#9E9E9E")
            st.markdown(
                f"""
                <div style='padding:15px; border-radius:10px; background:{color}; color:white; text-align:center; margin:20px 0;'>
                    <h3>Predicted Corrosion Severity: {severity}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Random sample generation
    st.subheader("🎲 Generate Random Sample")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_samples = st.number_input("Number of samples", min_value=1, max_value=100, value=5)
        
        if st.button("Generate Random Samples"):
            chemical_data = pd.read_csv("data/chemical_analysis.csv") if os.path.exists("data/chemical_analysis.csv") else None
            df = pd.DataFrame(chemical_data) if chemical_data is not None else chemical_data
           
            if df is not None:
                random_samples = generate_random_sample(df, n_samples)
                preds = predict_sample(random_samples)
                random_samples["Predicted_Severity"] = preds
                st.session_state.random_results = random_samples
            else:
                st.error("No dataset available for generating random samples.")
    
    with col2:
        if hasattr(st.session_state, 'random_results'):
            st.write("**Generated Samples:**")
            
            # Apply color styling
            def highlight_severity(val):
                colors = {
                    "Low": "background-color: #4CAF50; color: white;",
                    "Moderate": "background-color: #FFC107; color: black;",
                    "High": "background-color: #FF5722; color: white;",
                    "Severe": "background-color: #F44336; color: white;"
                }
                return colors.get(val, "")
            
            styled_df = st.session_state.random_results.style.applymap(
                highlight_severity, 
                subset=["Predicted_Severity"]
            )
            st.dataframe(styled_df, use_container_width=True)

    # File upload
    st.markdown("---")
    st.subheader("📁 Upload File for Batch Testing")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=["csv"],
        help="Upload a CSV file with the required feature columns"
    )
    
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {user_df.shape}")
            
            # Check if required columns exist
            missing_cols = [col for col in FEATURE_COLS if col not in user_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.write("**Available columns:**", list(user_df.columns))
            else:
                # Make predictions
                preds = predict_sample(user_df[FEATURE_COLS])
                user_df["Predicted_Severity"] = preds
                
                # Show results with color styling
                def highlight_severity(val):
                    colors = {
                        "Low": "background-color: #4CAF50; color: white;",
                        "Moderate": "background-color: #FFC107; color: black;",
                        "High": "background-color: #FF5722; color: white;",
                        "Severe": "background-color: #F44336; color: white;"
                    }
                    return colors.get(val, "")
                
                styled_df = user_df.style.applymap(
                    highlight_severity, 
                    subset=["Predicted_Severity"]
                )
                
                st.subheader("🎯 Prediction Results")
                st.dataframe(styled_df, use_container_width=True)
                
                # Summary statistics
                severity_counts = user_df["Predicted_Severity"].value_counts()
                st.subheader("📊 Severity Distribution")
                st.bar_chart(severity_counts)
                
                # Download button
                csv = user_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

    # Model information
    with st.expander("ℹ️ Model Information"):
        st.write("**Required Features:**")
        for i, col in enumerate(FEATURE_COLS, 1):
            st.write(f"{i}. {col}")
        
        st.write("**Possible Predictions:**")
        st.write("- Low: Minimal corrosion, normal operations")
        st.write("- Moderate: Some corrosion present, monitor closely") 
        st.write("- High: Significant corrosion, schedule maintenance")
        st.write("- Severe: Critical corrosion, immediate action required")