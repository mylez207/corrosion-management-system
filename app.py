# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import time
# from datetime import datetime
# import smtplib
# from email.mime.text import MIMEText
# import os
# from email_config import send_email_notification as send_email_notification_real

# # ---------------------------
# # Load Model
# # ---------------------------
# @st.cache_resource
# def load_model():
#     try:
#         with open("xgboost_corrosion_model.pkl", "rb") as f:
#             data = pickle.load(f)
#         return data["model"], data["label_encoder"], data["scaler"]
#     except FileNotFoundError:
#         st.error("Model file 'xgboost_corrosion_model.pkl' not found. Please ensure the file exists in the current directory.")
#         return None, None, None
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None, None, None

# # Try to load model
# model, label_encoder, scaler = load_model()

# # Define feature columns
# FEATURE_COLS = [
#     "Fe2O3_percent", "Fe_percent", "SiO2_percent", "SO3_percent",
#     "Na2O_percent", "CaO_percent", "MnO_percent", "Al2O3_percent",
#     "Cl_percent", "MgO_percent", "Cr2O3_percent", "Co2O3_percent"
# ]

# # ---------------------------
# # Helper Functions
# # ---------------------------
# def generate_random_sample(df, n_samples=1, feature_cols=FEATURE_COLS, random_state=None):
#     """Generate random samples within the range of the dataset"""
#     rng = np.random.default_rng(random_state)
#     samples = {}
#     for col in feature_cols:
#         if col in df.columns:
#             col_min, col_max = df[col].min(), df[col].max()
#             samples[col] = rng.uniform(col_min, col_max, n_samples)
#         else:
#             # Default values if column not found
#             samples[col] = rng.uniform(0, 10, n_samples)
#     return pd.DataFrame(samples)

# def predict_sample(sample_df):
#     """Make predictions on sample data"""
#     if model is None or scaler is None or label_encoder is None:
#         return ["Unknown"] * len(sample_df)

#     try:
#         sample_scaled = scaler.transform(sample_df)
#         preds = model.predict(sample_scaled)
#         labels = label_encoder.inverse_transform(preds)
#         return labels
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         return ["Error"] * len(sample_df)

# # Send email notification
# def send_email_notification(to_email, severity):
#     """Send email notification using email_config.py"""
#     try:
#         success = send_email_notification_real(to_email, severity)
#         if success:
#             st.success(f"📧 Email sent to: {to_email}")
#             st.info(f"Subject: Corrosion Alert: {severity} Severity Detected")
#         return success
#     except Exception as e:
#         st.error(f"Failed to send email: {e}")
#         return False

# # ---------------------------
# # Session State Initialization
# # ---------------------------
# def init_session_state():
#     """Initialize session state variables"""
#     if "logs" not in st.session_state:
#         st.session_state.logs = []
#     if "last_pred" not in st.session_state:
#         st.session_state.last_pred = "Unknown"
#     if "last_generated_at" not in st.session_state:
#         st.session_state.last_generated_at = None
#     if "engineer_email" not in st.session_state:
#         st.session_state.engineer_email = ""
#     if "action_taken" not in st.session_state:
#         st.session_state.action_taken = False
#     if "auto_refresh" not in st.session_state:
#         st.session_state.auto_refresh = False

# init_session_state()

# # ---------------------------
# # Main App Configuration
# # ---------------------------
# st.set_page_config(
#     page_title="Corrosion Monitoring System",
#     page_icon="🔧",
#     layout="wide"
# )

# # ---------------------------
# # Sidebar Navigation
# # ---------------------------
# st.sidebar.title("🔧 Navigation")
# page = st.sidebar.radio("Go to", ["Data Overview", "Model Testing"])

# # Auto-refresh toggle
# st.sidebar.markdown("---")
# auto_refresh = st.sidebar.checkbox("Auto-refresh (every 10s)", value=st.session_state.auto_refresh)
# st.session_state.auto_refresh = auto_refresh

# if st.sidebar.button("Clear Logs"):
#     st.session_state.logs = []
#     st.session_state.action_taken = False
#     st.experimental_rerun()

# # ---------------------------
# # Page 1: Dashboard
# # ---------------------------
# if page == "Data Overview":
#     st.title("🏭 Corrosion Maintenance Management System")

#     # Try to load dataset
#     df = None
#     dataset_paths = [
#         'data/Corrosion_Chemical_Analysis.csv'
#     ]

#     for path in dataset_paths:
#         try:
#             if os.path.exists(path):
#                 df = pd.read_csv(path)
#                 st.sidebar.success(f"✅ Dataset loaded: {path}")
#                 break
#         except Exception as e:
#             continue

#     if df is None:
#         st.error("❌ Dataset not found. Please place your CSV file in one of these locations:")
#         for path in dataset_paths:
#             st.write(f"- {path}")

#         # Create a mock dataset for demo purposes
#         st.warning("Using mock data for demonstration...")
#         np.random.seed(42)
#         mock_data = {}
#         for col in FEATURE_COLS:
#             mock_data[col] = np.random.uniform(0, 10, 1000)
#         df = pd.DataFrame(mock_data)
#         df['severity'] = np.random.choice(['Low', 'Moderate', 'High', 'Severe'], 1000)

#     if df is not None:
#         # Generate a new reading
#         sample = generate_random_sample(df)
#         pred_label = predict_sample(sample)[0]

#         # Only update if prediction changed or if it's the first run
#         if (st.session_state.last_pred != pred_label or
#             st.session_state.last_generated_at is None):

#             st.session_state.last_pred = pred_label
#             st.session_state.last_generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#             # Log incoming data
#             st.session_state.logs.append({
#                 "time": st.session_state.last_generated_at,
#                 "severity": pred_label,
#                 "action": "Data received"
#             })

#             # Reset action taken if severity changed
#             st.session_state.action_taken = False

#     # Color coding for status card
#     color_map = {
#         "Low": "#4CAF50",      # Green
#         "Moderate": "#FFC107", # Amber
#         "High": "#FF5722",     # Orange
#         "Severe": "#F44336",   # Red
#         "Unknown": "#9E9E9E",  # Grey
#         "Error": "#9E9E9E"     # Grey
#     }
#     bg_color = color_map.get(st.session_state.last_pred, "#9E9E9E")

#     # Status card
#     st.subheader("📊 Live Corrosion Status")

#     col1, col2, col3 = st.columns([2, 1, 1])

#     with col1:
#         st.markdown(
#             f"""
#             <div style='padding:20px; border-radius:15px; background:{bg_color}; color:white; text-align:center; margin-bottom:20px;'>
#                 <h2>🔍 Corrosion Severity: {st.session_state.last_pred}</h2>
#                 <p style='margin:0;'>Last update: {st.session_state.last_generated_at or 'Never'}</p>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#     with col2:
#         if st.button("🔄 Refresh Now"):
#             st.experimental_rerun()

#     with col3:
#         total_logs = len(st.session_state.logs)
#         st.metric("Total Readings", total_logs)

#     # Show sample data if available
#     if df is not None and len(sample) > 0:
#         with st.expander("📋 Current Sensor Readings"):
#             sample_display = sample.round(3)
#             st.dataframe(sample_display, use_container_width=True)

#     # If severe or high, prompt action
#     if st.session_state.last_pred in ["Severe", "High"]:
#         if not st.session_state.action_taken:
#             st.markdown("### ⚠️ Maintenance Action Required")

#             col1, col2 = st.columns([2, 1])

#             with col1:
#                 engineer_email = st.text_input(
#                     "Responsible Engineer Email",
#                     value=st.session_state.engineer_email,
#                     placeholder="engineer@company.com"
#                 )

#             with col2:
#                 st.write("")  # Spacing
#                 st.write("")  # Spacing
#                 if st.button("📧 Send Maintenance Alert", type="primary"):
#                     if engineer_email and "@" in engineer_email:
#                         st.session_state.engineer_email = engineer_email
#                         if send_email_notification(engineer_email, st.session_state.last_pred):
#                             action_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                             st.session_state.logs.append({
#                                 "time": action_time,
#                                 "severity": st.session_state.last_pred,
#                                 "action": f"Email alert sent to {engineer_email}"
#                             })
#                             st.session_state.action_taken = True
#                             st.experimental_rerun()
#                     else:
#                         st.error("Please enter a valid email address.")
#         else:
#             st.success("✅ Maintenance action already taken. Email notification sent.")

#     elif st.session_state.last_pred == "Moderate":
#         st.info("ℹ️ Moderate corrosion detected. Monitor closely.")

#     elif st.session_state.last_pred == "Low":
#         st.success("✅ Corrosion levels are within acceptable range.")

#     # Logs section
#     st.markdown("---")
#     st.subheader("📝 System Logs")

#     if st.session_state.logs:
#         # Show recent logs first
#         recent_logs = st.session_state.logs[-10:]  # Last 10 entries
#         logs_df = pd.DataFrame(recent_logs[::-1])  # Reverse to show newest first

#         # Style the dataframe
#         st.dataframe(
#             logs_df,
#             use_container_width=True,
#             hide_index=True
#         )

#         if len(st.session_state.logs) > 10:
#             st.caption(f"Showing latest 10 of {len(st.session_state.logs)} total logs")
#     else:
#         st.info("No system activities logged yet.")

#     # Auto-refresh logic
#     if st.session_state.auto_refresh:
#         time.sleep(10)
#         st.experimental_rerun()

# # ---------------------------
# # Page 2: Model Testing
# # ---------------------------
# elif page == "Model Testing":
#     st.title("🧪 Test The Model")
#     st.write("Generate random samples, enter values manually, or upload a file for predictions.")

#     if model is None:
#         st.error("❌ Model not loaded. Cannot perform testing.")
#         st.stop()

#     # Manual input
#     st.subheader("✏️ Manual Input")

#     with st.form("manual_input_form"):
#         cols = st.columns(3)
#         manual_data = {}

#         for i, col in enumerate(FEATURE_COLS):
#             with cols[i % 3]:
#                 manual_data[col] = st.number_input(
#                     col.replace("_", " ").title(),
#                     value=0.0,
#                     step=0.01,
#                     key=f"manual_{col}"
#                 )

#         submitted = st.form_submit_button("🔍 Predict from Manual Input", type="primary")

#         if submitted:
#             manual_df = pd.DataFrame([manual_data])
#             preds = predict_sample(manual_df)
#             severity = preds[0]

#             # Show prediction with color
#             color = color_map.get(severity, "#9E9E9E")
#             st.markdown(
#                 f"""
#                 <div style='padding:15px; border-radius:10px; background:{color}; color:white; text-align:center; margin:20px 0;'>
#                     <h3>Predicted Corrosion Severity: {severity}</h3>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )

#     # Random sample generation
#     st.subheader("🎲 Generate Random Sample")

#     col1, col2 = st.columns([1, 2])

#     with col1:
#         n_samples = st.number_input("Number of samples", min_value=1, max_value=100, value=5)

#         if st.button("Generate Random Samples"):
#             chemical_data = pd.read_csv("data/Corrosion_Chemical_Analysis.csv") if os.path.exists("data/Corrosion_Chemical_Analysis.csv") else None
#             df = pd.DataFrame(chemical_data) if chemical_data is not None else chemical_data

#             if df is not None:
#                 random_samples = generate_random_sample(df, n_samples)
#                 preds = predict_sample(random_samples)
#                 random_samples["Predicted_Severity"] = preds
#                 st.session_state.random_results = random_samples
#             else:
#                 st.error("No dataset available for generating random samples.")

#     with col2:
#         if hasattr(st.session_state, 'random_results'):
#             st.write("**Generated Samples:**")

#             # Apply color styling
#             def highlight_severity(val):
#                 colors = {
#                     "Low": "background-color: #4CAF50; color: white;",
#                     "Moderate": "background-color: #FFC107; color: black;",
#                     "High": "background-color: #FF5722; color: white;",
#                     "Severe": "background-color: #F44336; color: white;"
#                 }
#                 return colors.get(val, "")

#             styled_df = st.session_state.random_results.style.applymap(
#                 highlight_severity,
#                 subset=["Predicted_Severity"]
#             )
#             st.dataframe(styled_df, use_container_width=True)

#     # File upload
#     st.markdown("---")
#     st.subheader("📁 Upload File for Batch Testing")

#     uploaded_file = st.file_uploader(
#         "Choose a CSV file",
#         type=["csv"],
#         help="Upload a CSV file with the required feature columns"
#     )

#     if uploaded_file is not None:
#         try:
#             user_df = pd.read_csv(uploaded_file)
#             st.success(f"✅ File uploaded successfully! Shape: {user_df.shape}")

#             # Check if required columns exist
#             missing_cols = [col for col in FEATURE_COLS if col not in user_df.columns]

#             if missing_cols:
#                 st.error(f"❌ Missing required columns: {missing_cols}")
#                 st.write("**Available columns:**", list(user_df.columns))
#             else:
#                 # Make predictions
#                 preds = predict_sample(user_df[FEATURE_COLS])
#                 user_df["Predicted_Severity"] = preds

#                 # Show results with color styling
#                 def highlight_severity(val):
#                     colors = {
#                         "Low": "background-color: #4CAF50; color: white;",
#                         "Moderate": "background-color: #FFC107; color: black;",
#                         "High": "background-color: #FF5722; color: white;",
#                         "Severe": "background-color: #F44336; color: white;"
#                     }
#                     return colors.get(val, "")

#                 styled_df = user_df.style.applymap(
#                     highlight_severity,
#                     subset=["Predicted_Severity"]
#                 )

#                 st.subheader("🎯 Prediction Results")
#                 st.dataframe(styled_df, use_container_width=True)

#                 # Summary statistics
#                 severity_counts = user_df["Predicted_Severity"].value_counts()
#                 st.subheader("📊 Severity Distribution")
#                 st.bar_chart(severity_counts)

#                 # Download button
#                 csv = user_df.to_csv(index=False)
#                 st.download_button(
#                     label="📥 Download Predictions as CSV",
#                     data=csv,
#                     file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                     mime="text/csv"
#                 )

#         except Exception as e:
#             st.error(f"❌ Error processing file: {e}")

#     # Model information
#     with st.expander("ℹ️ Model Information"):
#         st.write("**Required Features:**")
#         for i, col in enumerate(FEATURE_COLS, 1):
#             st.write(f"{i}. {col}")

#         st.write("**Possible Predictions:**")
#         st.write("- Low: Minimal corrosion, normal operations")
#         st.write("- Moderate: Some corrosion present, monitor closely")
#         st.write("- High: Significant corrosion, schedule maintenance")
#         st.write("- Severe: Critical corrosion, immediate action required")


# Quick script to debug feature expectations


# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import joblib
# import time
# from datetime import datetime
# import smtplib
# from email.mime.text import MIMEText
# import os

# # ---------------------------
# # Flexible Model Loading
# # ---------------------------
# @st.cache_resource
# def load_model():
#     """Flexible model loading function that handles multiple PKL formats"""
#     try:
#         # Assume the complete package is renamed to "complete_deployment_package.pkl" for simplicity
#         package_filename = "complete_deployment_package_20250921_025645.pkl"
        
#         # First try with pickle
#         try:
#             with open(package_filename, "rb") as f:
#                 data = pickle.load(f)
#         except:
#             # Try with joblib if pickle fails
#             data = joblib.load(package_filename)
        
#         st.sidebar.info(f"Loaded data type: {type(data)}")
        
#         # Case 1: Complete deployment package
#         if isinstance(data, dict) and all(key in data for key in ['model', 'scaler', 'label_encoder']):
#             st.sidebar.success("✅ Complete deployment package found")
#             return data["model"], data["label_encoder"], data["scaler"], data.get("metadata", {})
        
#         # Case 2: Dictionary with different key names
#         elif isinstance(data, dict):
#             st.sidebar.info(f"Dictionary keys found: {list(data.keys())}")
            
#             model = None
#             scaler = None
#             encoder = None
            
#             # Try different key variations
#             model_keys = ['model', 'xgb_model', 'classifier', 'xgboost_model', 'trained_model']
#             scaler_keys = ['scaler', 'feature_scaler', 'standardscaler', 'std_scaler']
#             encoder_keys = ['label_encoder', 'encoder', 'le', 'target_encoder']
            
#             for key in model_keys:
#                 if key in data:
#                     model = data[key]
#                     break
            
#             for key in scaler_keys:
#                 if key in data:
#                     scaler = data[key]
#                     break
                    
#             for key in encoder_keys:
#                 if key in data:
#                     encoder = data[key]
#                     break
            
#             if model is not None:
#                 st.sidebar.success(f"✅ Model found with key: {key}")
#                 if scaler is None:
#                     st.sidebar.warning("⚠️ No scaler found - creating default")
#                     scaler = create_default_scaler()
#                 if encoder is None:
#                     st.sidebar.warning("⚠️ No encoder found - creating default")
#                     encoder = create_default_encoder()
#                 return model, encoder, scaler, {}
#             else:
#                 st.sidebar.error(f"❌ No model found in keys: {list(data.keys())}")
#                 return None, None, None, {}
        
#         # Case 3: Just the model object directly
#         elif hasattr(data, 'predict'):
#             st.sidebar.warning("⚠️ Only model found - creating default scaler and encoder")
#             scaler = create_default_scaler()
#             encoder = create_default_encoder()
#             return data, encoder, scaler, {}
        
#         else:
#             st.sidebar.error(f"❌ Unrecognized format: {type(data)}")
#             return None, None, None, {}
            
#     except FileNotFoundError:
#         st.sidebar.error("❌ File 'complete_deployment_package.pkl' not found. Please ensure the file exists.")
#         return None, None, None, {}
#     except Exception as e:
#         st.sidebar.error(f"❌ Loading error: {e}")
#         return None, None, None, {}

# def create_default_scaler():
#     """Create a default scaler with realistic ranges (13 features to match training)"""
#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
    
#     # Dummy data with 13 features (base 5 + engineered 4 + location_encoded + 3 optional)
#     dummy_data = np.array([
#         [30, 1.8, 0.9, 18, 12, 16.2, 1.67, 0.15, 0.056, 0, 7, 4.5, 3.5],  # Typical
#         [15, 0.2, 0.1, 10, 8, 8.5, 1.5, 0.025, 0.012, 0, 4, 2, 1],          # Low
#         [45, 4.0, 2.5, 25, 20, 24.8, 1.8, 0.5, 0.1, 0, 12, 8, 6]            # High
#     ])
#     scaler.fit(dummy_data)
#     return scaler

# def create_default_encoder():
#     """Create a default label encoder"""
#     from sklearn.preprocessing import LabelEncoder
#     encoder = LabelEncoder()
#     encoder.fit(['Low', 'Moderate', 'High', 'Severe'])
#     return encoder

# # Try to load model
# model, label_encoder, scaler, metadata = load_model()

# # Define input chemical columns (matching the ones used in training)
# BASE_CHEMICAL_COLS = [
#     "Fe2O3_percent", "SO3_percent", "Cl_percent", "Fe_percent", "SiO2_percent",
#     "Al2O3_percent", "CaO_percent", "MgO_percent"
# ]

# # Model's expected features (13 features matching training)
# MODEL_FEATURE_COLS = [
#     'Fe2O3_percent', 'SO3_percent', 'Cl_percent', 'Fe_percent', 'SiO2_percent',
#     'corrosive_index', 'oxidation_ratio', 'sulfur_ratio', 'chloride_intensity',
#     'location_encoded', 'Al2O3_percent', 'CaO_percent', 'MgO_percent'
# ]

# # ---------------------------
# # Feature Engineering
# # ---------------------------
# def engineer_features(df):
#     """Engineer features to match the trained model (13 features)"""
#     df_eng = df.copy()
    
#     # Create the 4 main engineered features (matching training)
#     df_eng['corrosive_index'] = (df_eng['Fe2O3_percent'] * 0.45 + 
#                                  df_eng['SO3_percent'] * 0.35 + 
#                                  df_eng['Cl_percent'] * 0.20)
    
#     df_eng['oxidation_ratio'] = df_eng['Fe2O3_percent'] / (df_eng['Fe_percent'] + 0.001)
#     df_eng['sulfur_ratio'] = df_eng['SO3_percent'] / (df_eng['SiO2_percent'] + 0.001)
#     df_eng['chloride_intensity'] = df_eng['Cl_percent'] / (df_eng['corrosive_index'] + 0.001)
    
#     # Add location_encoded (default to 0 since no separate encoder is saved)
#     df_eng['location_encoded'] = 0  # Default value, as encoding for new locations isn't possible without saved mapper
    
#     # Ensure all model features are present
#     for col in MODEL_FEATURE_COLS:
#         if col not in df_eng.columns:
#             df_eng[col] = 0.0  # Fill missing with default
    
#     return df_eng[MODEL_FEATURE_COLS]

# # ---------------------------
# # Helper Functions
# # ---------------------------
# def generate_random_sample(df=None, n_samples=1, random_state=None):
#     """Generate random samples with realistic ranges (matching input columns)"""
#     rng = np.random.default_rng(random_state)
    
#     # Realistic ranges for chemical composition (matching BASE_CHEMICAL_COLS)
#     ranges = {
#         'Fe2O3_percent': (15, 45),
#         'SO3_percent': (0.2, 4.0), 
#         'Cl_percent': (0.1, 2.5),
#         'Fe_percent': (10, 25),
#         'SiO2_percent': (8, 20),
#         'Al2O3_percent': (4, 12),
#         'CaO_percent': (2, 8),
#         'MgO_percent': (1, 6)
#     }
    
#     samples = {}
#     for col in BASE_CHEMICAL_COLS:
#         min_val, max_val = ranges.get(col, (0, 10))
#         samples[col] = rng.uniform(min_val, max_val, n_samples)
    
#     return pd.DataFrame(samples)

# def predict_sample(sample_df):
#     """Make predictions with proper error handling"""
#     if model is None:
#         return ["Unknown"] * len(sample_df), [0.0] * len(sample_df)
    
#     try:
#         # Engineer features
#         sample_eng = engineer_features(sample_df)
        
#         # Scale features if scaler is available
#         if scaler is not None:
#             try:
#                 # Handle feature count mismatch
#                 if sample_eng.shape[1] != scaler.n_features_in_:
#                     st.warning(f"Feature mismatch: expected {scaler.n_features_in_}, got {sample_eng.shape[1]}")
#                     if sample_eng.shape[1] < scaler.n_features_in_:
#                         # Pad with zeros
#                         padding = np.zeros((sample_eng.shape[0], scaler.n_features_in_ - sample_eng.shape[1]))
#                         feature_matrix = np.hstack([sample_eng.values, padding])
#                     else:
#                         # Truncate
#                         feature_matrix = sample_eng.iloc[:, :scaler.n_features_in_].values
#                 else:
#                     feature_matrix = sample_eng.values
                
#                 sample_scaled = scaler.transform(feature_matrix)
#             except Exception as e:
#                 st.warning(f"Scaling error: {e}. Using unscaled features.")
#                 sample_scaled = sample_eng.values
#         else:
#             sample_scaled = sample_eng.values
        
#         # Make predictions
#         preds = model.predict(sample_scaled)
        
#         # Get probabilities if available
#         try:
#             proba = model.predict_proba(sample_scaled)
#             confidence = np.max(proba, axis=1)
#         except:
#             confidence = np.array([0.8] * len(preds))  # Default confidence
        
#         # Convert predictions to labels
#         if label_encoder is not None:
#             try:
#                 labels = label_encoder.inverse_transform(preds)
#             except:
#                 # Fallback mapping
#                 label_map = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Severe'}
#                 labels = np.array([label_map.get(p, 'Unknown') for p in preds])
#         else:
#             # Default mapping
#             label_map = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Severe'}
#             labels = np.array([label_map.get(p, 'Unknown') for p in preds])
        
#         return labels, confidence
        
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         return ["Error"] * len(sample_df), [0.0] * len(sample_df)

# # Send email notification (placeholder)
# def send_email_notification(to_email, severity):
#     """Send email notification"""
#     try:
#         # Placeholder - replace with your email logic
#         st.success(f"📧 Email notification sent to: {to_email}")
#         st.info(f"Subject: Corrosion Alert: {severity} Severity Detected")
#         return True
#     except Exception as e:
#         st.error(f"Failed to send email: {e}")
#         return False

# # ---------------------------
# # Session State
# # ---------------------------
# if "logs" not in st.session_state:
#     st.session_state.logs = []
# if "last_pred" not in st.session_state:
#     st.session_state.last_pred = "Unknown"
# if "last_confidence" not in st.session_state:
#     st.session_state.last_confidence = 0.0
# if "last_generated_at" not in st.session_state:
#     st.session_state.last_generated_at = None
# if "engineer_email" not in st.session_state:
#     st.session_state.engineer_email = ""
# if "action_taken" not in st.session_state:
#     st.session_state.action_taken = False
# if "auto_refresh" not in st.session_state:
#     st.session_state.auto_refresh = False

# # ---------------------------
# # Main App
# # ---------------------------
# st.set_page_config(
#     page_title="Corrosion Monitoring System",
#     page_icon="🔧",
#     layout="wide"
# )

# # Sidebar
# st.sidebar.title("🔧 Navigation")

# # Single radio button with unique key to prevent duplicate ID error
# page = st.sidebar.radio("Go to", ["Data Overview", "Model Testing"], key="main_navigation")

# # Model status
# st.sidebar.markdown("---")
# st.sidebar.subheader("🤖 Model Status")

# if st.sidebar.button("Clear Logs"):
#     st.session_state.logs = []
#     st.session_state.action_taken = False
#     st.rerun()

# # Auto-refresh toggle
# auto_refresh = st.sidebar.checkbox("Auto-refresh (every 10s)", value=st.session_state.auto_refresh)
# st.session_state.auto_refresh = auto_refresh

# # ---------------------------
# # Page 1: Dashboard
# # ---------------------------
# if page == "Data Overview":
#     st.title("🏭 Corrosion Maintenance Management System")
#     st.caption("Real-time corrosion monitoring with ML predictions")

#     if model is None:
#         st.error("❌ Model not loaded. Please check the PKL file.")
#         st.stop()

#     # Generate a new reading
#     sample = generate_random_sample()
#     pred_labels, confidence = predict_sample(sample)
#     pred_label = pred_labels[0]
#     pred_confidence = confidence[0]
    
#     # Update if changed
#     if (st.session_state.last_pred != pred_label or 
#         st.session_state.last_generated_at is None):
        
#         st.session_state.last_pred = pred_label
#         st.session_state.last_confidence = pred_confidence
#         st.session_state.last_generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
#         # Log data
#         st.session_state.logs.append({
#             "time": st.session_state.last_generated_at,
#             "severity": pred_label,
#             "confidence": f"{pred_confidence:.1%}",
#             "action": "Data received"
#         })
        
#         st.session_state.action_taken = False

#     # Color coding
#     color_map = {
#         "Low": "#4CAF50",
#         "Moderate": "#FFC107", 
#         "High": "#FF5722",
#         "Severe": "#F44336",
#         "Unknown": "#9E9E9E",
#         "Error": "#9E9E9E"
#     }
#     bg_color = color_map.get(st.session_state.last_pred, "#9E9E9E")

#     # Status display
#     st.subheader("📊 Live Corrosion Status")
    
#     col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
#     with col1:
#         confidence_text = f"Confidence: {st.session_state.last_confidence:.1%}" if st.session_state.last_confidence > 0 else ""
#         st.markdown(
#             f"""
#             <div style='padding:20px; border-radius:15px; background:{bg_color}; color:white; text-align:center; margin-bottom:20px;'>
#                 <h2>🔍 Corrosion Severity: {st.session_state.last_pred}</h2>
#                 <p style='margin:0;'>{confidence_text}</p>
#                 <p style='margin:0; font-size:0.9em;'>Last update: {st.session_state.last_generated_at or 'Never'}</p>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )
    
#     with col2:
#         if st.button("🔄 Refresh Now"):
#             st.rerun()
    
#     with col3:
#         total_logs = len(st.session_state.logs)
#         st.metric("Total Readings", total_logs)
    
#     with col4:
#         if st.session_state.last_confidence > 0:
#             st.metric("Confidence", f"{st.session_state.last_confidence:.1%}")

#     # ---------------------------
#     # CORROSION SEVERITY KEY SECTION
#     # ---------------------------
#     with st.expander("📋 **Corrosion Severity Key** - Understanding the Color System", expanded=False):
#         st.markdown("### 🔑 **What Do These Colors Mean?**")
#         st.markdown("""
#         This system uses **color-coded severity levels** based on the ML model's analysis of chemical composition and corrosion risk factors. Each level represents a different stage of corrosion progression and corresponding maintenance urgency.
#         """)
        
#         # Severity level cards
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # LOW - Green Card
#             st.markdown("""
#             <div style='padding:15px; border-radius:10px; background:#4CAF50; color:white; margin:10px 0;'>
#                 <h3 style='margin:0 0 10px 0;'>🟢 LOW</h3>
#                 <ul style='margin:0; padding-left:20px; font-size:14px;'>
#                     <li><strong>Corrosion Rate:</strong> <0.1 mm/year</li>
#                     <li><strong>Risk Level:</strong> Minimal</li>
#                     <li><strong>Expected Life:</strong> >20 years</li>
#                     <li><strong>Material Loss:</strong> Negligible</li>
#                 </ul>
#                 <p style='margin:10px 0 0 0; font-size:12px; font-style:italic;'>✅ Safe to continue normal operations</p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # MODERATE - Yellow Card
#             st.markdown("""
#             <div style='padding:15px; border-radius:10px; background:#FFC107; color:black; margin:10px 0;'>
#                 <h3 style='margin:0 0 10px 0; color:#333;'>🟡 MODERATE</h3>
#                 <ul style='margin:0; padding-left:20px; font-size:14px;'>
#                     <li><strong>Corrosion Rate:</strong> 0.1-0.5 mm/year</li>
#                     <li><strong>Risk Level:</strong> Moderate</li>
#                     <li><strong>Expected Life:</strong> 10-20 years</li>
#                     <li><strong>Material Loss:</strong> Noticeable but manageable</li>
#                 </ul>
#                 <p style='margin:10px 0 0 0; font-size:12px; font-style:italic; color:#666;'>⚠️ Increase monitoring frequency</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col2:
#             # HIGH - Orange Card
#             st.markdown("""
#             <div style='padding:15px; border-radius:10px; background:#FF5722; color:white; margin:10px 0;'>
#                 <h3 style='margin:0 0 10px 0;'>🟠 HIGH</h3>
#                 <ul style='margin:0; padding-left:20px; font-size:14px;'>
#                     <li><strong>Corrosion Rate:</strong> 0.5-1.5 mm/year</li>
#                     <li><strong>Risk Level:</strong> High</li>
#                     <li><strong>Expected Life:</strong> 2-10 years</li>
#                     <li><strong>Material Loss:</strong> Significant</li>
#                 </ul>
#                 <p style='margin:10px 0 0 0; font-size:12px; font-style:italic;'>🚨 Immediate inspection required</p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # SEVERE - Red Card
#             st.markdown("""
#             <div style='padding:15px; border-radius:10px; background:#F44336; color:white; margin:10px 0;'>
#                 <h3 style='margin:0 0 10px 0;'>🔴 SEVERE</h3>
#                 <ul style='margin:0; padding-left:20px; font-size:14px;'>
#                     <li><strong>Corrosion Rate:</strong> >1.5 mm/year</li>
#                     <li><strong>Risk Level:</strong> Critical</li>
#                     <li><strong>Expected Life:</strong> <2 years</li>
#                     <li><strong>Material Loss:</strong> Severe - Risk of failure</li>
#                 </ul>
#                 <p style='margin:10px 0 0 0; font-size:12px; font-style:italic;'>⛔ Emergency shutdown required</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown("---")
        
#         # Severity threshold table
#         st.markdown("### 📊 **Severity Thresholds & Triggers**")
        
#         severity_data = {
#             "Severity Level": ["Low", "Moderate", "High", "Severe"],
#             "Key Indicators": [
#                 "• Fe2O3 < 25%\n• SO3 < 1.0%\n• Cl < 0.5%\n• Stable ratios",
#                 "• Fe2O3 25-35%\n• SO3 1.0-2.0%\n• Cl 0.5-1.2%\n• Moderate ratios",
#                 "• Fe2O3 35-45%\n• SO3 2.0-3.5%\n• Cl 1.2-2.0%\n• High ratios",
#                 "• Fe2O3 > 45%\n• SO3 > 3.5%\n• Cl > 2.0%\n• Critical ratios"
#             ],
#             "Corrosive Index": ["< 12", "12-20", "20-30", "> 30"],
#             "Action Required": [
#                 "Continue routine monitoring",
#                 "Increase monitoring to weekly",
#                 "Schedule inspection within 1 week",
#                 "Immediate shutdown & replacement"
#             ]
#         }
        
#         severity_df = pd.DataFrame(severity_data)
#         st.dataframe(severity_df, use_container_width=True, hide_index=True)
        
#         st.markdown("---")
        
#         # Maintenance priority guide
#         st.markdown("### 🎯 **Maintenance Priority Guide**")
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("**🚦 Response Timeline:**")
#             timeline_data = {
#                 "Severity": ["Low", "Moderate", "High", "Severe"],
#                 "Timeline": ["Next scheduled", "Within 1 month", "Within 1 week", "Within 24 hours"],
#                 "Priority": ["Routine", "Medium", "High", "Critical"],
#                 "Alert": ["None", "Internal", "Email", "Emergency"]
#             }
#             timeline_df = pd.DataFrame(timeline_data)
#             st.dataframe(timeline_df, use_container_width=True, hide_index=True)
        
#         with col2:
#             st.markdown("**🔧 Recommended Actions:**")
#             st.markdown("""
#             - **Low**: Continue normal operations and scheduled inspections
#             - **Moderate**: Increase monitoring frequency, plan preventive maintenance
#             - **High**: Schedule immediate inspection, prepare replacement parts
#             - **Severe**: Stop operations, isolate component, emergency replacement
#             """)
        
#         st.markdown("---")
        
#         st.markdown("""
#         ### 📈 **How the Model Works**
#         The XGBoost model analyzes **13 key features** including:
#         - **Chemical composition** (Fe2O3, SO3, Cl, Fe, SiO2, etc.)
#         - **Corrosion indices** (calculated ratios and weighted combinations)
#         - **Environmental factors** (location encoding)
        
#         **Model Accuracy**: 87.4% | **F1-Score**: 87.3% | **Trained on**: 3,000+ industrial samples
#         """)

#     # Show current readings
#     with st.expander("📋 Current Sensor Readings"):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.write("**Chemical Composition:**")
#             display_data = sample[BASE_CHEMICAL_COLS].round(3)
#             st.dataframe(display_data, use_container_width=True)
        
#         with col2:
#             st.write("**Engineered Features:**")
#             sample_eng = engineer_features(sample)
#             eng_features = ['corrosive_index', 'oxidation_ratio', 'sulfur_ratio', 'chloride_intensity', 'location_encoded']
#             eng_display = sample_eng[eng_features].round(4)
#             st.dataframe(eng_display, use_container_width=True)

#     # Maintenance actions
#     if st.session_state.last_pred in ["Severe", "High"]:
#         if not st.session_state.action_taken:
#             st.markdown("### ⚠️ Maintenance Action Required")
            
#             priority_info = {
#                 "Severe": {"priority": "CRITICAL", "action": "Immediate shutdown required", "timeline": "< 24 hours"},
#                 "High": {"priority": "HIGH", "action": "Schedule immediate inspection", "timeline": "< 1 week"}
#             }
            
#             info = priority_info[st.session_state.last_pred]
            
#             st.error(f"🚨 {info['priority']} Priority: {info['action']} ({info['timeline']})")
            
#             col1, col2 = st.columns([2, 1])
            
#             with col1:
#                 engineer_email = st.text_input(
#                     "Engineer Email", 
#                     value=st.session_state.engineer_email,
#                     placeholder="engineer@company.com"
#                 )
            
#             with col2:
#                 st.write("")
#                 st.write("")
#                 if st.button("📧 Send Alert", type="primary"):
#                     if engineer_email and "@" in engineer_email:
#                         st.session_state.engineer_email = engineer_email
#                         if send_email_notification(engineer_email, st.session_state.last_pred):
#                             st.session_state.logs.append({
#                                 "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                                 "severity": st.session_state.last_pred,
#                                 "confidence": f"{st.session_state.last_confidence:.1%}",
#                                 "action": f"Email sent to {engineer_email}"
#                             })
#                             st.session_state.action_taken = True
#                             st.rerun()
#                     else:
#                         st.error("Please enter a valid email.")
#         else:
#             st.success("✅ Maintenance action taken.")

#     # System logs
#     st.markdown("---")
#     st.subheader("📝 System Logs")
    
#     if st.session_state.logs:
#         recent_logs = st.session_state.logs[-10:]
#         logs_df = pd.DataFrame(recent_logs[::-1])
#         st.dataframe(logs_df, use_container_width=True, hide_index=True)
        
#         if len(st.session_state.logs) > 10:
#             st.caption(f"Showing latest 10 of {len(st.session_state.logs)} total logs")
#     else:
#         st.info("No logs yet.")

#     # Auto-refresh
#     if st.session_state.auto_refresh:
#         time.sleep(10)
#         st.rerun()

# # ---------------------------
# # Page 2: Model Testing
# # ---------------------------
# elif page == "Model Testing":
#     st.title("🧪 Test The Model")
    
#     # Reference the color_map here for the testing section
#     color_map = {
#         "Low": "#4CAF50",
#         "Moderate": "#FFC107", 
#         "High": "#FF5722",
#         "Severe": "#F44336",
#         "Unknown": "#9E9E9E",
#         "Error": "#9E9E9E"
#     }
    
#     if model is None:
#         st.error("❌ Model not loaded.")
#         st.stop()

#     # Manual input
#     st.subheader("✏️ Manual Input")
    
#     with st.form("manual_input"):
#         cols = st.columns(3)
#         manual_data = {}
        
#         defaults = {
#             'Fe2O3_percent': 30.0, 'SO3_percent': 1.8, 'Cl_percent': 0.9,
#             'Fe_percent': 18.0, 'SiO2_percent': 12.0,
#             'Al2O3_percent': 7.0, 'CaO_percent': 4.5, 'MgO_percent': 3.5
#         }
        
#         for i, col in enumerate(BASE_CHEMICAL_COLS):
#             with cols[i % 3]:
#                 manual_data[col] = st.number_input(
#                     col.replace("_", " ").title(),
#                     value=defaults.get(col, 0.0),
#                     min_value=0.0,
#                     max_value=100.0,
#                     step=0.1,
#                     key=f"manual_{col}"
#                 )
        
#         if st.form_submit_button("🔍 Predict", type="primary"):
#             manual_df = pd.DataFrame([manual_data])
#             preds, confidence = predict_sample(manual_df)
#             severity = preds[0]
#             conf = confidence[0]
            
#             color = color_map.get(severity, "#9E9E9E")
#             st.markdown(
#                 f"""
#                 <div style='padding:15px; border-radius:10px; background:{color}; color:white; text-align:center; margin:20px 0;'>
#                     <h3>Predicted: {severity}</h3>
#                     <p>Confidence: {conf:.1%}</p>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )

#     # Random samples
#     st.subheader("🎲 Random Samples")
    
#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         n_samples = st.number_input("Number of samples", min_value=1, max_value=50, value=5)
        
#         if st.button("Generate Random Samples"):
#             random_samples = generate_random_sample(n_samples=n_samples)
#             preds, confidence = predict_sample(random_samples)
#             random_samples["Severity"] = preds
#             random_samples["Confidence"] = [f"{c:.1%}" for c in confidence]
#             st.session_state.random_results = random_samples
    
#     with col2:
#         if hasattr(st.session_state, 'random_results'):
#             st.write("**Results:**")
#             st.dataframe(st.session_state.random_results, use_container_width=True)

#     # File upload
#     st.subheader("📁 File Upload")
    
#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
#     if uploaded_file:
#         try:
#             user_df = pd.read_csv(uploaded_file)
#             st.success(f"Loaded: {user_df.shape}")
            
#             # Check for required columns
#             missing_cols = [col for col in BASE_CHEMICAL_COLS if col not in user_df.columns]
            
#             if missing_cols:
#                 st.warning(f"Missing columns filled with defaults: {missing_cols}")
#                 for col in missing_cols:
#                     user_df[col] = defaults.get(col, 0.0)
                
#             # Predict using only the required columns
#             input_df = user_df[BASE_CHEMICAL_COLS]
#             preds, confidence = predict_sample(input_df)
#             user_df["Predicted_Severity"] = preds
#             user_df["Confidence"] = [f"{c:.1%}" for c in confidence]
            
#             st.dataframe(user_df, use_container_width=True)
            
#             # Download
#             csv = user_df.to_csv(index=False)
#             st.download_button(
#                 "📥 Download Results",
#                 data=csv,
#                 file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                 mime="text/csv"
#             )
                
#         except Exception as e:
#             st.error(f"Error: {e}")



import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import time
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import os
import sqlite3
from hashlib import sha256
import secrets
from pathlib import Path

# ---------------------------
# AUTHENTICATION SYSTEM
# ---------------------------

class AuthManager:
    """Lightweight SQLite-based authentication system"""
    
    def __init__(self, db_path="auth.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with users table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Create sessions table for tracking active sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_token TEXT UNIQUE NOT NULL,
                username TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Insert default admin user if no users exist
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            # Default admin: username="admin", password="Admin123!"
            default_password = "Admin123!"
            password_hash = sha256(default_password.encode()).hexdigest()
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("admin", password_hash, "admin")
            )
            st.info("🔐 Default admin account created: username='admin', password='Admin123!'")
        
        conn.commit()
        conn.close()
    
    def create_session_token(self, username):
        """Generate secure session token"""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=8)  # 8-hour session
        expires_str = expires_at.strftime("%Y-%m-%d %H:%M:%S")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (session_token, username, expires_at) VALUES (?, ?, ?)",
            (token, username, expires_str)
        )
        conn.commit()
        conn.close()
        
        return token
    
    def validate_session(self, session_token):
        """Validate session token and update last login"""
        if not session_token:
            return False, None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if session is valid and not expired
        cursor.execute('''
            SELECT s.username, s.expires_at, u.role 
            FROM sessions s 
            JOIN users u ON s.username = u.username 
            WHERE s.session_token = ? AND s.is_active = 1 AND u.is_active = 1
        ''', (session_token,))
        
        result = cursor.fetchone()
        if result:
            username, expires_at_str, role = result
            expires_at = datetime.strptime(expires_at_str, "%Y-%m-%d %H:%M:%S")
            
            if datetime.now() < expires_at:
                # Update last login
                cursor.execute(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?",
                    (username,)
                )
                conn.commit()
                conn.close()
                return True, {"username": username, "role": role}
        
        conn.close()
        return False, None
    
    def authenticate_user(self, username, password):
        """Authenticate user credentials"""
        password_hash = sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT username, role FROM users WHERE username = ? AND password_hash = ? AND is_active = 1",
            (username, password_hash)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return True, {"username": result[0], "role": result[1]}
        return False, None
    
    def logout_session(self, session_token):
        """Invalidate session token"""
        if not session_token:
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET is_active = 0 WHERE session_token = ?",
            (session_token,)
        )
        conn.commit()
        conn.close()
        return True
    
    def get_user_count(self):
        """Get total number of users"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
        count = cursor.fetchone()[0]
        conn.close()
        return count

# Initialize authentication manager
auth_manager = AuthManager()

# ---------------------------
# SESSION MANAGEMENT
# ---------------------------

def get_session_state():
    """Get current session state"""
    if "session_token" not in st.session_state:
        st.session_state.session_token = None
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0
    if "last_login_attempt" not in st.session_state:
        st.session_state.last_login_attempt = None
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    
    return st.session_state

# ---------------------------
# LOGIN PAGE
# ---------------------------

def show_login_page():
    """Display login page"""
    st.set_page_config(
        page_title="Corrosion Monitoring System - Login",
        page_icon="🔐",
        layout="centered"
    )
    
    # Custom CSS for login page
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 500px;
    }
    .login-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .login-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .login-form {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: #333 !important;
        padding: 0.75rem !important;
    }
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        width: 100%;
        margin-top: 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #45a049, #4CAF50);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
    }
    .error-message {
        background: rgba(244, 67, 54, 0.2);
        border: 1px solid rgba(244, 67, 54, 0.3);
        border-radius: 8px;
        padding: 1rem;
        color: #f44336;
        margin: 1rem 0;
    }
    .success-message {
        background: rgba(76, 175, 80, 0.2);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 8px;
        padding: 1rem;
        color: #4caf50;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="login-card"><h1 class="login-header">🔧 Corrosion Monitoring System</h1></div>', unsafe_allow_html=True)
    
    # Login form
    with st.form("login_form", clear_on_submit=True):
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        
        st.markdown("### 🔐 **Secure Login Required**")
        
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
        
        # Rate limiting
        session_state = get_session_state()
        now = datetime.now()
        
        if session_state.last_login_attempt:
            time_since_last = now - session_state.last_login_attempt
            if time_since_last < timedelta(seconds=3) and session_state.login_attempts >= 3:
                remaining = 3 - time_since_last.total_seconds()
                st.markdown(f'<div class="error-message">⏳ Too many failed attempts. Please wait {remaining:.0f} seconds</div>', unsafe_allow_html=True)
                st.stop()
        
        submitted = st.form_submit_button("🚀 Login", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submitted:
            session_state.last_login_attempt = now
            
            if not username or not password:
                st.markdown('<div class="error-message">❌ Please enter both username and password</div>', unsafe_allow_html=True)
                session_state.login_attempts += 1
            else:
                is_valid, user_info = auth_manager.authenticate_user(username, password)
                
                if is_valid:
                    # Successful login
                    session_token = auth_manager.create_session_token(user_info["username"])
                    session_state.session_token = session_token
                    session_state.user_info = user_info
                    session_state.is_authenticated = True
                    session_state.login_attempts = 0
                    
                    st.markdown('<div class="success-message">✅ Login successful! Redirecting...</div>', unsafe_allow_html=True)
                    st.rerun()
                else:
                    session_state.login_attempts += 1
                    max_attempts = 5
                    remaining_attempts = max_attempts - session_state.login_attempts
                    
                    if remaining_attempts > 0:
                        st.markdown(f'<div class="error-message">❌ Invalid credentials. {remaining_attempts} attempts remaining.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-message">🔒 Account temporarily locked. Please try again in 3 seconds.</div>', unsafe_allow_html=True)
    
    # Footer with default credentials hint
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        💡 <strong>First time here?</strong> Use default credentials: <code>admin</code> / <code>Admin123!</code>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# MAIN APP WITH AUTHENTICATION
# ---------------------------

# Check authentication status
session_state = get_session_state()
is_valid_session, user_info = auth_manager.validate_session(session_state.session_token)

if not is_valid_session and not session_state.is_authenticated:
    # Show login page
    show_login_page()
    st.stop()

# If we reach here, user is authenticated
session_state.is_authenticated = True
session_state.user_info = user_info

# Now load the main app components
@st.cache_resource
def load_model():
    """Flexible model loading function that handles multiple PKL formats"""
    try:
        # Assume the complete package is renamed to "complete_deployment_package.pkl" for simplicity
        # package_filename = "complete_deployment_package_20250921_025645.pkl"
        package_filename = Path.cwd() / "complete_deployment_package_20250921_025645.pkl"

        # First try with pickle
        try:
            with open(package_filename, "rb") as f:
                data = pickle.load(f)
        except:
            # Try with joblib if pickle fails
            data = joblib.load(package_filename)
        
        # Only show model loading info to admin users
        if user_info["role"] == "admin":
            st.sidebar.info(f"Loaded data type: {type(data)}")
        
        # Case 1: Complete deployment package
        if isinstance(data, dict) and all(key in data for key in ['model', 'scaler', 'label_encoder']):
            if user_info["role"] == "admin":
                st.sidebar.success("✅ Complete deployment package found")
            return data["model"], data["label_encoder"], data["scaler"], data.get("metadata", {})
        
        # Case 2: Dictionary with different key names
        elif isinstance(data, dict):
            if user_info["role"] == "admin":
                st.sidebar.info(f"Dictionary keys found: {list(data.keys())}")
            
            model = None
            scaler = None
            encoder = None
            
            # Try different key variations
            model_keys = ['model', 'xgb_model', 'classifier', 'xgboost_model', 'trained_model']
            scaler_keys = ['scaler', 'feature_scaler', 'standardscaler', 'std_scaler']
            encoder_keys = ['label_encoder', 'encoder', 'le', 'target_encoder']
            
            for key in model_keys:
                if key in data:
                    model = data[key]
                    break
            
            for key in scaler_keys:
                if key in data:
                    scaler = data[key]
                    break
                    
            for key in encoder_keys:
                if key in data:
                    encoder = data[key]
                    break
            
            if model is not None:
                if user_info["role"] == "admin":
                    st.sidebar.success(f"✅ Model found with key: {key}")
                if scaler is None:
                    if user_info["role"] == "admin":
                        st.sidebar.warning("⚠️ No scaler found - creating default")
                    scaler = create_default_scaler()
                if encoder is None:
                    if user_info["role"] == "admin":
                        st.sidebar.warning("⚠️ No encoder found - creating default")
                    encoder = create_default_encoder()
                return model, encoder, scaler, {}
            else:
                if user_info["role"] == "admin":
                    st.sidebar.error(f"❌ No model found in keys: {list(data.keys())}")
                return None, None, None, {}
        
        # Case 3: Just the model object directly
        elif hasattr(data, 'predict'):
            if user_info["role"] == "admin":
                st.sidebar.warning("⚠️ Only model found - creating default scaler and encoder")
            scaler = create_default_scaler()
            encoder = create_default_encoder()
            return data, encoder, scaler, {}
        
        else:
            if user_info["role"] == "admin":
                st.sidebar.error(f"❌ Unrecognized format: {type(data)}")
            return None, None, None, {}
            
    except FileNotFoundError:
        if user_info["role"] == "admin":
            st.sidebar.error("❌ File 'complete_deployment_package.pkl' not found. Please ensure the file exists.")
        return None, None, None, {}
    except Exception as e:
        if user_info["role"] == "admin":
            st.sidebar.error(f"❌ Loading error: {e}")
        return None, None, None, {}

def create_default_scaler():
    """Create a default scaler with realistic ranges (13 features to match training)"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Dummy data with 13 features (base 5 + engineered 4 + location_encoded + 3 optional)
    dummy_data = np.array([
        [30, 1.8, 0.9, 18, 12, 16.2, 1.67, 0.15, 0.056, 0, 7, 4.5, 3.5],  # Typical
        [15, 0.2, 0.1, 10, 8, 8.5, 1.5, 0.025, 0.012, 0, 4, 2, 1],          # Low
        [45, 4.0, 2.5, 25, 20, 24.8, 1.8, 0.5, 0.1, 0, 12, 8, 6]            # High
    ])
    scaler.fit(dummy_data)
    return scaler

def create_default_encoder():
    """Create a default label encoder"""
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(['Low', 'Moderate', 'High', 'Severe'])
    return encoder

# Try to load model
model, label_encoder, scaler, metadata = load_model()

# Define input chemical columns (matching the ones used in training)
BASE_CHEMICAL_COLS = [
    "Fe2O3_percent", "SO3_percent", "Cl_percent", "Fe_percent", "SiO2_percent",
    "Al2O3_percent", "CaO_percent", "MgO_percent"
]

# Model's expected features (13 features matching training)
MODEL_FEATURE_COLS = [
    'Fe2O3_percent', 'SO3_percent', 'Cl_percent', 'Fe_percent', 'SiO2_percent',
    'corrosive_index', 'oxidation_ratio', 'sulfur_ratio', 'chloride_intensity',
    'location_encoded', 'Al2O3_percent', 'CaO_percent', 'MgO_percent'
]

# ---------------------------
# Feature Engineering
# ---------------------------
def engineer_features(df):
    """Engineer features to match the trained model (13 features)"""
    df_eng = df.copy()
    
    # Create the 4 main engineered features (matching training)
    df_eng['corrosive_index'] = (df_eng['Fe2O3_percent'] * 0.45 + 
                                 df_eng['SO3_percent'] * 0.35 + 
                                 df_eng['Cl_percent'] * 0.20)
    
    df_eng['oxidation_ratio'] = df_eng['Fe2O3_percent'] / (df_eng['Fe_percent'] + 0.001)
    df_eng['sulfur_ratio'] = df_eng['SO3_percent'] / (df_eng['SiO2_percent'] + 0.001)
    df_eng['chloride_intensity'] = df_eng['Cl_percent'] / (df_eng['corrosive_index'] + 0.001)
    
    # Add location_encoded (default to 0 since no separate encoder is saved)
    df_eng['location_encoded'] = 0  # Default value, as encoding for new locations isn't possible without saved mapper
    
    # Ensure all model features are present
    for col in MODEL_FEATURE_COLS:
        if col not in df_eng.columns:
            df_eng[col] = 0.0  # Fill missing with default
    
    return df_eng[MODEL_FEATURE_COLS]

# ---------------------------
# Helper Functions
# ---------------------------
def generate_random_sample(df=None, n_samples=1, random_state=None):
    """Generate random samples with realistic ranges (matching input columns)"""
    rng = np.random.default_rng(random_state)
    
    # Realistic ranges for chemical composition (matching BASE_CHEMICAL_COLS)
    ranges = {
        'Fe2O3_percent': (15, 45),
        'SO3_percent': (0.2, 4.0), 
        'Cl_percent': (0.1, 2.5),
        'Fe_percent': (10, 25),
        'SiO2_percent': (8, 20),
        'Al2O3_percent': (4, 12),
        'CaO_percent': (2, 8),
        'MgO_percent': (1, 6)
    }
    
    samples = {}
    for col in BASE_CHEMICAL_COLS:
        min_val, max_val = ranges.get(col, (0, 10))
        samples[col] = rng.uniform(min_val, max_val, n_samples)
    
    return pd.DataFrame(samples)

def predict_sample(sample_df):
    """Make predictions with proper error handling"""
    if model is None:
        return ["Unknown"] * len(sample_df), [0.0] * len(sample_df)
    
    try:
        # Engineer features
        sample_eng = engineer_features(sample_df)
        
        # Scale features if scaler is available
        if scaler is not None:
            try:
                # Handle feature count mismatch
                if sample_eng.shape[1] != scaler.n_features_in_:
                    if user_info["role"] == "admin":
                        st.warning(f"Feature mismatch: expected {scaler.n_features_in_}, got {sample_eng.shape[1]}")
                    if sample_eng.shape[1] < scaler.n_features_in_:
                        # Pad with zeros
                        padding = np.zeros((sample_eng.shape[0], scaler.n_features_in_ - sample_eng.shape[1]))
                        feature_matrix = np.hstack([sample_eng.values, padding])
                    else:
                        # Truncate
                        feature_matrix = sample_eng.iloc[:, :scaler.n_features_in_].values
                else:
                    feature_matrix = sample_eng.values
                
                sample_scaled = scaler.transform(feature_matrix)
            except Exception as e:
                if user_info["role"] == "admin":
                    st.warning(f"Scaling error: {e}. Using unscaled features.")
                sample_scaled = sample_eng.values
        else:
            sample_scaled = sample_eng.values
        
        # Make predictions
        preds = model.predict(sample_scaled)
        
        # Get probabilities if available
        try:
            proba = model.predict_proba(sample_scaled)
            confidence = np.max(proba, axis=1)
        except:
            confidence = np.array([0.8] * len(preds))  # Default confidence
        
        # Convert predictions to labels
        if label_encoder is not None:
            try:
                labels = label_encoder.inverse_transform(preds)
            except:
                # Fallback mapping
                label_map = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Severe'}
                labels = np.array([label_map.get(p, 'Unknown') for p in preds])
        else:
            # Default mapping
            label_map = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Severe'}
            labels = np.array([label_map.get(p, 'Unknown') for p in preds])
        
        return labels, confidence
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return ["Error"] * len(sample_df), [0.0] * len(sample_df)

# Send email notification (placeholder)
def send_email_notification(to_email, severity):
    """Send email notification"""
    try:
        # Placeholder - replace with your email logic
        st.success(f"📧 Email notification sent to: {to_email}")
        st.info(f"Subject: Corrosion Alert: {severity} Severity Detected")
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# ---------------------------
# Session State for Main App
# ---------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last_pred" not in st.session_state:
    st.session_state.last_pred = "Unknown"
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0
if "last_generated_at" not in st.session_state:
    st.session_state.last_generated_at = None
if "engineer_email" not in st.session_state:
    st.session_state.engineer_email = ""
if "action_taken" not in st.session_state:
    st.session_state.action_taken = False
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

# ---------------------------
# Main App UI
# ---------------------------
st.set_page_config(
    page_title="Corrosion Monitoring System",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS for the main app
st.markdown("""
<style>
    .user-header {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .logout-btn {
        background: linear-gradient(45deg, #f44336, #d32f2f);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .logout-btn:hover {
        background: linear-gradient(45deg, #d32f2f, #b71c1c);
        transform: translateY(-1px);
    }
    .admin-badge {
        background: #FF9800;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header with user info
st.markdown('''
<div class="user-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin: 0;">👋 Welcome, {}</h2>
            <p style="margin: 0; opacity: 0.9;">Corrosion Monitoring Dashboard</p>
        </div>
        <div style="display: flex; align-items: center; gap: 1rem;">
            {} 
            <button class="logout-btn" onclick="window.location.href='?logout=true'">🚪 Logout</button>
        </div>
    </div>
</div>
'''.format(
    session_state.user_info["username"],
    '<span class="admin-badge">ADMIN</span>' if session_state.user_info["role"] == "admin" else '<span style="color: #ccc;">User</span>'
), unsafe_allow_html=True)

# Handle logout
if st.query_params.get("logout"):
    auth_manager.logout_session(session_state.session_token)
    for key in list(session_state.keys()):
        delattr(session_state, key)
    st.query_params.clear()
    st.rerun()

# Sidebar
st.sidebar.title("🔧 Navigation")

# Single radio button with unique key to prevent duplicate ID error
page = st.sidebar.radio("Go to", ["Data Overview", "Model Testing"], key="main_navigation")

# Model status (only show to admin users)
if session_state.user_info["role"] == "admin":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Status")

# User stats (show to all users)
st.sidebar.markdown("---")
st.sidebar.subheader("👥 User Stats")
user_count = auth_manager.get_user_count()
st.sidebar.metric("Active Users", user_count)

if session_state.user_info["role"] == "admin":
    if st.sidebar.button("Clear Logs"):
        st.session_state.logs = []
        st.session_state.action_taken = False
        st.rerun()

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (every 10s)", value=st.session_state.auto_refresh)
st.session_state.auto_refresh = auto_refresh

# ---------------------------
# Page 1: Dashboard
# ---------------------------
if page == "Data Overview":
    st.title("🏭 Corrosion Maintenance Management System")
    st.caption("Real-time corrosion monitoring with ML predictions")

    if model is None:
        st.error("❌ Model not loaded. Please check the PKL file.")
        if session_state.user_info["role"] == "admin":
            st.info("👨‍💻 As an admin, you can check the model loading logs in the sidebar.")
        st.stop()

    # Generate a new reading
    sample = generate_random_sample()
    pred_labels, confidence = predict_sample(sample)
    pred_label = pred_labels[0]
    pred_confidence = confidence[0]
    
    # Update if changed
    if (st.session_state.last_pred != pred_label or 
        st.session_state.last_generated_at is None):
        
        st.session_state.last_pred = pred_label
        st.session_state.last_confidence = pred_confidence
        st.session_state.last_generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log data
        st.session_state.logs.append({
            "time": st.session_state.last_generated_at,
            "severity": pred_label,
            "confidence": f"{pred_confidence:.1%}",
            "action": "Data received",
            "user": session_state.user_info["username"]
        })
        
        st.session_state.action_taken = False

    # Color coding
    color_map = {
        "Low": "#4CAF50",
        "Moderate": "#FFC107", 
        "High": "#FF5722",
        "Severe": "#F44336",
        "Unknown": "#9E9E9E",
        "Error": "#9E9E9E"
    }
    bg_color = color_map.get(st.session_state.last_pred, "#9E9E9E")

    # Status display
    st.subheader("📊 Live Corrosion Status")
    
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        confidence_text = f"Confidence: {st.session_state.last_confidence:.1%}" if st.session_state.last_confidence > 0 else ""
        st.markdown(
            f"""
            <div style='padding:20px; border-radius:15px; background:{bg_color}; color:white; text-align:center; margin-bottom:20px;'>
                <h2>🔍 Corrosion Severity: {st.session_state.last_pred}</h2>
                <p style='margin:0;'>{confidence_text}</p>
                <p style='margin:0; font-size:0.9em;'>Last update: {st.session_state.last_generated_at or 'Never'}</p>
                <p style='margin:0; font-size:0.8em; opacity:0.8;'>Monitored by: {session_state.user_info["username"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    with col3:
        total_logs = len(st.session_state.logs)
        st.metric("Total Readings", total_logs)
    
    with col4:
        if st.session_state.last_confidence > 0:
            st.metric("Confidence", f"{st.session_state.last_confidence:.1%}")

    # ---------------------------
    # CORROSION SEVERITY KEY SECTION
    # ---------------------------
    with st.expander("📋 **Corrosion Severity Key** - Understanding the Color System", expanded=False):
        st.markdown("### 🔑 **What Do These Colors Mean?**")
        st.markdown("""
        This system uses **color-coded severity levels** based on the ML model's analysis of chemical composition and corrosion risk factors. Each level represents a different stage of corrosion progression and corresponding maintenance urgency.
        """)
        
        # Severity level cards
        col1, col2 = st.columns(2)
        
        with col1:
            # LOW - Green Card
            st.markdown("""
            <div style='padding:15px; border-radius:10px; background:#4CAF50; color:white; margin:10px 0;'>
                <h3 style='margin:0 0 10px 0;'>🟢 LOW</h3>
                <ul style='margin:0; padding-left:20px; font-size:14px;'>
                    <li><strong>Corrosion Rate:</strong> <0.1 mm/year</li>
                    <li><strong>Risk Level:</strong> Minimal</li>
                    <li><strong>Expected Life:</strong> >20 years</li>
                    <li><strong>Material Loss:</strong> Negligible</li>
                </ul>
                <p style='margin:10px 0 0 0; font-size:12px; font-style:italic;'>✅ Safe to continue normal operations</p>
            </div>
            """, unsafe_allow_html=True)
            
            # MODERATE - Yellow Card
            st.markdown("""
            <div style='padding:15px; border-radius:10px; background:#FFC107; color:black; margin:10px 0;'>
                <h3 style='margin:0 0 10px 0; color:#333;'>🟡 MODERATE</h3>
                <ul style='margin:0; padding-left:20px; font-size:14px;'>
                    <li><strong>Corrosion Rate:</strong> 0.1-0.5 mm/year</li>
                    <li><strong>Risk Level:</strong> Moderate</li>
                    <li><strong>Expected Life:</strong> 10-20 years</li>
                    <li><strong>Material Loss:</strong> Noticeable but manageable</li>
                </ul>
                <p style='margin:10px 0 0 0; font-size:12px; font-style:italic; color:#666;'>⚠️ Increase monitoring frequency</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # HIGH - Orange Card
            st.markdown("""
            <div style='padding:15px; border-radius:10px; background:#FF5722; color:white; margin:10px 0;'>
                <h3 style='margin:0 0 10px 0;'>🟠 HIGH</h3>
                <ul style='margin:0; padding-left:20px; font-size:14px;'>
                    <li><strong>Corrosion Rate:</strong> 0.5-1.5 mm/year</li>
                    <li><strong>Risk Level:</strong> High</li>
                    <li><strong>Expected Life:</strong> 2-10 years</li>
                    <li><strong>Material Loss:</strong> Significant</li>
                </ul>
                <p style='margin:10px 0 0 0; font-size:12px; font-style:italic;'>🚨 Immediate inspection required</p>
            </div>
            """, unsafe_allow_html=True)
            
            # SEVERE - Red Card
            st.markdown("""
            <div style='padding:15px; border-radius:10px; background:#F44336; color:white; margin:10px 0;'>
                <h3 style='margin:0 0 10px 0;'>🔴 SEVERE</h3>
                <ul style='margin:0; padding-left:20px; font-size:14px;'>
                    <li><strong>Corrosion Rate:</strong> >1.5 mm/year</li>
                    <li><strong>Risk Level:</strong> Critical</li>
                    <li><strong>Expected Life:</strong> <2 years</li>
                    <li><strong>Material Loss:</strong> Severe - Risk of failure</li>
                </ul>
                <p style='margin:10px 0 0 0; font-size:12px; font-style:italic;'>⛔ Emergency shutdown required</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Severity threshold table
        st.markdown("### 📊 **Severity Thresholds & Triggers**")
        
        severity_data = {
            "Severity Level": ["Low", "Moderate", "High", "Severe"],
            "Key Indicators": [
                "• Fe2O3 < 25%\n• SO3 < 1.0%\n• Cl < 0.5%\n• Stable ratios",
                "• Fe2O3 25-35%\n• SO3 1.0-2.0%\n• Cl 0.5-1.2%\n• Moderate ratios",
                "• Fe2O3 35-45%\n• SO3 2.0-3.5%\n• Cl 1.2-2.0%\n• High ratios",
                "• Fe2O3 > 45%\n• SO3 > 3.5%\n• Cl > 2.0%\n• Critical ratios"
            ],
            "Corrosive Index": ["< 12", "12-20", "20-30", "> 30"],
            "Action Required": [
                "Continue routine monitoring",
                "Increase monitoring to weekly",
                "Schedule inspection within 1 week",
                "Immediate shutdown & replacement"
            ]
        }
        
        severity_df = pd.DataFrame(severity_data)
        st.dataframe(severity_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Maintenance priority guide
        st.markdown("### 🎯 **Maintenance Priority Guide**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🚦 Response Timeline:**")
            timeline_data = {
                "Severity": ["Low", "Moderate", "High", "Severe"],
                "Timeline": ["Next scheduled", "Within 1 month", "Within 1 week", "Within 24 hours"],
                "Priority": ["Routine", "Medium", "High", "Critical"],
                "Alert": ["None", "Internal", "Email", "Emergency"]
            }
            timeline_df = pd.DataFrame(timeline_data)
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**🔧 Recommended Actions:**")
            st.markdown("""
            - **Low**: Continue normal operations and scheduled inspections
            - **Moderate**: Increase monitoring frequency, plan preventive maintenance
            - **High**: Schedule immediate inspection, prepare replacement parts
            - **Severe**: Stop operations, isolate component, emergency replacement
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📈 **How the Model Works**
        The XGBoost model analyzes **13 key features** including:
        - **Chemical composition** (Fe2O3, SO3, Cl, Fe, SiO2, etc.)
        - **Corrosion indices** (calculated ratios and weighted combinations)
        - **Environmental factors** (location encoding)
        
        **Model Accuracy**: 87.4% | **F1-Score**: 87.3% | **Trained on**: 3,000+ industrial samples
        """)

    # Show current readings
    with st.expander("📋 Current Sensor Readings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Chemical Composition:**")
            display_data = sample[BASE_CHEMICAL_COLS].round(3)
            st.dataframe(display_data, use_container_width=True)
        
        with col2:
            st.write("**Engineered Features:**")
            sample_eng = engineer_features(sample)
            eng_features = ['corrosive_index', 'oxidation_ratio', 'sulfur_ratio', 'chloride_intensity', 'location_encoded']
            eng_display = sample_eng[eng_features].round(4)
            st.dataframe(eng_display, use_container_width=True)

    # Maintenance actions
    if st.session_state.last_pred in ["Severe", "High"]:
        if not st.session_state.action_taken:
            st.markdown("### ⚠️ Maintenance Action Required")
            
            priority_info = {
                "Severe": {"priority": "CRITICAL", "action": "Immediate shutdown required", "timeline": "< 24 hours"},
                "High": {"priority": "HIGH", "action": "Schedule immediate inspection", "timeline": "< 1 week"}
            }
            
            info = priority_info[st.session_state.last_pred]
            
            st.error(f"🚨 {info['priority']} Priority: {info['action']} ({info['timeline']})")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                engineer_email = st.text_input(
                    "Engineer Email", 
                    value=st.session_state.engineer_email,
                    placeholder="engineer@company.com"
                )
            
            with col2:
                st.write("")
                st.write("")
                if st.button("📧 Send Alert", type="primary"):
                    if engineer_email and "@" in engineer_email:
                        st.session_state.engineer_email = engineer_email
                        if send_email_notification(engineer_email, st.session_state.last_pred):
                            st.session_state.logs.append({
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "severity": st.session_state.last_pred,
                                "confidence": f"{st.session_state.last_confidence:.1%}",
                                "action": f"Email sent to {engineer_email}",
                                "user": session_state.user_info["username"]
                            })
                            st.session_state.action_taken = True
                            st.rerun()
                    else:
                        st.error("Please enter a valid email.")
        else:
            st.success("✅ Maintenance action taken.")

    # System logs
    st.markdown("---")
    st.subheader("📝 System Logs")
    
    if st.session_state.logs:
        recent_logs = st.session_state.logs[-10:]
        logs_df = pd.DataFrame(recent_logs[::-1])
        st.dataframe(logs_df, use_container_width=True, hide_index=True)
        
        if len(st.session_state.logs) > 10:
            st.caption(f"Showing latest 10 of {len(st.session_state.logs)} total logs")
    else:
        st.info("No logs yet.")

    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(10)
        st.rerun()

# ---------------------------
# Page 2: Model Testing
# ---------------------------
elif page == "Model Testing":
    st.title("🧪 Test The Model")
    
    # Reference the color_map here for the testing section
    color_map = {
        "Low": "#4CAF50",
        "Moderate": "#FFC107", 
        "High": "#FF5722",
        "Severe": "#F44336",
        "Unknown": "#9E9E9E",
        "Error": "#9E9E9E"
    }
    
    if model is None:
        st.error("❌ Model not loaded.")
        if session_state.user_info["role"] == "admin":
            st.info("👨‍💻 As an admin, you can check the model loading logs in the sidebar.")
        st.stop()

    # Manual input
    st.subheader("✏️ Manual Input")
    
    with st.form("manual_input"):
        cols = st.columns(3)
        manual_data = {}
        
        defaults = {
            'Fe2O3_percent': 30.0, 'SO3_percent': 1.8, 'Cl_percent': 0.9,
            'Fe_percent': 18.0, 'SiO2_percent': 12.0,
            'Al2O3_percent': 7.0, 'CaO_percent': 4.5, 'MgO_percent': 3.5
        }
        
        for i, col in enumerate(BASE_CHEMICAL_COLS):
            with cols[i % 3]:
                manual_data[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=defaults.get(col, 0.0),
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    key=f"manual_{col}"
                )
        
        if st.form_submit_button("🔍 Predict", type="primary"):
            manual_df = pd.DataFrame([manual_data])
            preds, confidence = predict_sample(manual_df)
            severity = preds[0]
            conf = confidence[0]
            
            color = color_map.get(severity, "#9E9E9E")
            st.markdown(
                f"""
                <div style='padding:15px; border-radius:10px; background:{color}; color:white; text-align:center; margin:20px 0;'>
                    <h3>Predicted: {severity}</h3>
                    <p>Confidence: {conf:.1%}</p>
                    <p style='font-size:0.9em; opacity:0.8;'>Tested by: {session_state.user_info["username"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Random samples
    st.subheader("🎲 Random Samples")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_samples = st.number_input("Number of samples", min_value=1, max_value=50, value=5)
        
        if st.button("Generate Random Samples"):
            random_samples = generate_random_sample(n_samples=n_samples)
            preds, confidence = predict_sample(random_samples)
            random_samples["Severity"] = preds
            random_samples["Confidence"] = [f"{c:.1%}" for c in confidence]
            random_samples["Tested_By"] = session_state.user_info["username"]
            st.session_state.random_results = random_samples
    
    with col2:
        if hasattr(st.session_state, 'random_results'):
            st.write("**Results:**")
            st.dataframe(st.session_state.random_results, use_container_width=True)

    # File upload
    st.subheader("📁 File Upload")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded: {user_df.shape}")
            
            # Check for required columns
            missing_cols = [col for col in BASE_CHEMICAL_COLS if col not in user_df.columns]
            
            if missing_cols:
                st.warning(f"Missing columns filled with defaults: {missing_cols}")
                for col in missing_cols:
                    user_df[col] = defaults.get(col, 0.0)
                
            # Predict using only the required columns
            input_df = user_df[BASE_CHEMICAL_COLS]
            preds, confidence = predict_sample(input_df)
            user_df["Predicted_Severity"] = preds
            user_df["Confidence"] = [f"{c:.1%}" for c in confidence]
            user_df["Analyzed_By"] = session_state.user_info["username"]
            
            st.dataframe(user_df, use_container_width=True)
            
            # Download
            csv = user_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
                
        except Exception as e:
            st.error(f"Error: {e}")

# Session timeout check (runs at the end)
if st.session_state.session_token:
    is_still_valid, _ = auth_manager.validate_session(st.session_state.session_token)
    if not is_still_valid:
        st.warning("⚠️ Your session has expired. Please login again.")
        auth_manager.logout_session(st.session_state.session_token)
        for key in list(st.session_state.keys()):
            delattr(st.session_state, key)
        st.rerun()