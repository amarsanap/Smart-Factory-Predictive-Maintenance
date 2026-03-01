import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(page_title="AI Predictive Maintenance", layout="centered")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Load the model and encoder
    with open('pdm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

try:
    model, encoder = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- USER INTERFACE ---
st.title("🛠️ Smart Factory: Predictive Maintenance")
st.markdown("Enter real-time sensor data to check for potential failure risks.")

# Sidebar for inputs
st.sidebar.header("Machine Settings")

# 1. Machine Type (We keep this for UI, even if model only uses 3 features)
m_type = st.sidebar.selectbox("Select Machine Type", encoder.classes_)

# 2. The 3 main sensor features
temp = st.sidebar.slider("Temperature (°C)", 0.0, 150.0, 65.0)
vib = st.sidebar.slider("Vibration (mms)", 0.0, 50.0, 10.0)
power = st.sidebar.number_input("Power Consumption (kW)", value=100.0)

# Main Dashboard Logic
st.subheader(f"Monitoring: {m_type}")

if st.button("Run Diagnostic Analysis"):
    # IMPORTANT: We only send the 3 features the model expects
    # This matches the 'Expected: 3' from your error message
    input_data = np.array([[temp, vib, power]])
    
    # Generate Prediction
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Display Results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Failure Probability", f"{probability:.1%}")
            
        with col2:
            if prediction == 1:
                st.error("STATUS: CRITICAL RISK")
            else:
                st.success("STATUS: HEALTHY")
        
        # Actionable Advice
        if probability > 0.8:
            st.warning("⚠️ CRITICAL: Schedule immediate inspection for " + m_type)
        elif probability > 0.2:
            st.info("ℹ️ CAUTION: Increased wear detected in " + m_type)
        else:
            st.balloons()
            st.write("Machine is operating within normal parameters.")

    except ValueError as e:
        st.error(f"Internal Model Error: {e}")
        st.info("Tip: If the model still expects a different number of features, we may need to retrain it.")

st.divider()
st.caption("Developed by Amar Sanap | AI-Powered Maintenance System 2026")