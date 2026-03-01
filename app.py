import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        with open('pdm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, encoder = load_assets()

if model is None or encoder is None:
    st.stop()

# --- USER INTERFACE ---
st.title("🛠️ Smart Factory: AI Predictive Maintenance")
st.markdown("### Real-time Machine Health Monitoring")

# Layout with two columns: Inputs on the left, Results on the right
col_input, col_result = st.columns([1, 2])

with col_input:
    st.header("Sensor Inputs")
    m_type = st.selectbox("Machine Type", encoder.classes_)
    
    # These 3 must be in the EXACT order your model was trained on
    temp = st.slider("Temperature (°C)", 0.0, 150.0, 65.0)
    vib = st.slider("Vibration (m/s²)", 0.0, 50.0, 5.0)
    power = st.number_input("Power Consumption (kW)", value=100.0)
    
    # Toggle for technical users
    show_debug = st.checkbox("Show Model Debugger")

with col_result:
    st.header(f"Diagnostic Status: {m_type}")
    
    if st.button("Analyze Machine Health"):
        # 1. Prepare input (Ensure order matches training: Temp, Vib, Power)
        input_data = np.array([[temp, vib, power]])
        
        # 2. Prediction logic
        prediction = model.predict(input_data)[0]
        # Get probability of failure (Class 1)
        raw_probs = model.predict_proba(input_data)[0]
        failure_prob = raw_probs[1]
        
        # 3. Visual Gauges/Metrics
        m1, m2 = st.columns(2)
        m1.metric("Risk Level", f"{failure_prob:.1%}")
        
        if prediction == 1:
            m2.error("CRITICAL: FAILURE LIKELY")
        else:
            m2.success("STABLE: HEALTHY")
            
        # 4. Progress bar for visual risk
        st.progress(failure_prob)
        
        # 5. Actionable Guidance
        if failure_prob > 0.8:
            st.warning(f"🚨 IMMEDIATE ACTION REQUIRED: Sensor readings for {m_type} indicate imminent mechanical failure.")
        elif failure_prob > 0.4:
            st.info(f"⚠️ MAINTENANCE ADVISORY: Stress patterns detected. Schedule an inspection for {m_type} within 7 days.")
        else:
            st.balloons()
            st.write("✅ Machine is performing within optimal parameters.")

        # --- DEBUGGER SECTION ---
        if show_debug:
            st.divider()
            st.subheader("🔍 Technical Debugger")
            st.write("Input Vector sent to AI:", input_data)
            st.write(f"Raw Model Probability Distribution: [Healthy: {raw_probs[0]:.4f}, Failure: {raw_probs[1]:.4f}]")
            st.write("Tip: If probability is 0% even with high vibration, check if your model was trained on 'Scaled' data.")

st.divider()
st.caption("Developed by Amar Sanap | COEP Tech University | Predictive Maintenance System v1.0")
