import streamlit as st
import pickle
import numpy as np

# Load your model
model = pickle.load(open('pdm_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

st.title("Factory Health Monitoring System")
st.write("Enter sensor data to predict failure risk.")

# User Inputs
m_type = st.selectbox("Machine Type", encoder.classes_)
temp = st.slider("Temperature (°C)", 20.0, 150.0, 65.0)
vib = st.slider("Vibration (mms)", 0.0, 50.0, 10.0)
pwr = st.number_input("Power Consumption (kW)", value=100.0)

if st.button("Run Diagnostics"):
    m_encoded = encoder.transform([m_type])[0]
    features = np.array([[m_encoded, temp, vib, pwr]])

    prob = model.predict_proba(features)[0][1]

    if prob > 0.5:
        st.error(f"⚠️ ALERT: High Risk of Failure ({prob:.1%})")
    else:
        st.success(f"✅ System Healthy ({prob:.1%})")