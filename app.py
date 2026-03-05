import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():

    with open("pdm_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("encoder_pdm.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, encoder


model, encoder = load_assets()

# ---------------- UI ----------------
st.title("🛠️ Smart Factory: AI Predictive Maintenance")
st.write("Enter machine sensor data to predict possible failure.")

col1, col2 = st.columns([1,2])

# ---------------- INPUT ----------------
with col1:

    st.header("Machine Settings")

    machine_type = st.selectbox(
        "Select Machine Type",
        encoder.classes_
    )

    temperature = st.slider(
        "Temperature (°C)",
        0.0,150.0,65.0
    )

    vibration = st.slider(
        "Vibration (mm/s)",
        0.0,50.0,5.0
    )

# ---------------- RESULT ----------------
with col2:

    st.header("Prediction Result")

    if st.button("Analyze Machine Health"):

        machine_encoded = encoder.transform([machine_type])[0]

        # IMPORTANT: Match model features exactly
        input_data = pd.DataFrame(
            [[machine_encoded, temperature, vibration]],
            columns=[
                "Machine_Type",
                "Temperature",
                "Vibration"
            ]
        )

        try:

            prediction = model.predict(input_data, validate_features=False)[0]
            prob = model.predict_proba(input_data, validate_features=False)[0][1]

            st.metric("Failure Risk", f"{prob:.2%}")

            if prediction == 1:
                st.error("⚠️ FAILURE LIKELY")
            else:
                st.success("✅ MACHINE HEALTHY")

            st.progress(int(prob*100))

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed by Amar Sanap | Smart Factory Predictive Maintenance")
