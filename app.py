import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")

# ---------------- LOAD MODEL & ENCODER ----------------
@st.cache_resource
def load_assets():
    try:
        with open('pdm_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('encoder_pdm.pkl', 'rb') as f:
            encoder = pickle.load(f)

        return model, encoder

    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None


model, encoder = load_assets()

if model is None or encoder is None:
    st.stop()

# ---------------- UI ----------------
st.title("🛠️ Smart Factory: AI Predictive Maintenance")
st.markdown("### Real-Time Machine Health Monitoring System")

col_input, col_result = st.columns([1,2])

# ---------------- INPUT SECTION ----------------
with col_input:

    st.header("Sensor Inputs")

    m_type = st.selectbox(
        "Machine Type",
        encoder.classes_
    )

    temp = st.slider(
        "Temperature (°C)",
        0.0, 150.0, 65.0
    )

    vib = st.slider(
        "Vibration (mm/s)",
        0.0, 50.0, 5.0
    )

    show_debug = st.checkbox("Show Technical Debugger")

# ---------------- RESULT SECTION ----------------
with col_result:

    st.header("Diagnostic Result")

    if st.button("Analyze Machine Health"):

        # Encode machine type
        m_type_encoded = encoder.transform([m_type])[0]

        # Prepare input data (ONLY 3 FEATURES)
        input_data = pd.DataFrame({
            "Machine_Type":[m_type_encoded],
            "Temperature_C":[temp],
            "Vibration_mms":[vib]
        })

        # Prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        failure_prob = probabilities[1]

        # Metrics
        col1, col2 = st.columns(2)

        col1.metric("Failure Risk", f"{failure_prob:.1%}")

        if prediction == 1:
            col2.error("CRITICAL: FAILURE LIKELY")
        else:
            col2.success("SYSTEM STABLE")

        # Risk Progress Bar
        st.progress(int(failure_prob*100))

        # Maintenance Advice
        if failure_prob > 0.8:
            st.warning(
                "🚨 IMMEDIATE ACTION REQUIRED: Machine failure likely within 7 days."
            )

        elif failure_prob > 0.4:
            st.info(
                "⚠️ Maintenance Recommended: Schedule inspection soon."
            )

        else:
            st.success(
                "✅ Machine operating within safe limits."
            )

        # ---------------- Feature Importance ----------------
        st.subheader("Feature Importance")

        try:
            importance = model.feature_importances_

            features = [
                "Machine_Type",
                "Temperature",
                "Vibration"
            ]

            fig, ax = plt.subplots()

            ax.barh(features, importance)

            ax.set_title("Model Feature Importance")

            st.pyplot(fig)

        except:
            st.write("Feature importance not available for this model.")

        # ---------------- DEBUGGER ----------------
        if show_debug:

            st.divider()

            st.subheader("Technical Debugger")

            st.write("Input Data Sent To Model:")
            st.write(input_data)

            st.write(
                f"Probability Distribution → Healthy: {probabilities[0]:.4f} | Failure: {probabilities[1]:.4f}"
            )

# ---------------- FOOTER ----------------
st.divider()

st.caption(
"Developed by Amar Sanap | COEP Technological University Pune | 2026"
)
