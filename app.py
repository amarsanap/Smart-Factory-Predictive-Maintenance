import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Predictive Maintenance",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():
    model = pickle.load(open("pdm_model.pkl", "rb"))
    encoder = pickle.load(open("encoder_pdm.pkl", "rb"))
    return model, encoder

model, encoder = load_assets()

# ---------------- TITLE ----------------
st.title("🛠️ Smart Factory: AI Predictive Maintenance")
st.markdown("### Real-Time Machine Health Monitoring System")

# Layout
col1, col2 = st.columns([1,2])

# ---------------- INPUT PANEL ----------------
with col1:

    st.header("Machine Sensor Inputs")

    machine_type = st.selectbox(
        "Machine Type",
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

    power = st.number_input(
        "Power Consumption (kW)",
        value=100.0
    )

    debug = st.checkbox("Show Technical Debugger")

# ---------------- RESULT PANEL ----------------
with col2:

    st.header("Machine Health Result")

    if st.button("Analyze Machine"):

        # Encode machine type
        machine_encoded = encoder.transform([machine_type])[0]

        # Create dataframe EXACTLY like training data
        input_data = pd.DataFrame({
            "Machine_Type":[machine_encoded],
            "Temperature_C":[temperature],
            "Vibration_mms":[vibration],
            "Power_Consumption_kW":[power]
        })

        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        failure_prob = probability[1]

        # Metrics
        m1, m2 = st.columns(2)

        m1.metric(
            "Failure Risk",
            f"{failure_prob:.1%}"
        )

        if prediction == 1:
            m2.error("CRITICAL: FAILURE LIKELY")
        else:
            m2.success("SYSTEM STABLE")

        # Progress bar
        st.progress(int(failure_prob*100))

        # Advice
        if failure_prob > 0.8:
            st.warning(
                "🚨 Immediate maintenance required. Failure risk extremely high."
            )

        elif failure_prob > 0.4:
            st.info(
                "⚠️ Maintenance recommended soon."
            )

        else:
            st.success(
                "✅ Machine operating within safe limits."
            )

        # ---------------- FEATURE IMPORTANCE ----------------
        st.subheader("Model Feature Importance")

        try:
            importance = model.feature_importances_

            features = [
                "Machine Type",
                "Temperature",
                "Vibration",
                "Power"
            ]

            fig, ax = plt.subplots()

            ax.barh(features, importance)

            ax.set_title("Feature Importance")

            st.pyplot(fig)

        except:
            st.write("Feature importance unavailable for this model.")

        # ---------------- DEBUGGER ----------------
        if debug:

            st.divider()

            st.subheader("Technical Debugger")

            st.write("Input Data Sent To Model:")
            st.write(input_data)

            st.write(
                f"Probability → Healthy: {probability[0]:.4f} | Failure: {probability[1]:.4f}"
            )

# ---------------- FOOTER ----------------
st.divider()

st.caption(
"Developed by Amar Sanap | Smart Factory Predictive Maintenance System"
)
