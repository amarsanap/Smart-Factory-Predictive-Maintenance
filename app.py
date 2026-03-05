import streamlit as st
import pickle
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
st.title("🛠️ Smart Factory: AI Predictive Maintenance System")
st.write("Enter real-time machine sensor data to predict possible equipment failure.")

col1, col2 = st.columns([1,2])

# ---------------- INPUT SECTION ----------------
with col1:

    st.header("Machine Sensor Inputs")

    machine_type = st.selectbox(
        "Select Machine Type",
        encoder.classes_
    )

    air_temp = st.slider(
        "Air Temperature (K)",
        295, 310, 300
    )

    process_temp = st.slider(
        "Process Temperature (K)",
        305, 325, 310
    )

    rotational_speed = st.slider(
        "Rotational Speed (RPM)",
        1100, 3000, 1500
    )

    torque = st.slider(
        "Torque (Nm)",
        20, 80, 40
    )

    tool_wear = st.slider(
        "Tool Wear (min)",
        0, 250, 50
    )

# ---------------- RESULT SECTION ----------------
with col2:

    st.header("Prediction Result")

    if st.button("Analyze Machine Health"):

        # Encode machine type
        machine_encoded = encoder.transform([machine_type])[0]

        # Create dataframe exactly like training features
        input_data = pd.DataFrame({
            "Type":[machine_encoded],
            "Air temperature [K]":[air_temp],
            "Process temperature [K]":[process_temp],
            "Rotational speed [rpm]":[rotational_speed],
            "Torque [Nm]":[torque],
            "Tool wear [min]":[tool_wear]
        })

        try:

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.metric("Failure Risk", f"{probability:.2%}")

            if prediction == 1:
                st.error("⚠️ FAILURE LIKELY")
                st.warning("Maintenance recommended immediately.")
            else:
                st.success("✅ MACHINE HEALTHY")

            st.progress(int(probability * 100))

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed by Amar Sanap | COEP Technological University | AI Predictive Maintenance Project 2026")
