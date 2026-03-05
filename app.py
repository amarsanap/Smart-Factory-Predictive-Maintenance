import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")

# ---------------- LOAD MODEL & ASSETS ----------------
@st.cache_resource
def load_assets():
    # Load the model dictionary
    with open("pdm_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    # Extract specific items from the dictionary
    actual_model = model_data["model"]
    feature_names = model_data["features"]

    # Load the label encoder
    with open("encoder_pdm.pkl", "rb") as f:
        encoder = pickle.load(f)

    return actual_model, encoder, feature_names

# Unpack all three needed components
model, encoder, model_features = load_assets()

# ---------------- UI ----------------
st.title("🛠️ Smart Factory: AI Predictive Maintenance System")
st.write("Enter real-time machine sensor data to predict possible equipment failure.")

col1, col2 = st.columns([1,2])

# ---------------- INPUT SECTION ----------------
with col1:
    st.header("Machine Sensor Inputs")

    # Use the classes from the loaded encoder for the dropdown
    machine_type = st.selectbox(
        "Select Machine Type",
        encoder.classes_
    )

    temperature = st.slider(
        "Temperature (°C)",
        20.0, 120.0, 65.0
    )

    vibration = st.slider(
        "Vibration (mm/s)",
        0.0, 10.0, 3.0
    )

# ---------------- RESULT SECTION ----------------
with col2:
    st.header("Prediction Result")

    if st.button("Analyze Machine Health"):
        try:
            # 1. Encode the categorical input
            machine_encoded = encoder.transform([machine_type])[0]

            # 2. Create DataFrame with user inputs
            input_data = pd.DataFrame({
                "Machine_Type": [machine_encoded],
                "Temperature_C": [temperature],
                "Vibration_mms": [vibration]
            })

            # 3. CRITICAL: Match the column order used during training
            input_data = input_data[model_features]

            # 4. Make Predictions
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            # 5. Display Results
            st.metric("Failure Risk Probability", f"{probability:.2%}")

            if prediction == 1:
                st.error("⚠️ FAILURE LIKELY")
                st.warning("Maintenance recommended within the next 7 days.")
            else:
                st.success("✅ MACHINE HEALTHY")
                st.info("No immediate maintenance required.")

            st.progress(int(probability * 100))

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed by Amar Sanap | AI Predictive Maintenance Project 2026")
