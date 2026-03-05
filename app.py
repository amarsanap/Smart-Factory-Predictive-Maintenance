import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Predictive Maintenance",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():

    with open("pdm_model.pkl","rb") as f:
        model = pickle.load(f)

    with open("encoder_pdm.pkl","rb") as f:
        encoder = pickle.load(f)

    return model, encoder


model, encoder = load_assets()


# ---------------- TITLE ----------------
st.title("🛠️ Smart Factory: AI Predictive Maintenance System")

st.write(
"Enter machine sensor data to predict equipment failure risk."
)


col1, col2 = st.columns([1,2])


# ---------------- INPUT SECTION ----------------
with col1:

    st.header("Machine Inputs")

    machine_type = st.selectbox(
        "Machine Type",
        encoder.classes_
    )

    temperature = st.slider(
        "Temperature (°C)",
        20.0,
        120.0,
        65.0
    )

    vibration = st.slider(
        "Vibration (mm/s)",
        0.0,
        10.0,
        3.0
    )


# ---------------- PREDICTION SECTION ----------------
with col2:

    st.header("Prediction Result")

    if st.button("Analyze Machine"):

        machine_encoded = encoder.transform([machine_type])[0]

        input_data = pd.DataFrame({

            "Machine_Type":[machine_encoded],
            "Temperature_C":[temperature],
            "Vibration_mms":[vibration]

        })

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        st.metric("Failure Probability", f"{probability:.2%}")

        if prediction == 1:

            st.error("⚠️ FAILURE LIKELY")

            st.warning(
            "Maintenance recommended within 7 days."
            )

        else:

            st.success("✅ Machine is Healthy")


        st.progress(int(probability*100))


# ---------------- FOOTER ----------------
st.markdown("---")

st.caption(
"Developed by Amar Sanap | COEP Technological University | AI Predictive Maintenance Project"
)
