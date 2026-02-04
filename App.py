import os
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Car Price Decision Assistant",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Decision Assistant")
st.caption("Predict fair car price and get a Buy | Negotiate | Avoid decision.")


# -----------------------------
# Helper: Robust paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Change this to your actual model file name (exact spelling)
MODEL_FILENAME = "car_price_prediction_model.joblib"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# Try to auto-detect a CSV for dropdown options
csv_files = glob.glob(os.path.join(BASE_DIR, "*.csv"))
DEFAULT_CSV_PATH = csv_files[0] if csv_files else None


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model(path: str):
    obj = joblib.load(path)
    return obj


if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found.\n\nExpected: `{MODEL_FILENAME}` inside:\n{BASE_DIR}\n\n"
        "Fix: Put the .joblib model file in the same folder as App.py "
        "or rename MODEL_FILENAME to the correct file name."
    )
    st.stop()

rf_model = load_model(MODEL_PATH)

# Guard: ensure it's a real model/pipeline
if not hasattr(rf_model, "predict"):
    st.error(
        f"The loaded .joblib is NOT a trained model.\n\n"
        f"Loaded type: `{type(rf_model)}`\n\n"
        "Fix: In your notebook, save the trained pipeline like:\n"
        "`joblib.dump(rf_pipeline, 'car_price_prediction_model.joblib')`"
    )
    st.stop()


# -----------------------------
# Load data (only for dropdowns)
# -----------------------------
df = None
if DEFAULT_CSV_PATH:
    try:
        df = pd.read_csv(DEFAULT_CSV_PATH)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    except Exception as e:
        st.warning(f"CSV found but failed to load: {os.path.basename(DEFAULT_CSV_PATH)}\n\n{e}")
        df = None


# -----------------------------
# Sidebar / Inputs
# -----------------------------
st.subheader("🔧 Enter Car Details")

# If CSV is loaded, populate dropdowns from it; else use manual text inputs
if df is not None and "brand" in df.columns and "model" in df.columns:
    brand_list = sorted(df["brand"].dropna().unique().tolist())
    brand = st.selectbox("Brand", brand_list)

    # Filter model choices by brand if possible
    model_list = sorted(df.loc[df["brand"] == brand, "model"].dropna().unique().tolist())
    if len(model_list) == 0:
        model_list = sorted(df["model"].dropna().unique().tolist())

    model = st.selectbox("Model", model_list)
else:
    st.info("No CSV loaded for dropdowns. Using manual inputs for Brand & Model.")
    brand = st.text_input("Brand (e.g., hyundai)", value="hyundai").strip().lower()
    model = st.text_input("Model (e.g., i20)", value="i20").strip().lower()


seller_type = st.selectbox("Seller Type", ["individual", "dealer", "trustmark_dealer"])
fuel_type = st.selectbox("Fuel Type", ["petrol", "diesel", "cng", "lpg", "electric"])
transmission_type = st.selectbox("Transmission Type", ["manual", "automatic"])

vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=300000, value=50000, step=1000)

mileage = st.number_input("Mileage (km/l)", min_value=0.0, max_value=60.0, value=18.0, step=0.1)
engine = st.number_input("Engine (CC)", min_value=500, max_value=6000, value=1200, step=50)
max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=1000.0, value=80.0, step=1.0)

seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10], index=2)

asking_price = st.number_input("Asking Price ($)", min_value=0, max_value=50_000_000, value=500_000, step=10_000)


# -----------------------------
# Decision function
# -----------------------------
def decision_rule(asking: float, predicted: float, tolerance: float = 0.10) -> str:
    lower = predicted * (1 - tolerance)
    upper = predicted * (1 + tolerance)

    if asking < lower:
        return "✅ Strong Buy (Undervalued)"
    elif asking > upper:
        return "⚠️ Overpriced (Negotiate / Avoid)"
    else:
        return "🟦 Fairly Priced"


# -----------------------------
# Predict
# -----------------------------
st.divider()
if st.button("Evaluate Price 🚀"):

    # Feature engineering
    power_per_cc = float(max_power) / float(engine) if engine else 0.0
    log_km_driven = np.log1p(km_driven)

    # IMPORTANT: Must match the training schema
    input_df = pd.DataFrame({
        "brand": [brand],
        "model": [model],
        "seller_type": [seller_type],
        "fuel_type": [fuel_type],
        "transmission_type": [transmission_type],
        "vehicle_age": [vehicle_age],
        "km_driven": [km_driven],
        "mileage": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "seats": [seats],
        "power_per_cc": [power_per_cc],
        "log_km_driven": [log_km_driven]
    })

    try:
        # Model predicts log(price). Convert back to normal price
        pred_log = rf_model.predict(input_df)[0]
        predicted_price = float(np.expm1(pred_log))

        decision = decision_rule(asking_price, predicted_price, tolerance=0.10)

        st.success("Prediction Complete ✅")
        st.metric("Predicted Fair Price", f"${predicted_price:,.0f}")
        st.metric("Your Asking Price", f"${asking_price:,.0f}")
        st.write("### Decision")
        st.write(decision)

    except Exception as e:
        st.error("Prediction failed. This usually means your input columns don't match the model training schema.")
        st.code(str(e))
        st.stop()
