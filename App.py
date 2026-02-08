import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import requests
import gdown

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
# Model download settings
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILENAME = "car_price_prediction_model.joblib"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

#MODEL_URL = "https://drive.google.com/uc?export=download&id=1g_atyU9dC-R_7Ace7O07x9_lqehKfzfz"
FILE_ID = "1g_atyU9dC-R_7Ace7O07x9_lqehKfzfz"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

def looks_like_html(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(512).lower()
        return b"<html" in head or b"<!doctype html" in head
    except Exception:
        return False


@st.cache_resource
def load_model():
    # Download if missing
    if not os.path.exists(MODEL_PATH):
        st.info("Model not found locally. Downloading from Google Drive... ⏳")

        # gdown will handle the big-file "Download anyway" confirmation
        out = gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

        if out is None or not os.path.exists(MODEL_PATH):
            st.error("Download failed. Google Drive may be rate-limiting or access is restricted.")
            st.stop()

        st.success("Model downloaded successfully ✅")

    # Show file size
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    st.write(f"Downloaded model size: **{size_mb:.2f} MB**")

    # Validate
    if size_mb < 5.0 or looks_like_html(MODEL_PATH):
        # Bad download (HTML or tiny file). Delete and stop.
        try:
            os.remove(MODEL_PATH)
        except Exception:
            pass

        st.error(
            "Downloaded file is not a valid model binary (looks like HTML or is too small).\n\n"
            "Cause: Google Drive returned a confirmation/permission page.\n"
            "Fix: Ensure the file is shared as 'Anyone with the link (Viewer)' and reboot."
        )
        st.stop()

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        if not hasattr(model, "predict"):
            st.error(f"Loaded object is not a trained model. Type: {type(model)}")
            st.stop()

        st.success("Model loaded successfully ✅")
        return model

    except Exception as e:
        st.error("joblib.load() failed (model may be incompatible with current sklearn version).")
        st.code(repr(e))
        st.stop()


rf_model = load_model()

# -----------------------------
# Sidebar: Model information
# -----------------------------
with st.sidebar:
    st.header("ℹ️ Model Info")

    with st.expander("How the prediction works", expanded=False):
        st.markdown("""
        - **Model:** Random Forest Regressor  
        - **Target:** log(selling_price)  
        - **Feature engineering:**  
          - power_per_cc = max_power / engine  
          - log_km_driven = log1p(km_driven)  
        """)

    with st.expander("Decision rules", expanded=False):
        st.markdown("""
        - ✅ **Strong Buy:** asking < 90% of predicted value  
        - 🟦 **Fairly Priced:** within ±10% of predicted value  
        - ⚠️ **Overpriced:** asking > 110% of predicted value  
        """)

    with st.expander("Deployment notes", expanded=False):
        st.markdown("""
        - Model is downloaded at runtime and cached  
        - Environment is pinned for compatibility  
        """)


# Final safety check
if not hasattr(rf_model, "predict"):
    st.error(f"Loaded object is not a trained model. Loaded type: {type(rf_model)}")
    st.stop()


# -----------------------------
# Optional: Upload CSV for dropdowns
# -----------------------------
st.subheader("📁 Optional: Upload dataset for Brand/Model dropdowns")
uploaded_csv = st.file_uploader("Upload car dataset CSV (optional)", type=["csv"])

df = None
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        st.success("CSV loaded ✅")
    except Exception as e:
        st.warning("CSV failed to load. You can still proceed with manual inputs.")
        st.code(str(e))
        df = None


# -----------------------------
# Inputs
# -----------------------------
st.subheader("🧾 Enter Car Details")

col1, col2 = st.columns(2)

# Brand + Model
if df is not None and "brand" in df.columns and "model" in df.columns:
    brand_list = sorted(df["brand"].dropna().astype(str).unique().tolist())
    brand = col1.selectbox("Brand", brand_list)

    model_list = sorted(df.loc[df["brand"] == brand, "model"].dropna().astype(str).unique().tolist())
    if len(model_list) == 0:
        model_list = sorted(df["model"].dropna().astype(str).unique().tolist())

    model = col2.selectbox("Model", model_list)
else:
    brand = col1.text_input("Brand (e.g., hyundai)", value="hyundai").strip().lower()
    model = col2.text_input("Model (e.g., i20)", value="i20").strip().lower()

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
# Decision Logic
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
    if engine == 0:
        st.error("Engine cannot be 0.")
        st.stop()

    power_per_cc = float(max_power) / float(engine)
    log_km_driven = float(np.log1p(km_driven))

    # IMPORTANT: must match training schema
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
        pred_log = rf_model.predict(input_df)[0]
        predicted_price = float(np.expm1(pred_log))
        decision = decision_rule(asking_price, predicted_price, tolerance=0.10)

        st.success("Prediction Complete ✅")
        st.metric("Predicted Fair Price", f"${predicted_price:,.0f}")
        st.metric("Your Asking Price", f"${asking_price:,.0f}")
        st.write("### Decision")
        st.write(decision)

    except Exception as e:
        st.error("Prediction failed. This usually means the model expects different input columns.")
        st.code(str(e))
        st.stop()


