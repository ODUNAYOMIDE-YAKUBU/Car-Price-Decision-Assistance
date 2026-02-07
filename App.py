import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import requests


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
# Google Drive model download
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILENAME = "car_price_prediction_model.joblib"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# Your direct download URL
MODEL_URL = "https://drive.google.com/uc?export=download&id=1g_atyU9dC-R_7Ace7O07x9_lqehKfzfz"

def _download_file_from_gdrive(url: str, dest_path: str) -> None:
    """
    Downloads large files from Google Drive by handling the 'download_warning' token.
    """
    session = requests.Session()

    # First request
    response = session.get(url, stream=True, timeout=180)
    response.raise_for_status()

    # Check for confirmation token in cookies
    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    # If token exists, confirm download
    if token:
        url = url + "&confirm=" + token
        response = session.get(url, stream=True, timeout=180)
        response.raise_for_status()

    # Write to file
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB
            if chunk:
                f.write(chunk)


def download_from_gdrive(url: str, dest_path: str) -> None:
    """
    Downloads large files from Google Drive by handling the 'download_warning' token.
    """
    session = requests.Session()
    resp = session.get(url, stream=True, timeout=180)
    resp.raise_for_status()

    token = None
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        confirm_url = url + "&confirm=" + token
        resp = session.get(confirm_url, stream=True, timeout=180)
        resp.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def looks_like_html(path: str) -> bool:
    """
    Google Drive sometimes returns an HTML confirmation page instead of the file.
    This detects that case.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(512)
        return b"<html" in head.lower() or b"<!doctype html" in head.lower()
    except Exception:
        return False


@st.cache_resource
def load_model():
    # Download if missing
    if looks_like_html(MODEL_PATH):
    # Delete the bad file so the next reboot triggers a fresh download
        try:
            os.remove(MODEL_PATH)
        except Exception:
            pass

    st.error(
        "Downloaded file is HTML (Google Drive confirmation page), not a real .joblib model.\n\n"
        "Fix: Make the Drive file 'Anyone with the link (Viewer)' and reboot the app."
    )
    st.stop()

rf_model = load_model()
# Final safety check
if not hasattr(rf_model, "predict"):
    st.error(f"Loaded object is not a model. Type: {type(rf_model)}")
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

# Brand + Model (dropdown if CSV uploaded, else manual)
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

asking_price = st.number_input("Asking Price (₦)", min_value=0, max_value=50_000_000, value=500_000, step=10_000)


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

    # Defensive checks
    if engine == 0:
        st.error("Engine cannot be 0.")
        st.stop()

    power_per_cc = float(max_power) / float(engine)
    log_km_driven = float(np.log1p(km_driven))

    # IMPORTANT: input_df must match training schema
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
        st.metric("Predicted Fair Price", f"₦{predicted_price:,.0f}")
        st.metric("Your Asking Price", f"₦{asking_price:,.0f}")
        st.write("### Decision")
        st.write(decision)

    except Exception as e:
        st.error("Prediction failed. This usually means your model expects different input columns.")
        st.code(str(e))
        st.stop()



