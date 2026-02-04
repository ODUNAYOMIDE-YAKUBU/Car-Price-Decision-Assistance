# 🚗 Car Price Decision Assistant

An end-to-end machine learning application that predicts the **fair selling price of used cars** and provides **Buy / Negotiate / Avoid** recommendations by comparing model-estimated value with the seller’s asking price. The system is deployed as an interactive **Streamlit web app** for real-world decision support.

---

## 📌 Project Overview

Pricing used cars is challenging due to non-linear depreciation, brand influence, and performance variations. Buyers often lack a reliable benchmark to determine whether a listed price is fair.

This project addresses that challenge by building a **data-driven decision assistant** that:
- Predicts a fair market price for a car
- Compares it against the seller’s asking price
- Outputs a clear, actionable decision

The focus is not only on prediction accuracy, but also on **decision intelligence and deployment readiness**.

---

## 📊 Dataset Description

The dataset contains structured information about used vehicles, including:

- Vehicle details: brand, model, fuel type, transmission
- Usage & condition: vehicle age, kilometers driven
- Performance metrics: engine capacity, max power, mileage, seating
- Target variable: `selling_price`

This structure makes the problem well-suited for **supervised regression on tabular data**.

---

## 🧠 Algorithm Selection

### Primary Algorithm: **Random Forest Regressor**

The final model used in this project is a **Random Forest Regressor**, implemented within an end-to-end sklearn Pipeline.

#### Why Random Forest?
- Captures **non-linear relationships** in car pricing
- Automatically models **feature interactions** (e.g., age × brand × power)
- Robust to outliers caused by luxury vehicles
- Performs strongly on structured, real-world business data

### Baseline Model
A **Linear Regression** model was trained as a baseline to assess problem complexity. While interpretable, it underperformed due to the non-linear nature of car price depreciation.

---

## 🔧 Model Pipeline Architecture

To ensure consistency between training and deployment, the model was built using a single sklearn Pipeline:

- **Categorical features:** One-Hot Encoded  
- **Numerical features:** Passed through with engineered transformations  
- **Estimator:** Random Forest Regressor  

This approach eliminates training–deployment mismatch and ensures reliable inference in production.

---

## ⚙️ Feature Engineering

To improve model performance and stability, the following features were engineered:

- **Power-to-engine ratio (`power_per_cc`)**  
  Captures performance efficiency rather than raw engine size.

- **Log-transformed kilometers driven (`log_km_driven`)**  
  Reduces skew from extreme usage values.

- **Log-transformed target (`log_selling_price`)**  
  Stabilizes variance and improves model learning.

---

## 📈 Model Evaluation & Accuracy

The models were evaluated using a train/test split and the following metrics:

- **MAE (Mean Absolute Error)** – average prediction error  
- **RMSE (Root Mean Squared Error)** – penalizes large errors  
- **R² Score** – proportion of variance explained  

### Performance Summary
- Random Forest consistently outperformed Linear Regression across all metrics
- Predictions showed **low bias** in budget and premium price segments
- Slight underestimation observed for ultra-luxury vehicles due to limited samples

> **Result:**  
The Random Forest model achieved strong predictive performance and demonstrated reliable generalization for real-world pricing decisions.

*(Exact metric values can be found in the evaluation notebook and visual diagnostics.)*

---

## 🎯 Decision Intelligence Layer

Rather than returning a single numeric prediction, the system applies a **decision layer**:

- A fair price range is defined around the predicted value
- The seller’s asking price is compared to this range
- The output is a clear recommendation:

| Outcome | Decision |
|------|--------|
| Asking price < fair value | ✅ Strong Buy |
| Asking price within range | 🟦 Fairly Priced |
| Asking price > fair value | ⚠️ Overpriced |

This makes the model **actionable for end users**, not just accurate.

---

## 🌐 Deployment

The trained pipeline is deployed using **Streamlit**, providing:
- Interactive user inputs
- Real-time predictions
- Clear pricing decisions
- Robust error handling
- Deployment-ready architecture (local & cloud)

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib
- Git & GitHub

---

## ▶️ Run the App Locally

```bash
pip install -r requirements.txt
python -m streamlit run App.py
