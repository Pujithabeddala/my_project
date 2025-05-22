# app.py

import streamlit as st
import joblib
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    return joblib.load("best_model_logistic_regression.pkl")

model = load_model()

# App title
st.title("💳 Credit Card Fraud Detection")

st.markdown("Enter the transaction details to predict whether it's **fraudulent** or **legitimate**.")

# Input feature names
feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
input_data = {}

# Input fields
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

input_df = pd.DataFrame([input_data])

# Prediction button
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.markdown(
                f"""
                <div class="prediction-box">
                    🚨 Fraudulent Transaction Detected!<br>
                    Confidence: {prob:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-box">
                    ✅ Legitimate Transaction<br>
                    Confidence: {1 - prob:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:
        st.exception(f"❌ Prediction failed due to: {e}")
