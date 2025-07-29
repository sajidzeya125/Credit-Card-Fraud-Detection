import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('Credit Card Fraud Detection')

try:
    model = joblib.load('xgb_fraud_model.pkl')
    scaler = joblib.load('amount_time_scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
            'V28', 'Amount']

input_type = st.sidebar.radio("Select Input Method:", ("Upload CSV", "Manual Input"))

if input_type == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        try:
            df[['Time','Amount']] = scaler.transform(df[['Time','Amount']])
            preds = model.predict(df[features])
            df['Prediction'] = np.where(preds == 1, 'Fraud', 'Not Fraud')
            st.write(df.head())
            st.success(f"Total {len(df)} transactions; Fraud detected: {(preds == 1).sum()}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

elif input_type == "Manual Input":
    inputs = {}
    for feat in features:
        inputs[feat] = st.number_input(feat, value=0.0)
    input_df = pd.DataFrame([inputs])
    try:
        input_df[['Time','Amount']] = scaler.transform(input_df[['Time','Amount']])
        pred = model.predict(input_df)[0]
        st.write("Prediction:", "FRAUD" if pred==1 else "NOT FRAUD")
    except Exception as e:
        st.error(f"Error during prediction: {e}")