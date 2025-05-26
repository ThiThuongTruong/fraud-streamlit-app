import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load model and scaler
autoencoder = load_model("autoencoder_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load processed data
df = pd.read_csv("data_processed.csv")

# App title
st.title("🔍 Provider-Level Fraud Detection")

# Select a provider from dropdown
provider_ids = df["Provider"].unique()
selected_provider = st.selectbox("Select a Provider:", provider_ids)

# Extract the row for selected provider
row = df[df["Provider"] == selected_provider]
true_label = row["PotentialFraud"].values[0]

# Get features for prediction
X = row.select_dtypes(include=[np.number]).drop(
    columns=["Label", "ReconstructionError", "PredictedLabel"], errors="ignore"
)
X_scaled = scaler.transform(X)
X_pred = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

# Define threshold (same as training)
threshold = 0.05
prediction = "⚠️ Fraud" if mse[0] > threshold else "✅ Not Fraud"

# Show results
st.subheader("Result:")
st.write(f"**Prediction**: {prediction}")
st.write(f"**Reconstruction Error**: {mse[0]:.6f}")
