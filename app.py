import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model và scaler đã huấn luyện
model = load_model("autoencoder_model.h5")
scaler = joblib.load("scaler_autoencoder.pkl")

st.title("🔍 Provider Fraud Detection App")
st.markdown("Upload a new dataset to detect potential fraudulent providers.")

uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Dự phòng giữ ID
    if 'ProviderID' in df.columns:
        id_col = df['ProviderID']
    else:
        id_col = df.index

    # Tiền xử lý
    df_processed = df.select_dtypes(include=[np.number])
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_processed.dropna(axis=1, how='all', inplace=True)
    df_processed = df_processed.loc[:, df_processed.nunique() > 1]
    df_processed = df_processed.fillna(df_processed.mean())

    # Chuẩn hóa
    X_scaled = scaler.transform(df_processed)

    # Dự đoán với autoencoder
    reconstructions = model.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)

    # Threshold
    threshold = np.percentile(mse, 95)
    is_fraud = mse > threshold

    # Tạo kết quả
    result_df = pd.DataFrame({
        'ProviderID': id_col,
        'fraud_score': mse,
        'is_fraud': is_fraud
    })

    st.markdown("### 📋 Detection Results Preview")
    st.dataframe(result_df.head(10))

    st.markdown(f"🔴 **Threshold (95th percentile):** {threshold:.6f}")
    st.metric("⚠️ Fraudulent Providers Detected", is_fraud.sum())

    # Tải file kết quả
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results", data=csv, file_name="fraud_detection_results.csv", mime="text/csv")
