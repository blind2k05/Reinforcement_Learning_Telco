import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from stable_baselines3 import PPO

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Telco AI Co-Pilot", page_icon="📡", layout="wide")

# --- MEMUAT MODEL (Di-cache agar aplikasi tidak lambat) ---
@st.cache_resource
def load_models():
    # 1. Memuat AI Agent (RL)
    rl_model = PPO.load("ppo_telco_churn_model")
    
    # 2. Melatih ulang Random Forest kilat untuk prediksi
    train_df = pd.read_csv('train_telco.csv')
    
    # Memisahkan fitur dan label
    features_df = train_df.drop(columns=['Churn'])
    X_train = features_df.values
    y_train = train_df['Churn'].values
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Mengambil pelanggan baris pertama sebagai "Manekin" dasar
    baseline_profile = X_train[0].copy()
    
    # BARU: Mengambil daftar nama kolom agar kita tahu persis indeks posisinya
    feature_columns = features_df.columns.tolist()
    
    return rl_model, rf_model, baseline_profile, feature_columns

# Memanggil 4 output dari fungsi
rl_model, rf_model, baseline_profile, feature_cols = load_models()

# --- HEADER APLIKASI ---
st.title("📡 Telco Retention AI Co-Pilot")
st.markdown("Asisten cerdas berbasis **Reinforcement Learning** untuk memandu keputusan retensi pelanggan secara *real-time*.")
st.divider()

# --- LAYOUT DUA KOLOM ---
col1, col2 = st.columns([1, 2])

#--- KOLOM KIRI: INPUT DATA PELANGGAN ---
with col1:
    st.subheader("👤 Profil Pelanggan")
    st.info("Pilih ID pelanggan dasar untuk memuat 27 karakteristik tersembunyi (tipe kontrak, layanan, dll).")
    
    # BARU: Dropdown untuk mengganti "Manekin"
    customer_id = st.selectbox("Pilih ID Base Profile:", range(0, 50), index=0)
    
    st.divider()
    st.markdown("**Modifikasi Profil:**")
    tenure = st.slider("Lama Berlangganan (Bulan)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    monthly_charges = st.slider("Tagihan Bulanan (Skala Normalisasi)", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    total_charges = st.slider("Total Tagihan (Skala Normalisasi)", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
    
    # Memuat fitur dan mencari indeks dinamis (pastikan X_train dan feature_cols bisa diakses dari fungsi load_models)
    train_df = pd.read_csv('train_telco.csv')
    features_df = train_df.drop(columns=['Churn'])
    X_train_full = features_df.values
    
    user_input = np.zeros((1, len(feature_cols)), dtype=np.float32)
    
    # Memasukkan profil dasar sesuai ID yang dipilih pengguna
    user_input[0] = X_train_full[customer_id].copy()
    
    try:
        idx_tenure = feature_cols.index('tenure')
        idx_monthly = feature_cols.index('MonthlyCharges')
        idx_total = feature_cols.index('TotalCharges')
        
        user_input[0, idx_tenure] = tenure
        user_input[0, idx_monthly] = monthly_charges
        user_input[0, idx_total] = total_charges
    except ValueError as e:
        st.error(f"Kesalahan kolom: {e}")

# --- KOLOM KANAN: ANALISIS & REKOMENDASI AI ---
with col2:
    st.subheader("🧠 Analisis & Keputusan AI")
    
    # 1. Prediksi Risiko (Supervised Learning)
    churn_prob = rf_model.predict_proba(user_input)[0][1] * 100
    
    st.markdown("#### 1. Predictive Risk (Random Forest)")
    if churn_prob > 50:
        st.error(f"⚠️ Risiko Churn Tinggi: {churn_prob:.1f}%")
    else:
        st.success(f"✅ Pelanggan Aman (Risiko: {churn_prob:.1f}%)")
        
    st.divider()
    
    # 2. Rekomendasi Tindakan (Reinforcement Learning)
    st.markdown("#### 2. Prescriptive Action (RL Agent)")
    
    action, _ = rl_model.predict(user_input[0], deterministic=True)
    
    action_dict = {
        0: ("⚪ Tidak Ada Tindakan (Do Nothing)", "Biaya: $0 | Pelanggan loyal, tidak perlu bakar uang promo.", "normal"),
        1: ("🟡 Berikan Diskon Tagihan", "Biaya: $5 | Probabilitas retensi naik 60%.", "warning"),
        2: ("🔵 Berikan Ekstra Kuota", "Biaya: $3 | Efektif untuk pelanggan yang butuh data tambahan.", "info"),
        3: ("🔴 Panggilan Proaktif CS", "Biaya: $15 | Tindakan darurat, peluang selamat 80%.", "error")
    }
    
    tindakan, detail, status = action_dict[int(action)]
    
    st.success(f"**Tindakan Terbaik:** {tindakan}")
    st.caption(f"💡 **Alasan Bisnis:** {detail}")

    st.divider()
    st.markdown("#### 💬 Draf Pesan Otomatis (Integrasi Lanjutan)")
    if action == 0:
        st.text_area("Template WhatsApp:", "Halo! Terima kasih telah setia menjadi pelanggan kami. Nikmati terus jaringan tercepat dari kami!", disabled=True)
    elif action == 1:
        st.text_area("Template WhatsApp:", "Kabar gembira! Khusus untuk Anda bulan ini, kami berikan diskon tagihan spesial. Yuk klaim sekarang di aplikasi!", disabled=True)
    elif action == 2:
        st.text_area("Template WhatsApp:", "Suka streaming? Ada kejutan EKSTRA KUOTA gratis yang baru saja masuk ke nomor Anda. Cek sekarang!", disabled=True)
    else:
        st.text_area("SOP Tim Customer Service:", "Segera telepon pelanggan ini dalam 1x24 jam. Tanyakan keluhan jaringan mereka dan tawarkan solusi prioritas.", disabled=True)