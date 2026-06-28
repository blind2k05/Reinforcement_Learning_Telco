import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from stable_baselines3 import PPO
from telco_env import TelcoChurnEnv

def run_comparison():
    print("[1/4] Memuat data dan melatih model Supervised Learning (Random Forest)...")
    
    # 1. Melatih Baseline Supervised Learning (Prediktif Tradisional)
    # Kita menggunakan data training yang sama persis dengan yang digunakan RL
    train_df = pd.read_csv('train_telco.csv') 
    X_train = train_df.drop(columns=['Churn']).values
    y_train = train_df['Churn'].values

    # Melatih model Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("      Model Random Forest berhasil dilatih!")

    print("[2/4] Mempersiapkan Environment dan RL Agent...")
    test_env = TelcoChurnEnv('test_telco.csv')
    rl_model = PPO.load("ppo_telco_churn_model")

    num_episodes = test_env.total_customers

    # Variabel penampung hasil finansial
    rl_cumulative = 0
    sl_cumulative = 0
    no_action_cumulative = 0

    rl_history = []
    sl_history = []
    no_action_history = []

    print(f"[3/4] Mengevaluasi {num_episodes} pelanggan (RL vs Random Forest vs Do Nothing)...")

    for _ in range(num_episodes):
        obs, _ = test_env.reset()
        
        # --- SKENARIO 1: RL Agent (Prescriptive / Smart Promo) ---
        # AI menimbang biaya dan risiko untuk memilih aksi terbaik secara mandiri
        rl_action, _states = rl_model.predict(obs, deterministic=True)
        _, rl_reward, _, _, _ = test_env.step(rl_action)
        rl_cumulative += rl_reward
        rl_history.append(rl_cumulative)

        # --- SKENARIO 2: Supervised Learning (Predictive / Standard Promo) ---
        # Model menebak: Apakah pelanggan ini akan Churn (1) atau Loyal (0)?
        sl_pred = rf_model.predict([obs])[0]
        
        # SOP Konvensional Perusahaan:
        # Jika ditebak Loyal -> Jangan lakukan apa-apa (Aksi 0)
        # Jika ditebak Churn -> Pukul rata beri Promo Diskon Tagihan (Aksi 1, Biaya 5)
        sl_action = 1 if sl_pred == 1 else 0
        _, sl_reward, _, _, _ = test_env.step(sl_action)
        sl_cumulative += sl_reward
        sl_history.append(sl_cumulative)

        # --- SKENARIO 3: Tanpa Promo (Do Nothing Baseline) ---
        _, no_action_reward, _, _, _ = test_env.step(0)
        no_action_cumulative += no_action_reward
        no_action_history.append(no_action_cumulative)

    print("\n--- HASIL EVALUASI FINANSIAL AKHIR ---")
    print(f"Total Nilai 'Tanpa Promo'         : {no_action_cumulative}")
    print(f"Total Nilai 'Supervised Learning' : {sl_cumulative}")
    print(f"Total Nilai 'RL Agent (AI)'       : {rl_cumulative}")

    print("\n[4/4] Membuat Grafik Komparasi Bisnis...")
    
    # --- VISUALISASI ---
    plt.figure(figsize=(10, 6))
    plt.plot(rl_history, label='RL Agent (Prescriptive Promo)', color='#2ca02c', linewidth=2.5)
    plt.plot(sl_history, label='Random Forest (Predictive SOP)', color='#1f77b4', linestyle='--', linewidth=2)
    plt.plot(no_action_history, label='Tanpa Promo (Do Nothing)', color='#d62728', linestyle=':')

    plt.title('Komparasi Profit: Reinforcement Learning vs Supervised Learning', fontsize=15, fontweight='bold')
    plt.xlabel('Jumlah Pelanggan Dievaluasi', fontsize=12)
    plt.ylabel('Kumulatif Finansial (Poin Reward)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('rl_vs_sl_comparison.png', dpi=300)
    print("[Sukses] Grafik komparasi disimpan sebagai 'rl_vs_sl_comparison.png'")
    plt.show()

if __name__ == "__main__":
    run_comparison()