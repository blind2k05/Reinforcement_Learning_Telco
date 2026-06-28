import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from telco_env import TelcoChurnEnv

def evaluate_model():
    print("[1/3] Memuat data pengujian dan otak AI...")
    # Memuat environment dengan data uji (pelanggan yang belum pernah dilihat AI saat latihan)
    test_env = TelcoChurnEnv('test_telco.csv')
    # Memuat model AI yang sudah pintar dari hasil training sebelumnya
    model = PPO.load("ppo_telco_churn_model")

    # Jumlah pelanggan uji yang akan dievaluasi
    num_episodes = test_env.total_customers

    # Variabel penampung total keuntungan/kerugian untuk masing-masing skenario
    rl_cumulative = 0
    random_cumulative = 0
    no_action_cumulative = 0

    # List untuk menyimpan riwayat poin demi poin agar bisa digambar menjadi grafik garis
    rl_history = []
    random_history = []
    no_action_history = []

    print(f"[2/3] Mengevaluasi {num_episodes} pelanggan secara simulasi...")

    # Menguji satu per satu pelanggan di dalam dataset test
    for _ in range(num_episodes):
        # Reset environment untuk mengambil profil 1 pelanggan acak
        obs, _ = test_env.reset()

        # --- SKENARIO 1: AI Agent (Smart Promo) ---
        # deterministic=True memastikan AI menggunakan strategi terbaiknya, bukan menebak acak (eksplorasi)
        action, _states = model.predict(obs, deterministic=True)
        _, rl_reward, _, _, _ = test_env.step(action)
        rl_cumulative += rl_reward
        rl_history.append(rl_cumulative)

        # --- SKENARIO 2: Promo Acak (Blind Promo) ---
        # Mengambil tindakan acak (0, 1, 2, atau 3) seperti tebak-tebakan
        random_action = test_env.action_space.sample()
        _, random_reward, _, _, _ = test_env.step(random_action)
        random_cumulative += random_reward
        random_history.append(random_cumulative)

        # --- SKENARIO 3: Tanpa Promo (Do Nothing) ---
        # Aksi 0 berarti perusahaan tidak melakukan intervensi apa pun kepada pelanggan
        _, no_action_reward, _, _, _ = test_env.step(0)
        no_action_cumulative += no_action_reward
        no_action_history.append(no_action_cumulative)

    # Menampilkan ringkasan hasil akhir secara angka di terminal
    print("\n--- HASIL EVALUASI FINANSIAL ---")
    print(f"Total Nilai Skenario 'Tanpa Promo' : {no_action_cumulative}")
    print(f"Total Nilai Skenario 'Promo Acak'  : {random_cumulative}")
    print(f"Total Nilai Skenario 'AI Agent'    : {rl_cumulative}")

    print("\n[3/3] Membuat Grafik Visualisasi Bisnis...")
    
    # --- VISUALISASI MATPLOTLIB ---
    # Menggambar ketiga skenario ke dalam satu grafik untuk membandingkan kinerjanya secara visual
    plt.figure(figsize=(10, 6))
    plt.plot(rl_history, label='AI Agent (Smart Promo)', color='#2ca02c', linewidth=2.5)
    plt.plot(random_history, label='Promo Acak (Blind Promo)', color='#ff7f0e', linestyle='--')
    plt.plot(no_action_history, label='Tanpa Promo (Do Nothing)', color='#d62728', linestyle=':')
    
    # Pengaturan estetika grafik (Judul, Label Sumbu, dan Legenda)
    plt.title('Dampak Bisnis: Kumulatif Profit/Loss Strategi Retensi', fontsize=15, fontweight='bold')
    plt.xlabel('Jumlah Pelanggan Dievaluasi', fontsize=12)
    plt.ylabel('Kumulatif Finansial (Poin Reward)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Menyimpan hasil grafik ke file gambar resolusi tinggi untuk bahan presentasi/GitHub
    plt.savefig('business_impact_evaluation.png', dpi=300)
    print("[Sukses] Grafik berhasil disimpan sebagai 'business_impact_evaluation.png'")
    
    # Menampilkan jendela grafik secara langsung di layar
    plt.show()

# Blok ini memastikan script hanya berjalan jika dieksekusi langsung, 
# bukan saat di-import oleh file Python lain.
if __name__ == "__main__":
    evaluate_model()