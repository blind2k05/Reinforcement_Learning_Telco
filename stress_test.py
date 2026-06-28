import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from telco_env import TelcoChurnEnv

def run_stress_test():
    print("[1/2] Memuat Simulasi Krisis Bisnis...")
    # MENGAKTIFKAN CRISIS MODE!
    env = TelcoChurnEnv('test_telco.csv', crisis_mode=True)
    model = PPO.load("ppo_telco_churn_model")

    rl_history, no_action_history = [], []
    rl_cumulative, no_action_cumulative = 0, 0

    print(f"[2/2] Menguji {env.total_customers} pelanggan di tengah badai krisis...")
    for _ in range(env.total_customers):
        obs, _ = env.reset()
        
        # Skenario 1: AI Agent menghadapi krisis
        action, _ = model.predict(obs, deterministic=True)
        _, rl_reward, _, _, _ = env.step(action)
        rl_cumulative += rl_reward
        rl_history.append(rl_cumulative)

        # Skenario 2: Perusahaan pasrah (Do Nothing)
        _, no_action_reward, _, _, _ = env.step(0)
        no_action_cumulative += no_action_reward
        no_action_history.append(no_action_cumulative)

    # Visualisasi
    plt.figure(figsize=(10, 6))
    plt.plot(rl_history, label='AI Agent (Crisis Resilience)', color='#2ca02c', linewidth=2.5)
    plt.plot(no_action_history, label='Tanpa Promo (Pasrah)', color='#d62728', linestyle=':')
    
    plt.title('Stress Test: Ketahanan Finansial Strategi Saat Krisis Market', fontsize=14, fontweight='bold')
    plt.xlabel('Jumlah Pelanggan Dievaluasi')
    plt.ylabel('Kumulatif Finansial (ARPU Points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('stress_test_result.png', dpi=300)
    print("\n[Sukses] Grafik ketahanan disimpan sebagai 'stress_test_result.png'")
    plt.show()

if __name__ == "__main__":
    run_stress_test()