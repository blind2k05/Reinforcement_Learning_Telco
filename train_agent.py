import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from telco_env import TelcoChurnEnv

def main():
    print("[1/4] menginisialisasi environment latihan")

    raw_env = TelcoChurnEnv(csv_file='train_telco.csv')
    raw_env = Monitor(raw_env)
    env = DummyVecEnv([lambda:raw_env])

    print("[2/4] membangun otak agen RL (dengan MLOps TensorBoard)")
    
    # --- PENAMBAHAN MLOPS ---
    # Parameter tensorboard_log ditambahkan untuk merekam grafik performa latihan AI
    model = PPO(
        policy="MlpPolicy",
        env = env,
        verbose=1,
        learning_rate=0.0003,
        gamma=0.99,
        tensorboard_log="./ppo_telco_tensorboard/"
    )
    
    print("[3/4] memulai proses latihan")
    # Karena aturan environment baru lebih rumit (ada denda 300 poin dll),
    # disarankan menaikkan timesteps agar AI punya waktu untuk paham aturannya.
    model.learn(total_timesteps=150000)

    print("[4/4] pelatihan selesai menyimpan model")
    model.save("ppo_telco_churn_model")
    print("Sukses menyimpan otak ai dengan nama 'ppo_telco_churn_model'")
    
    print("\n[MLOps] Cek grafik metrik pelatihan dengan mengetik ini di terminal:")
    print("tensorboard --logdir ./ppo_telco_tensorboard/")

if __name__ == "__main__":
    main()