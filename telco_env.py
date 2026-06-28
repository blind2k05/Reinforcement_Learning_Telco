import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np

class TelcoChurnEnv(gym.Env):
    """
    Simulasi Retensi Pelanggan Telco (Versi Enterprise v2.0).
    Dilengkapi dengan Value-Based Reward, Action Masking, dan Crisis Mode.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, csv_file, crisis_mode=False):
        super(TelcoChurnEnv, self).__init__()

        self.df = pd.read_csv(csv_file)
        self.features = self.df.drop(columns=['Churn']).values
        self.labels = self.df['Churn'].values
        
        self.total_customers = len(self.df)
        self.current_customers_idx = 0
        
        # Hook untuk simulasi stress-test (krisis pasar/inflasi)
        self.crisis_mode = crisis_mode

        # Ruang Aksi (0: Diam, 1: Diskon, 2: Kuota, 3: Telepon CS)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.features.shape[1],), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_customers_idx = np.random.randint(0, self.total_customers)
        obs = self.features[self.current_customers_idx].astype(np.float32)
        return obs, {}
    
    def step(self, action):
        actual_churn = self.labels[self.current_customers_idx]
        
        # Mengambil fitur spesifik pelanggan untuk logika dinamis (Asumsi sudah di-scale 0-1)
        current_monthly_charges = self.df.iloc[self.current_customers_idx]['MonthlyCharges']
        current_tenure = self.df.iloc[self.current_customers_idx]['tenure']

        # --- VALUE-BASED REWARD ---
        # Nilai pelanggan tidak lagi +100 flat. Dihitung proporsional dari tagihannya.
        customer_value = 50.0 + (current_monthly_charges * 150.0)

        # --- EFEK CRISIS MODE ---
        # Jika krisis: Harga promo mahal x2, dan peluang sukses penahanan turun 20%
        cost_multiplier = 2.0 if self.crisis_mode else 1.0
        success_penalty = 0.20 if self.crisis_mode else 0.0 

        retained = False
        action_cost = 0
        rule_penalty = 0

        # --- ACTION MASKING ---
        # Aturan: Pelanggan baru (tenure < 0.1) dilarang ditelepon CS karena resource terbatas.
        if action == 3 and current_tenure < 0.1:
            rule_penalty = 300  # Denda berat agar AI tidak mengulangi pelanggaran ini
            action_cost = 15 * cost_multiplier
            retained = False if actual_churn == 1 else True
        else:
            # --- SKENARIO NORMAL & KRISIS ---
            if actual_churn == 0:
                retained = True 
                if action == 1: action_cost = 5 * cost_multiplier
                if action == 2: action_cost = 3 * cost_multiplier
                if action == 3: action_cost = 15 * cost_multiplier
            else:
                if action == 0:
                    action_cost = 0
                    retained = False
                elif action == 1:
                    action_cost = 5 * cost_multiplier
                    retained = np.random.rand() < max(0.0, 0.60 - success_penalty)
                elif action == 2:
                    action_cost = 3 * cost_multiplier
                    retained = np.random.rand() < max(0.0, 0.40 - success_penalty)
                elif action == 3:
                    action_cost = 15 * cost_multiplier
                    retained = np.random.rand() < max(0.0, 0.80 - success_penalty)

        # --- PERHITUNGAN REWARD AKHIR ---
        if retained:
            reward = customer_value - action_cost - rule_penalty
        else:
            reward = -customer_value - action_cost - rule_penalty

        terminated = True
        truncated = False

        info = {
            "retained": retained,
            "cost": action_cost,
            "value": customer_value,
            "actual_baseline": actual_churn
        }

        next_obs = np.zeros(self.features.shape[1], dtype=np.float32)

        return next_obs, float(reward), terminated, truncated, info
    
    def render(self):
        pass

    def close(self):
        pass