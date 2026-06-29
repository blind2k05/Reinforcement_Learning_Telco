# 🚀 AI-Driven Telecom Customer Retention: A Reinforcement Learning Approach

## 📌 Business Context & Objective
This project demonstrates how **Reinforcement Learning (RL)** can be leveraged to optimize customer retention strategies within the Telecommunications industry. 

Instead of relying on guesswork or randomized promotional campaigns that drain marketing budgets, this RL agent learns autonomously to allocate the right retention interventions (e.g., Discounts, Extra Quota, or Customer Service Calls) strictly to customers who are genuinely at risk of churning. The ultimate goal is to maximize **Profitability** and **Customer Lifetime Value (CLV)** while minimizing operational waste.

## 📊 Dataset & Data Preparation
The system relies on a telecommunications customer churn dataset, processed through a rigorous pipeline to ensure the AI learns from clean, relevant, and realistic scenarios:

* **`Telco_Customer_Churn.csv`**: The raw dataset containing customer demographics, subscribed services, and churn status.
* **`clean_telco_data.csv`**: The pre-processed dataset. This step includes handling missing values, feature normalization, categorical encoding, and the **strict removal of data-leaking variables** (such as future churn reasons or scores) to ensure the model's evaluation reflects real-world constraints.
* **`train_telco.csv`**: The designated training set used to train the Reinforcement Learning agent.
* **`test_telco.csv`**: The hold-out test set used exclusively for performance evaluation and stress testing in unseen environments.

> *Note: The complete data preparation pipeline is documented in `data_prep.ipynb`.*

## 📂 Project Structure
* `telco_env.py`: The Custom Gym Environment acting as the simulator for the AI.
* `train_agent.py`: The primary script for training the RL agent from scratch.
* `evaluate.py`: The script to evaluate the trained agent and visualize its business profitability.
* `ppo_telco_churn_model.zip`: The pre-trained AI model (Trained PPO Agent).

---

## 📈 Key Results & Business Impact

To validate the system's effectiveness, several data-driven evaluations were conducted. Here is the interpretation of our experimental results:

### 1. RL vs. Supervised Learning Comparison (`rl_vs_sl_comparison.png`)
* **What is it?**: A comparative analysis between a static predictive model (Random Forest) and our prescriptive optimization model (Reinforcement Learning).
* **Interpretation**: While Random Forest delivers high predictive accuracy, the RL model yields **substantially higher profitability**. This proves that our dynamic retention strategy is highly capital-efficient, completely avoiding budget waste on false positives.

### 2. Crisis Stress Testing (`stress_test_result.png`)
* **What is it?**: A simulation evaluating the model's resilience when operational costs double (simulating inflation or economic downturns).
* **Interpretation**: The graph highlights the agent's adaptability. Under crisis conditions, the AI autonomously shifts to a more selective intervention strategy (prioritizing high-value VIP customers), keeping the company's profit margins positive despite hostile market conditions.

### 3. Business Impact Evaluation (`business_impact_evaluation.png`)
* **What is it?**: Projected operational **Cost Savings** and increases in **Customer Lifetime Value (CLV)**.
* **Interpretation**: This translates technical metrics into management terms. It visualizes the exact revenue saved by deploying the AI Co-Pilot compared to random or manual intervention methods, highlighting massive efficiency gains for the Customer Service department.

---

## 🚀 Quick Run Guide 

These instructions are designed for seamless execution directly within Visual Studio Code (VS Code) or your preferred IDE.

### Step 1: Workspace Setup
1. Clone or download this repository.
2. Open **VS Code**.
3. Navigate to **File > Open Folder...** and select the downloaded project directory.

### Step 2: Virtual Environment Setup
It is highly recommended to use a virtual environment to prevent dependency conflicts.
1. Open the integrated terminal in VS Code (`Ctrl` + `` ` ``).
2. Create a virtual environment named `venv`:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

* **Windows Users:**

```Bash
.\venv\Scripts\activate
```

* **Mac/Linux Users:**

```Bash
source venv/bin/activate
```

*(A successful activation will display (venv) at the beginning of your terminal prompt).*

### Step 3: Install Dependencies
With the virtual environment active, install all required packages:

```Bash
pip install -r requirements.txt
```
(Wait about 1-2 minute untill process is done).

### Step 4: Run Business Evaluation
To observe how the AI optimizes the company's retention budget compared to randomized baseline campaigns, execute:

```Bash
python evaluate.py
```
> **Note:** This script automatically loads test_telco.csv and evaluates the agent's decisions across 1,400+ unseen customers.

### Step 5: (Optional) Retrain the AI Agent
To observe the agent learning from scratch via trial and error, initiate the training sequence:

```Bash
python train_agent.py
```
This executes 100,000 training timesteps and updates the `ppo_telco_churn_model.zip` file with the newly optimized weights.

### Step 6: (Optional) Launch AI Dashboard
For an interactive visual representation of the agent's decision-making process:

```Bash
streamlit run app.py
```

### Step 7: (Optional) Advanced Training Monitoring
**To monitor the TensorBoard:**
```bash
python -m tensorboard.main --logdir ./ppo_telco_tensorboard/
```
### Step 8: (Optional) Execute Stress Test
Curious how the model reacts to sudden market shifts or doubled operational costs? Run the stress test to ensure the agent maintains financial viability under crisis:

```bash
python stress_test.py
```


