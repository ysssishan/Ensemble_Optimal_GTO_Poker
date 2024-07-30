# %% [markdown]
# # Import
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

# %%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# %%
from Basic_Leduc_Game import LeducholdemGame
from Game_Env import LeducholdemEnv
from Random_Agent import RandomAgent
from ICM_EA_MCCFR_Agent import ICM_EA_MCCFR_Agent


# %% [markdown]
# # Match Setting
# Initial env for a competition
env = LeducholdemEnv(
    config={'allow_step_back':True,
            'small_blind': 10,
            'allowed_raise_num': 2,
            'seed':42})

# %%
# load agents

icm_ea_mccfr_agent = ICM_EA_MCCFR_Agent(env, model_path='./icm_ea_mccfr_agent')
icm_ea_mccfr_agent.load()
random_agent = RandomAgent(num_actions=4)

# %%
# set agents
env.set_agents([
    random_agent,
    icm_ea_mccfr_agent,
    ])
print(">> Load agent")

# %% [markdown]
# # Matches

# %%
# Setting
# Total number of matches simulation
total_iterations = 1

# Initialisze lists to store results of each full iteration
all_EA_wealth_list = []
all_TA_wealth_list = []
all_EA_wins = []
all_TA_wins = []
all_EA_payoffs = []
all_TA_payoffs = []

# %%
# Run matches


# Define the total iterations
total_iterations = 1

for i in range(total_iterations):
    # Initialize wealth and win counters for each iteration
    EA_wealth = 1000
    TA_wealth = 1000
    EA_wins = 0
    TA_wins = 0
    EA_wealth_list = []
    TA_wealth_list = []
    small_blind = 10
    EA_payoffs = []
    TA_payoffs =[]
    
    while len(EA_wealth_list) < 10:
        env.game.small_blind = small_blind
        env.game.big_blind = 2 * small_blind
        env.game.raise_amount = env.game.big_blind

        trajectories, payoffs = env.run(is_training=False)
        EA_payoffs.append(payoffs[0])
        TA_payoffs.append(payoffs[1])

        if payoffs[0] > 0:
            EA_wins += 1
        else:
            TA_wins += 1

        EA_wealth += payoffs[0]
        TA_wealth += payoffs[1]

        # Append cumulative wealth to lists
        EA_wealth_list.append(EA_wealth)
        TA_wealth_list.append(TA_wealth)

        # Double the small blind
        small_blind *= 2

        # Check if either player needs to be reset
        if EA_wealth < 0 or TA_wealth < 0:
            EA_wealth = 1000
            TA_wealth = 1000
            small_blind = 10

print(sum(EA_payoffs))
print(sum(TA_payoffs))

print(EA_wins)
print(TA_wins)
# %% [markdown]
# # Results

plt.figure()
for i, sublist in enumerate(all_EA_wealth_list):
    plt.plot(sublist, color='blue', marker='o', linestyle='-',alpha=0.1)
for i, sublist in enumerate(all_TA_wealth_list):
    plt.plot(sublist, color='red', marker='o', linestyle='-',alpha=0.1)

plt.title('Cumulative Wealth')
plt.xlabel('iteration')
plt.ylabel('cumulated wealth')
plt.xlim(0, 20)
plt.ylim(0, 2000)

# 显示图形
plt.show()



# %%
# print(f"Ensemble-average strategy wins {EA_wins} times.")

# %%
# print(f"Time-average strategy wins {total_iterations * hands_per_iteration - EA_wins} times.")

# %%
# if EA_wins < (total_iterations * hands_per_iteration - EA_wins):
#     print(f"⭕️ Time-average strategy is more optimal than Ensemble-average.")
# else:
#     print(f"❌ Time-average strategy is not more optimal than Ensemble-average.")

# %%
with open('./icm_ea_mccfr_agent/average_policy.pkl', 'rb') as f:
    average_policy_data = pickle.load(f)
average_policy_df = pd.DataFrame(list(average_policy_data.items()), columns=['Key', 'Average policy [Call, Raise, Fold, Check]']) 
average_policy_df.head(10)
# %%
average_policy_df[average_policy_df['Key'] == {"obs": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 20.0], "action_record": []}]
# %%
