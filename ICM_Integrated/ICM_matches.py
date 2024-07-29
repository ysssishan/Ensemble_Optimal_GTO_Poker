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
from ICM_EA_MCCFR_Agent import ICM_EA_MCCFR_Agent
from ICM_TA_MCCFR_Agent import ICM_TA_MCCFR_Agent


# %% [markdown]
# # Match Setting
# Initial env for a competition
env = LeducholdemEnv(
    config={'allow_step_back':True,
            'small_blind': 1,
            'allowed_raise_num': 2,
            'seed':42})

# %%
# Load pretrained agents
icm_ea_mccfr_agent = ICM_EA_MCCFR_Agent(env, model_path='./icm_ea_mccfr_agent')
icm_ea_mccfr_agent.load()
icm_ta_mccfr_agent = ICM_TA_MCCFR_Agent(env, model_path='./icm_ta_mccfr_agent')
icm_ta_mccfr_agent.load()

# %%
# set agents
env.set_agents([
    icm_ea_mccfr_agent,
    icm_ta_mccfr_agent,
    ])
print(">> Pre-trained model")


# %% [markdown]
# # Matches

# %%
# Setting
# Total number of matches simulation
total_iterations = 100

# Competition hands in one match
hands_per_iteration = 100

# Initialize lists to store results of each full iteration
all_EA_wealth_list = []
all_TA_wealth_list = []
all_EA_wins = []
all_TA_wins = []
all_EA_payoffs = []
all_TA_payoffs = []

# %%
# Run matches
# Outer loop with progress bar
# Outer loop with progress bar
for iteration in tqdm(range(total_iterations), desc="Total Iterations"):
    # Initialize wealth and win counters for each iteration
    EA_wealth = 1000
    TA_wealth = 1000
    EA_wins = 0
    TA_wins = 0
    EA_wealth_list = []
    TA_wealth_list = []
    EA_payoffs = []
    TA_payoffs = []

    # Initialize small blind
    small_blind = 1
    env.game.small_blind = small_blind

    # Inner loop with progress bar
    for hand in tqdm(range(hands_per_iteration), desc=f"Iteration {iteration + 1}", leave=False):
        # Simulate a match hand
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
        small_blind *= 1.1
        env.game.small_blind = small_blind

        # Check if match should end
        if (EA_wealth < small_blind * 2 or TA_wealth < small_blind * 2) or (EA_wealth < 0 or TA_wealth < 0):
            break

    # Store the results of this iteration
    all_EA_wealth_list.append(EA_wealth_list)
    all_TA_wealth_list.append(TA_wealth_list)
    all_EA_wins.append(EA_wins)
    all_TA_wins.append(TA_wins)
    all_EA_payoffs.append(EA_payoffs)
    all_TA_payoffs.append(TA_payoffs)

# %% [markdown]
# # Results

plt.figure()
for i, sublist in enumerate(all_EA_wealth_list):
    plt.plot(sublist, color='blue', marker='o', linestyle='-',alpha=0.5)

plt.title('Cumulative Wealth of Ensemble-average strategy')
plt.xlabel('iteration')
plt.ylabel('cumulated wealth')
plt.xlim(0, 100)
plt.ylim(0, 2000)

# 显示图形
plt.show()

# %% [markdown]
# # Results

plt.figure()
for i, sublist in enumerate(all_TA_wealth_list):
    plt.plot(sublist, color='red', marker='o', linestyle='-',alpha=0.5)

plt.title('Cumulative Wealth of Time-average strategy')
plt.xlabel('iteration')
plt.ylabel('cumulated wealth')
plt.xlim(0, 100)
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

