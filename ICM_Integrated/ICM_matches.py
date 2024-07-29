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
total_iterations = 30

# Competition hands in one match
hands_per_iteration = 5000

# Initialize lists to store results of each full iteration
all_EA_wealth = {}
all_TA_wealth = {}
all_EA_wins = {}
all_TA_wins = {}
all_EA_payoffs = {}
all_TA_payoffs = {}

# %%
# Run matches
# Outer loop with progress bar
# Outer loop with progress bar
for round in tqdm(range(total_iterations), desc="Total Iterations"):
    
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
    small_blind = 10
    env.game.small_blind = small_blind
    env.game.big_blind = 2 * env.game.small_blind
    env.game.raised_amount = env.game.big_blind

    # Inner loop with progress bar
    for hand in tqdm(range(hands_per_iteration), desc=f"Iteration {round + 1}", leave=False):
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
        small_blind = 0.1 * env.get_payoffs()[env.game.game_pointer]
        env.game.small_blind = small_blind
        env.game.big_blind = 2 * env.game.small_blind
        env.game.raised_amount = env.game.big_blind

        # Check if match should end
        if (EA_wealth < small_blind * 2 or TA_wealth < small_blind * 2) or (EA_wealth < 0 or TA_wealth < 0):
            break

    # Store the results of this iteration
    all_EA_wealth[round] = EA_wealth_list
    all_TA_wealth[round] = TA_wealth_list
    all_EA_wins[round] = EA_wins
    all_TA_wins[round] = TA_wins
    all_EA_payoffs[round] = EA_payoffs
    all_TA_payoffs[round]= TA_payoffs

# %% [markdown]
# # Results

# 创建图形
plt.figure(figsize=(15, 10))

# 迭代每个round并绘制每个玩家的累积筹码值以及差值
for round_num in all_EA_wealth:
    # 获取当前round的累积筹码值
    wealth_EA = all_EA_wealth[round_num]
    wealth_TA = all_TA_wealth[round_num]
    
    # 计算差值
    wealth_diff = [a - b for a, b in zip(wealth_EA, wealth_TA)]
    
    # 绘制累积筹码值
    plt.plot(wealth_EA, label=f'Ensemble average strategy - Round {round_num}' if round_num == 1 else "", color='blue', alpha=0.3)
    plt.plot(wealth_TA, label=f'Time average strategy - Round {round_num}' if round_num == 1 else "", color='red', alpha=0.3)
    
    # 绘制差值
    plt.plot(wealth_diff, label=f'Difference - Round {round_num}' if round_num == 1 else "", color='green', alpha=0.3)

# 添加图例和标签
plt.xlabel('Hands')
plt.ylabel('Cumulative Chips / Difference')
plt.title('Cumulative Chips and Differences over 5000 Hands in 20 Rounds for Two Players')
plt.legend(loc='upper left')

# 显示图形
plt.show()

# %%

fig, ax1 = plt.subplots(figsize=(15, 10))

# 创建第二个y轴
ax2 = ax1.twinx()

for round_num in all_EA_wealth:
    # 获取当前round的累积筹码值
    wealth_EA = all_EA_wealth[round_num]
    wealth_TA = all_TA_wealth[round_num]
    wealth_diff = [a - b for a, b in zip(wealth_EA, wealth_TA)]
    
    ax1.plot(wealth_EA, label=f'Player A - Round {round_num}' if round_num == 1 else "", color='blue', alpha=0.3)
    ax1.plot(wealth_TA, label=f'Player B - Round {round_num}' if round_num == 1 else "", color='red', alpha=0.3)
    # ax2.plot(wealth_diff, label=f'Difference - Round {round_num}' if round_num == 1 else "", color='green', alpha=0.3)

ax1.set_xlabel('Hands')
ax1.set_ylabel('Cumulative Chips')
ax2.set_ylabel('Difference')
ax1.set_title('Cumulative Chips and Differences over 5000 Hands in 20 Rounds for Two Players')

fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.show()
# %%
