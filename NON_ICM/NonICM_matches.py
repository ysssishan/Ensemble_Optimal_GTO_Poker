# %%
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from Basic_Leduc_Game import LeducholdemGame
from Game_Env import LeducholdemEnv
from NonICM_EA_MCCFR_Agent import NonICM_EA_MCCFR_Agent
from NonICM_TA_MCCFR_Agent import NonICM_TA_MCCFR_Agent

# %%
# Initial env for a competition
env = LeducholdemEnv(
    config={'allow_step_back':True,
            'small_blind': 1,
            'allowed_raise_num': 2,
            'seed':42})

# %%
# Load pretrained agents
nonicm_ea_mccfr_agent = NonICM_EA_MCCFR_Agent(env, model_path='./nonicm_ea_mccfr_agent')
nonicm_ea_mccfr_agent.load()
nonicm_ta_mccfr_agent = NonICM_TA_MCCFR_Agent(env, model_path='./nonicm_ta_mccfr_agent')
nonicm_ta_mccfr_agent.load()

# %%
# set agents
env.set_agents([
    nonicm_ea_mccfr_agent,
    nonicm_ta_mccfr_agent,
    ])

print(">> Pre-trained model")


# %%
# Competition
iteration = 10000

# Initialize lists to store cumulative wealth values for each iteration
EA_cumulative_wealth_list = []
TA_cumulative_wealth_list = []

EA_wealth = []
TA_wealth = []
EA_wins = 0
EA_cumulative_wealth = 0
TA_cumulative_wealth = 0

for i in range(iteration):
    while (True):
        # print(">> Start a new competition")

        trajectories, payoffs = env.run(is_training=False)
        EA_wealth.append(payoffs[0])
        TA_wealth.append(payoffs[1])

        if payoffs[0] > 0:
            EA_wins += 1
        
        EA_cumulative_wealth += payoffs[0]
        TA_cumulative_wealth += payoffs[1]

        # Append cumulative wealth to lists
        EA_cumulative_wealth_list.append(EA_cumulative_wealth)
        TA_cumulative_wealth_list.append(TA_cumulative_wealth)

        break

TA_wins = iteration - EA_wins
# %%
print(EA_wins)

# %%
print(TA_wins)

# %%
print(EA_cumulative_wealth)

# %%
print(TA_cumulative_wealth)

# %%
# Plot cumulative wealth
plt.figure(figsize=(12, 6))
plt.plot(EA_wealth, label='EA Wealth', color='blue')
plt.plot(TA_wealth, label='TA Wealth', color='red')
plt.xlabel('Iteration')
plt.ylabel('Payoffs')
plt.title('Payoffs of EA and TA Over Iterations')
plt.legend()
plt.grid(True)
plt.show()


# %%
# Plot cumulative wealth
plt.figure(figsize=(12, 6))
plt.plot(EA_cumulative_wealth_list, label='EA Cumulative Wealth', color='blue')
plt.plot(TA_cumulative_wealth_list, label='TA Cumulative Wealth', color='red')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Wealth')
plt.title('Cumulative Wealth of EA and TA Over Iterations')
plt.legend()
plt.grid(True)
plt.show()