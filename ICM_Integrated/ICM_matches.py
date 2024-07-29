# %%
import numpy as np
import pandas as pd
import pickle

from Basic_Leduc_Game import LeducholdemGame
from Game_Env import LeducholdemEnv
from ICM_EA_MCCFR_Agent import ICM_EA_MCCFR_Agent
from ICM_TA_MCCFR_Agent import ICM_TA_MCCFR_Agent

# %%
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


# %%
# Competition
iteration = 20000

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
