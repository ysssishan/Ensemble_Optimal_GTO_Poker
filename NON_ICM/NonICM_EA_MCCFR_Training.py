# %%
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# %%
from Basic_Leduc_Game import LeducholdemGame
from Game_Env import LeducholdemEnv
from NonICM_EA_MCCFR_Agent import NonICM_EA_MCCFR_Agent

# %%
def train(agent, num_iterations):
    with tqdm(total=num_iterations, desc="Training Progress") as pbar:
        for i in range(num_iterations):
            agent.train()
            agent.reset_chipstack()
            agent.env.game.small_blind = 1
            agent.env.game.big_blind = 2 * agent.env.game.small_blind
            agent.env.game.raise_amount = agent.env.game.big_blind
            
            pbar.set_postfix({"Iteration": i + 1}, refresh=True)
            pbar.update(1)


# %%
# Creat LeducholdemEnvSimplified Env

env = LeducholdemEnv(
    config={'allow_step_back':True,
            'small_blind': 1,
            'allowed_raise_num': 2,
            'seed':42})

# Creat CFR Agent 
nonicm_ea_mccfr_agent = NonICM_EA_MCCFR_Agent(env, init_chipstack_pair=np.array([1000.0, 1000.0]),small_blind_multiplier=1.1)

# Train CFR Agent
num_iterations = 1000
train(nonicm_ea_mccfr_agent, num_iterations)

# Save
nonicm_ea_mccfr_agent.save()


# %% [markdown]
# ## Results

# %%
with open('./nonicm_ea_mccfr_agent/policy.pkl', 'rb') as f:
    policy_data = pickle.load(f)
policy_df = pd.DataFrame(list(policy_data.items()), columns=['Obs', 'Probability [Call, Raise, Fold, Check]'])

policy_df


# %%
with open('./nonicm_ea_mccfr_agent/regrets.pkl', 'rb') as f:
    regrets_data = pickle.load(f)
regrets_df = pd.DataFrame(list(regrets_data.items()), columns=['Obs', 'Regret [Call, Raise, Fold, Check]'])
regrets_df['positive_regret_sum'] = regrets_df['Regret [Call, Raise, Fold, Check]'].apply(lambda x: sum(v for v in x if v > 0))
regrets_df.head(10)


# %%
with open('./nonicm_ea_mccfr_agent/average_policy.pkl', 'rb') as f:
    average_policy_data = pickle.load(f)
average_policy_df = pd.DataFrame(list(average_policy_data.items()), columns=['Key', 'Average policy [Call, Raise, Fold, Check]'])
average_policy_df.head(10)



