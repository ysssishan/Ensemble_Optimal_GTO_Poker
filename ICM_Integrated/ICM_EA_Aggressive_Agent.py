# %%
import numpy as np
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ICM_EA_MCCFR_Agent import ICM_EA_MCCFR_Agent


class ICM_EA_AggressiveAgent(ICM_EA_MCCFR_Agent):

    def action_probs(self, obs, legal_actions, policy):
        obs_dict = json.loads(obs)
        obs_array = np.array(obs_dict['obs'])

        # 手牌部分是obs_array的前3个元素
        hand_card = obs_array[:3]
        # K: [0, 0, 1], Q: [0, 1, 0], J: [1, 0, 0]
        if np.array_equal(hand_card, [0, 0, 1]):  # K
            if 2 in legal_actions:  # Raise action is 2
                action_probs = np.zeros(self.env.num_actions)
                action_probs[2] = 1.0
                return action_probs
        elif np.array_equal(hand_card, [0, 1, 0]):  # Q
            action_probs = np.zeros(self.env.num_actions)
            for action in legal_actions:
                if action != 2:  # Ensure Raise action is not chosen
                    action_probs[action] = 1.0 / (len(legal_actions) - 1)
            return action_probs
        elif np.array_equal(hand_card, [1, 0, 0]):  # J
            if 0 in legal_actions:  # Fold action is 0
                action_probs = np.zeros(self.env.num_actions)
                action_probs[0] = 1.0
                return action_probs

        return super().action_probs(obs, legal_actions, policy)
# %%
