# %%
import numpy as np


# %%
class PassiveAgent(object):
    ''' A passive agent for Leduc Hold'em. This agent favors low-risk, low-reward actions. '''

    def __init__(self, num_actions):
        ''' Initialize the passive agent

        Args:
            num_actions (int): The size of the output action space
        '''
        self.use_raw = False
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        ''' Predict the action given the current state in generating training data.

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action predicted by the passive agent
        '''
        legal_actions = list(state['legal_actions'].keys())
        
        # Define passive actions (e.g., call/check as 0 and 3)
        passive_actions = [0, 3]

        # Choose a passive action if available
        possible_actions = [action for action in legal_actions if action in passive_actions]
        
        if possible_actions:
            action = np.random.choice(possible_actions)
        else:
            action = np.random.choice(legal_actions)
        
        return action

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            This function also provides action probabilities for analysis.

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action predicted by the passive agent
            probs (list): The list of action probabilities
        '''
        legal_actions = list(state['legal_actions'].keys())
        probs = [0 for _ in range(self.num_actions)]
        
        # Define passive actions (e.g., call/check as 0 and 3)
        passive_actions = [0, 3]

        if any(action in legal_actions for action in passive_actions):
            # Allocate high probability to passive actions
            total_prob = 0.7
            for action in passive_actions:
                if action in legal_actions:
                    probs[action] = total_prob / len(passive_actions)
            remaining_prob = 1 - total_prob
            for action in legal_actions:
                if action not in passive_actions:
                    probs[action] = remaining_prob / (len(legal_actions) - len(passive_actions))
        else:
            for action in legal_actions:
                probs[action] = 1 / len(legal_actions)

        info = {
            'probs': {state['raw_legal_actions'][i]: probs[action] for i, action in enumerate(legal_actions)}
        }

        return self.step(state), info