# %%
import numpy as np

# %%
class AggressiveAgent(object):
    ''' An aggressive agent for Leduc Hold'em. This agent favors high-risk, high-reward actions. '''

    def __init__(self, num_actions):
        ''' Initialize the aggressive agent

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
            action (int): The action predicted by the aggressive agent
        '''
        legal_actions = list(state['legal_actions'].keys())
        
        # Define aggressive action (e.g., raise)
        agg_action = 1  # Assuming action 1 is 'raise'

        if agg_action in legal_actions:
            action = agg_action
        else:
            action = np.random.choice(legal_actions)
        
        return action

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            This function also provides action probabilities for analysis.

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action predicted by the aggressive agent
            probs (list): The list of action probabilities
        '''
        legal_actions = list(state['legal_actions'].keys())
        probs = [0 for _ in range(self.num_actions)]
        
        # Define aggressive action (e.g., raise)
        agg_action = 1  # Assuming action 1 is 'raise'

        info = {}
        if agg_action in legal_actions:
            probs[agg_action] = 0.7
            remaining_prob = 1 - probs[agg_action]
            for action in legal_actions:
                if action != agg_action:
                    probs[action] = remaining_prob / (len(legal_actions) - 1)
        else:
            for action in legal_actions:
                probs[action] = 1 / len(legal_actions)

        info = {
            'probs': {state['raw_legal_actions'][i]: probs[action] for i, action in enumerate(legal_actions)}
        }

        return self.step(state), info

