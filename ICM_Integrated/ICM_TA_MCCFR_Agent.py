# %%
import numpy as np
import json
import collections
from math import log
import os
import pickle
from tqdm import tqdm

# %%
class ICM_TA_MCCFR_Agent():
    
    def __init__(self, env, model_path='./icm_ta_mccfr_agent',
                 max_hands = 100, 
                 init_chipstack_pair=np.array([10.0, 10.0]), small_blind_multiplier=1.05,
                 prize_structure = np.array([70000, 30000])):
        ''' 
        Args:
            env (Env): Env class
            model_path: where to store the model and results
            num_simulations: number of Monte Carlo simulations per hand of Leduc Holdem
            small_blind_multiplier: the growth rate of small blind (and big blind, betting and raising amount)
        '''  
        self.use_raw = False
        self.env = env
        self.model_path = model_path
        
        # initial params related to special match setting
        self.init_chipstack_pair = init_chipstack_pair
        self.chipstack_pair = self.init_chipstack_pair
        self.small_blind_multiplier = small_blind_multiplier
        self.prize_structure = prize_structure
        self.max_hands = max_hands
        
        # initial file to store and update strategy,average strategy, regret
        self.policy = {}
        self.regrets = {}
        self.average_policy = {}
        self.state_utilities = {}

        # initial terminal variables for creating new tree
        self.terminal_payoffs = {}
        self.terminal_nodes = {}
        
        # initial counting variables for debuging and weighting
        self.iteration = 0
        self.hands = 0

    """check if repeated games/tournaments should continue"""
    def can_continue(self):
        '''
        Repeated games/tournaments only continue when:
        - each player has more than 0 chips, and
        - each player's chipstack is greater than or equal to the big blind (the betting amount)
        '''
        # Check if all players have more than 0 chips
        all_chipstacks_pos = all(player_chipstack > 0 for player_chipstack in self.chipstack_pair)
        
        # Check if all players' chipstacks are above or equal to the big blind
        all_chipstacks_above_big_blind = all(player_chipstack >= self.env.game.big_blind for player_chipstack in self.chipstack_pair)
        
        # Check if the current game count is less than the maximum allowed games
        within_max_games_limit = self.hands < self.max_hands

        return all_chipstacks_pos and all_chipstacks_above_big_blind and within_max_games_limit

    """reset chipstacks to initial one to play more repeated games/tournaments"""
    def reset_chipstack(self):
        self.chipstack_pair = self.init_chipstack_pair

    """simulate one hand of game"""
    def train(self):
        
        self.iteration += 1

        # print(
        #     f"💪The {self.iteration}st/nd/th Repeated Leduc Holdem Simulation\n"
        #     f"start with chipstack pair {self.chipstack_pair}\n"
        #     f"and small blind {self.env.game.small_blind}."
        #     )
        
        # Continue training while the repeated game/tournament conditions allow
        while self.can_continue():
            
            try:
                self.hands += 1                
                # print(f'** Hand{self.hands} Leduc Holdem start with chipstacks {self.chipstack_pair}')
                # print(f'** Hand{self.hands} Leduc Holdem start with small blind {self.env.game.small_blind}')

                # perform Monte Carlo simulation
                for player_id in range(self.env.num_players):
                    self.env.reset()
                    probs = np.ones(self.env.num_players)
                    self.simulate_game(probs, player_id)                  
                
                # Update the policy based on the simulations       
                self.update_policy()

                # tournament and increasing blind setting
                # Specifically for multi-stage games/repeated games/tournaments
                self.chipstack_pair = self.update_chipstacks()
                self.env.game.small_blind *= self.small_blind_multiplier
                self.env.game.big_blind = 2 * self.env.game.small_blind

            except Exception as e:
                print(f"An error occurred: {e}")
                break
        # print(f'🌟The {self.iteration}st/nd/th Repeated Leduc Holdem end with chipstacks {self.chipstack_pair}🌟')
        # print(f'✅The {self.iteration}st/nd/th Monte Carlo Simulation Done✅')

    """simulate one hand of Leduc Holdem"""
    def simulate_game(self, probs, player_id):
        ''' 
        Args:
            probs: The reach probability of the current node
            player_id: The player to update the value
        Returns:
            state_utilities (list): The expected utilities for all the players
        '''

        if self.env.is_over():
            # Get end-game chipstack for all players
            pre_chipstacks = self.chipstack_pair
            pre_icm = self.calculate_ICM(pre_chipstacks,self.prize_structure)
            end_chipstacks = self.update_chipstacks()
            end_icm = self.calculate_ICM(end_chipstacks,self.prize_structure)

            # Calculate the growth rate relative to the initial chipstack
            growth_rate = end_icm / pre_icm
            log_growth_rate = np.log(growth_rate)
            
            # print(f"pre_chipstacks {pre_chipstacks}")
            # print(f"pre_icm {pre_icm}")
            # print(f"end_chipstacks {end_chipstacks}")
            # print(f"end_icm {end_icm}")
            # print(f"log growth rate {log_growth_rate}")
            return log_growth_rate

        current_player = self.env.get_player_id()

        action_utilities = {}
        state_utility = np.zeros(self.env.num_players)
        obs, legal_actions = self.get_state(current_player)
        action_probs = self.action_probs(obs, legal_actions, self.policy)

        # Initialize state utility for MCCFR
        sampled_action = self.sample_action(legal_actions, action_probs, epsilon=0.6)
        action_prob = action_probs[sampled_action]
        new_probs = probs.copy()
        new_probs[current_player] *= action_prob

        # Simulate taking the sampled action
        self.env.step(sampled_action)
        utility = self.simulate_game(new_probs, player_id)
        self.env.step_back()

        state_utility += action_prob * utility
        action_utilities[sampled_action] = utility

        if not current_player == player_id:
            return state_utility

        # Update regret and average policy
        if obs not in self.state_utilities:
            self.state_utilities[obs] = []  # Initialize state utility list for the current state
        self.state_utilities[obs].append(state_utility)

        # Calculate regret and update average policy
        player_prob = probs[current_player]
        counterfactual_prob = (np.prod(probs[:current_player]) *
                                np.prod(probs[current_player + 1:]))
        player_state_utility = state_utility[current_player]

        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.num_actions)
        if obs not in self.average_policy:
            self.average_policy[obs] = np.zeros(self.env.num_actions)
        
        # Calculate regret for the sampled action
        for action in legal_actions:
            if action in action_utilities:
                action_prob = action_probs[action]
                regret = counterfactual_prob * (action_utilities.get(action)[current_player]
                        - player_state_utility)
                self.regrets[obs][action] += regret
                # print(f'regret of action {action} is {regret}')
                self.average_policy[obs][action] += self.iteration * player_prob * action_prob

        return state_utility

    """Calculate expected payoffs based on current win probabilities and prize structure."""
    def calculate_ICM(self, chipstack_pair, prize_structure):
        '''
        Returns:
            np.ndarray: An array of calculated icm utilities.
        '''
        total_chips = np.sum(chipstack_pair)
        icm_utilities = np.zeros(len(chipstack_pair))
        
        for i, player_chipstack in enumerate(chipstack_pair):
            win_prob = player_chipstack / total_chips
            icm_utilities[i] = (prize_structure[0] * win_prob + prize_structure[1] * (1 - win_prob))
        return icm_utilities
    
    """Monte Carlo Sample an action based on the action probabilities"""
    def sample_action(self, legal_actions, action_probs, epsilon=0.6):
        '''
        Args:
            legal_actions (list): List of legal actions available in the current state.
            action_probs (numpy.array): Probabilities of actions according to the policy.
            epsilon (float): The probability of choosing a random action (epsilon-greedy parameter).
        Returns:
            action (int): The chosen action.
        '''
        if np.random.rand() < epsilon:
            # With probability epsilon, choose a random legal action
            return np.random.choice(legal_actions)
        else:
            # Otherwise, choose an action based on the action probabilities
            return np.random.choice(legal_actions, p=action_probs[legal_actions])

    """Update chipstack after one hand of Leduc Holdem"""
    def update_chipstacks(self):
        new_chipstack_pair = np.copy(self.chipstack_pair)
        new_chipstack_pair += self.env.get_payoffs()
        # print(f"payoffs {self.env.get_payoffs()}")
        return new_chipstack_pair
    
    """Update the policy using regret matching for each observation."""
    def update_policy(self):
        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)

    """Update the average policy to keep track of the convergence."""
    def update_average_policy(self):
        for obs in self.policy:
            if obs in self.average_policy:
                self.average_policy[obs] = (
                    (self.average_policy[obs] * (self.iteration - 1) + self.policy[obs]) / self.iteration
                )
            else:
                self.average_policy[obs] = self.policy[obs]
        # print(self.average_policy)
    
    """Compute action probabilities using regret matching."""
    def regret_matching(self, obs):
        '''
        Args:
            obs: The current observation/state.
        Returns:
            action_probs (np.array): The action probabilities for the given observation.
        '''
        regret = self.regrets[obs]
        positive_regret_sum = np.sum(np.maximum(regret, 0)) # we only care positive regret
        action_probs = np.zeros(self.env.num_actions)
        if positive_regret_sum > 0:
            for action in range(self.env.num_actions):
                action_probs[action] = max(0.0, regret[action] / positive_regret_sum)
        else:
            action_probs = np.ones(self.env.num_actions) / self.env.num_actions # Equal probability if no positive regret

        return action_probs
    
    """Get action probabilities for a given observation and set of legal actions."""
    def action_probs(self, obs, legal_actions, policy):
        '''
        Args:
            obs: The current observation/state.
            legal_actions: List of legal actions available in the current state.
            policy: The current policy to use for generating action probabilities.

        Returns:
            action_probs (np.array): The action probabilities for the legal actions.
        '''
        if obs not in policy:
            action_probs = np.zeros(self.env.num_actions)
            action_probs[legal_actions] = 1.0 / len(legal_actions) # Equal probability for legal actions
            policy[obs] = action_probs # Initialize the policy for the observation
        else:
            action_probs = policy[obs]
        action_probs = self.remove_illegal(action_probs, legal_actions) # Get the action probabilities from the policy
        return action_probs

    """Remove illegal actions and normalize theprobability vector"""
    '''Only legal actions should be allocated probabilities, we won't take actions that we can not take'''
    '''Is called in action_probs'''
    def remove_illegal(self, action_probs, legal_actions):
        ''' 
        Args:
            action_probs (numpy.array): A 1 dimention numpy array.
            legal_actions (list): A list of indices of legal actions.
        Returns:
            probd (numpy.array): A normalized vector without legal actions.
        '''
        probs = np.zeros(action_probs.shape[0])
        probs[legal_actions] = action_probs[legal_actions]
        if np.sum(probs) == 0:
            probs[legal_actions] = 1 / len(legal_actions)
        else:
            probs /= sum(probs)
        return probs

    """Given a state, predict action based on average policy"""
    '''This is used when we play game with other agents/human and would like to compare the performance'''
    def eval_step(self, state):
        ''' 
        Args:
            state (numpy.array): State representation
        Returns:
            action (int): Predicted action
            info (dict): A dictionary containing information
        '''
        probs = self.action_probs(state['obs'].tostring(), list(state['legal_actions'].keys()), self.average_policy)
        action = np.random.choice(len(probs), p=probs)
        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: float(probs[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}
        return action, info

    """Get state_str of the player, the information set"""
    def get_state(self, player_id):
        ''' 
        Args:
            player_id (int): The player id
        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        '''
        # Retrieve the state of the given player from the environment(env class and game class)
        state = self.env.get_state(player_id)
        obs = np.array_str(state['obs'])
        
        state_dict = {
            # 'player_id': player_id,
            'obs': state['obs'].tolist(),  # Convert numpy array to list for json serialization
            'action_record': state['action_record']} # Include the action record in the state
        combined_obs_str = json.dumps(state_dict)
        # Return the combined observation string and the list of legal action indices
        return obs, list(state['legal_actions'].keys())
    
    """Save model"""        
    def save(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'),'wb')
        pickle.dump(self.average_policy, average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'),'wb')
        pickle.dump(self.regrets, regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'wb')
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    """Load model"""   
    def load(self):
        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'rb')
        self.policy = pickle.load(policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'),'rb')
        self.average_policy = pickle.load(average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'),'rb')
        self.regrets = pickle.load(regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'rb')
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()



# %%
