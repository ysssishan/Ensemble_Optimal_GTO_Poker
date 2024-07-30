# %% [markdown]
# # Game Env

# %%
import hashlib
import numpy as np
import collections
import os
import argparse
import pickle
import struct
import json
from collections import OrderedDict

from Basic_Leduc_Game import LeducholdemGame

# %% [markdown]
# ### Seeding

# %%
def colorize(string, color, bold=False, highlight = False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    attrs = ';'.join(attr)
    return '\x1b[%sm%s\x1b[0m' % (attrs, string)

def error(msg, *args):
    print(colorize('%s: %s'%('ERROR', msg % args), 'red'))

def np_random(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise error.Error('Seed must be a non-negative integer or omitted, not {}'.format(seed))

    seed = create_seed(seed)

    rng = np.random.RandomState()
    rng.seed(_int_list_from_bigint(hash_seed(seed)))
    return rng, seed

def hash_seed(seed=None, max_bytes=8):
    if seed is None:
        seed = create_seed(max_bytes=max_bytes)
    _hash = hashlib.sha512(str(seed).encode('utf8')).digest()
    return _bigint_from_bytes(_hash[:max_bytes])

def create_seed(a=None, max_bytes=8):
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, str):
        a = a.encode('utf8')
        a += hashlib.sha512(a).digest()
        a = _bigint_from_bytes(a[:max_bytes])
    elif isinstance(a, int):
        a = a % 2**(8 * max_bytes)
    else:
        raise error.Error('Invalid type for seed: {} ({})'.format(type(a), a))

    return a

def _bigint_from_bytes(_bytes):
    sizeof_int = 4
    padding = sizeof_int - len(_bytes) % sizeof_int
    _bytes += b'\0' * padding
    int_count = int(len(_bytes) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), _bytes)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum

def _int_list_from_bigint(bigint):
    # Special case 0
    if bigint < 0:
        raise error.Error('Seed must be non-negative, not {}'.format(bigint))
    elif bigint == 0:
        return [0]

    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints

# %% [markdown]
# ### Baseline Leduc Holdem Env

# %% [markdown]
# **Basic game env.**

# %%
class LeducholdemEnv:
    """Initialize the Leducholdem environment"""
    def __init__(self, config):    
        self.game = LeducholdemGame(config)
        self.action_recorder = []
        self.allow_step_back = self.game.allow_step_back = config['allow_step_back']
        
        # Get the number of players/actions in this game
        self.num_players = self.game.get_num_players()
        self.num_actions = self.game.get_num_actions()
        
        # A counter for the timesteps
        self.timestep = 0
        
        # Set random seed, default is None
        self.seed(config['seed'])
        
        self.actions = ['call', 'raise', 'fold', 'check']
        self.card2index = {"SJ": 0, "SQ": 1, "SK": 2, "HJ": 0, "HQ": 1, "HK": 2}
    
    """Start a new game"""
    def reset(self):
        ''' 
        Returns:
            (tuple): Tuple containing:
                (numpy.array): The beginning state of the game
                (int): The beginning player
        '''     
        state, player_id = self.game.init_game()
        self.action_recorder = []
        return self._extract_state(state), player_id
    
    """Step forward"""
    def step(self, action, raw_action=False): 
        ''' 
        Args:
            action (int): The action taken by the current player
            raw_action (boolean): True if the action is a raw action
        Returns:
            (tuple): Tuple containing:
                (dict): The next state
                (int): The ID of the next player
        '''   
        if not raw_action:
            action = self._decode_action(action)
        self.timestep += 1
        # Record the action for human interface
        self.action_recorder.append((self.get_player_id(), action))
        next_state, player_id = self.game.step(action)
        return self._extract_state(next_state), player_id

    """Take one step backward."""
    def step_back(self):
        ''' 
        Returns:
            (tuple): Tuple containing:
                (dict): The previous state
                (int): The ID of the previous player
        Note: Error will be raised if step back from the root node.
        '''
        if not self.allow_step_back:
            raise Exception("Step back not allowed")

        if not self.game.step_back():
            return False

        # Remove the last action from action_recorder
        if self.action_recorder:
            self.action_recorder.pop()
            
        player_id = self.get_player_id()
        state = self.get_state(player_id)

        return state, player_id
    
    """Set the agents that will interact with the environment. Must be called before `run`."""
    def set_agents(self, agents):
        '''
        Args:
            agents (list): List of Agent classes
        '''        
        self.agents = agents

    """Run a complete game, either for evaluation or training RL agent."""
    def run(self, is_training=False):
        '''
        Args:
            is_training (boolean): True if for training purpose.
        Returns:
            (tuple) Tuple containing:
                (list): A list of trajectories generated from the environment.
                (list): A list payoffs. Each entry corresponds to one player.
        Note: The trajectories are 3-dimension list.
                The first dimension is for different players.
                The second dimension is for different transitions. 
                The third dimension is for the contents of each transition.
        '''    
        trajectories = [[] for _ in range(self.num_players)]
        state, player_id = self.reset()

        # Loop to play the game
        trajectories[player_id].append(state)
        while not self.is_over():
            # Agent plays
            if not is_training:
                action, _ = self.agents[player_id].eval_step(state)
                print(f'state {state}')
                obs_in_agent = json.dumps({'obs': state['obs'].tolist(), 'action_record': state['action_record']})
                print(f'obs_in sagent {obs_in_agent}')
            else:
                action = self.agents[player_id].step(state)

            # Environment steps
            next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
            # Save action
            trajectories[player_id].append(action)

            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.game.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.num_players):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        return trajectories, payoffs

    """Check whether the current game is over"""
    def is_over(self):
        ''' 
        Returns:
            (boolean): True if current game is over
        '''
        return self.game.is_over()

    """Get the current player id"""
    def get_player_id(self):
        ''' 
        Returns:
            (int): The id of the current player
        '''
        return self.game.get_player_id()

    """Get the state given player id"""
    def get_state(self, player_id):
        '''     
        Args:
            player_id (int): The player id
        Returns:
            (numpy.array): The observed state of the player
        '''
        return self._extract_state(self.game.get_state(player_id))

    """Get the payoffs of players"""
    def get_payoffs(self):
        ''' 
        Returns:
            (list): A list of payoffs for each player.
        '''
        return self.game.get_payoffs()

    """Get the perfect information of the current state"""
    def get_perfect_information(self):
        ''' 
        Returns:
            (dict): A dictionary of all the perfect information of the current state
        ''' 
        state = {}
        state['chips'] = [self.game.players[i].in_chips for i in range(self.num_players)]
        state['public_card'] = self.game.public_card.get_index() if self.game.public_card else None
        state['hand_cards'] = [self.game.players[i].hand.get_index() for i in range(self.num_players)]
        state['current_round'] = self.game.round_counter
        state['current_player'] = self.game.game_pointer
        state['legal_actions'] = self.game.get_legal_actions()
        return state

    """Extract useful information from state for RL"""
    def _extract_state(self, state): 
        ''' 
        Args:
            state (dict): The raw state
        Returns:
            (numpy.array): The extracted state
        '''   
        extracted_state = {}

        legal_actions = OrderedDict({self.actions.index(a): None for a in state['legal_actions']})
        extracted_state['legal_actions'] = legal_actions # actions index (int, not str)

        public_card = state['public_card']
        hand = state['hand']
        
        obs = np.zeros(8)
        obs[self.card2index[hand]] = 1
        if public_card:
            obs[self.card2index[public_card]+3] = 1
        obs[6] = state['my_chips']
        obs[7] = sum(state['all_chips'])-state['my_chips']
        
        extracted_state['obs'] = obs
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder

        return extracted_state

    """Decode Action id to the action in the game"""
    def _decode_action(self, action_id):
        ''' 
        Args:
            action_id (int): The id of the action
        Returns:
            (string): The action that will be passed to the game engine.      
        '''
        legal_actions = self.game.get_legal_actions()
        if self.actions[action_id] not in legal_actions:
            if 'check' in legal_actions:
                return 'check'
            else:
                return 'fold'
        return self.actions[action_id]

    """Get all legal actions for current state"""
    def _get_legal_actions(self):
        ''' 
        Returns:
            (list): A list of legal actions' id.    
        '''
        return self.game.get_legal_actions()

    '''Seeding'''
    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        self.game.np_random = self.np_random
        return seed


