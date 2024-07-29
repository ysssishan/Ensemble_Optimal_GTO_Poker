
# %% [markdown]
# ## Leduc Hold'em Game Code

# %% [markdown]
# ### Import

# %%
from copy import deepcopy, copy
import numpy as np
import csv
import os

# %% [markdown]
# ### Card

# %% [markdown]
# **Card stores the suit and rank of a single card.**
#     
#     Note:
#         The suit variable in a standard card game should be one of [S, H, D, C, BJ, RJ] meaning [Spades, Hearts, Diamonds, Clubs, Black Joker, Red Joker]
#         Similarly the rank variable should be one of [A, 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K]

# %%
class Card:
    
    suit = None
    rank = None
    valid_suit = ['S', 'H', 'D', 'C', 'BJ', 'RJ']
    valid_rank = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

    """Initialize the suit and rank of a card"""
    def __init__(self, suit, rank):        
        '''
        Args:
            suit: string, suit of the card, should be one of valid_suit
            rank: string, rank of the card, should be one of valid_rank
        '''
        self.suit = suit
        self.rank = rank

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        else:
            # don't attempt to compare against unrelated types
            return NotImplemented

    def __hash__(self):
        suit_index = Card.valid_suit.index(self.suit)
        rank_index = Card.valid_rank.index(self.rank)
        return rank_index + 100 * suit_index

    """Get string representation of a card."""
    def __str__(self):
        ''' 
        Returns:
            string: the combination of rank and suit of a card. Eg: AS, 5H, JD, 3C, ...
        '''    
        return self.rank + self.suit
  
    """Get index of a card.""" 
    def get_index(self):
        '''    
        Returns:
            string: the combination of suit and rank of a card. Eg: 1S, 2H, AD, BJ, RJ...
        '''
        return self.suit + self.rank

# %% [markdown]
# ### Util

# %%
def set_seed(seed):
    if seed is not None:
        import subprocess
        import sys

        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
        if 'torch' in installed_packages:
            import torch
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)


"""Get the coresponding number of a rank."""
def rank2int(rank):
    '''
    Args:
        rank(str): rank stored in Card objec
    Returns:
        (int): the number corresponding to the rank
    Note:
        1. If the input rank is an empty string, the function will return -1.
        2. If the input rank is not valid, the function will return None.
    '''
    if rank == '':
        return -1
    elif rank.isdigit():
        if int(rank) >= 2 and int(rank) <= 10:
            return int(rank)
        else:
            return None
    elif rank == 'A':
        return 14
    elif rank == 'T':
        return 10
    elif rank == 'J':
        return 11
    elif rank == 'Q':
        return 12
    elif rank == 'K':
        return 13
    return None


# %% [markdown]
# ### Dealer

# %%
class LeducholdemDealer:
    
    """Initialize a leducholdem dealer class"""
    def __init__(self, np_random):

        self.np_random = np_random
        self.deck = [Card('S', 'J'), Card('H', 'J'), Card('S', 'Q'), Card('H', 'Q'), Card('S', 'K'), Card('H', 'K')]
        self.shuffle()
        self.pot = 0
    
    def shuffle(self):
        self.np_random.shuffle(self.deck)

    """Deal one card from the deck"""
    def deal_card(self): 
        '''
        Returns:
            (Card): The drawn card from the deck
    '''   
        return self.deck.pop()

# %% [markdown]
# ### Player

# %%
class LeducholdemPlayer:
    
    """Initilize a player."""
    def __init__(self, player_id, np_random):
        ''' 
        Args:
            player_id (int): The id of the player
        '''
        self.np_random = np_random
        self.player_id = player_id
        self.status = "alive"
        self.hand = None 
        self.in_chips = 0 # The chips that this player has put in until now

    """Encode the state for the player"""
    def get_state(self, public_card, all_chips, legal_actions): 
        ''' 
        Args:
            public_card (object): The public card that seen by all the players
            all_chips (int): The chips that all players have put in
        Returns:
            (dict): The state of the player
        '''   
        state = {}
        state['hand'] = self.hand.get_index()
        
        state['public_card'] = public_card.get_index() if public_card else None
        
        state['all_chips'] = all_chips
        state['my_chips'] = self.in_chips
        state['legal_actions'] = legal_actions
        return state

    """Return the id of the player"""
    def get_player_id(self):
        return self.player_id

# %% [markdown]
# ### Game Winner Judger

# %%
class LeducholdemJudger:
    
    """Initialize a judger class"""
    def __init__(self, np_random):    
        self.np_random = np_random

    """Judge the winner of the game."""
    @staticmethod
    def judge_game(players, public_card):
        '''
        Args:
            players (list): The list of players who play the game
            public_card (object): The public card that seen by all the players
        Returns:
            (list): Each entry of the list corresponds to one entry of the
        '''

        # Judge who are the winners
        winners = [0] * len(players) # in leduc: a 2 elements list
        fold_count = 0
        ranks = []
        
        # If every player folds except one, the alive player is the winner
        for idx, player in enumerate(players):
            ranks.append(rank2int(player.hand.rank))
            if player.status == 'folded':
               fold_count += 1
            elif player.status == 'alive':
                alive_idx = idx
        if fold_count == (len(players) - 1):
            winners[alive_idx] = 1
        
        # If any of the players matches the public card wins
        if public_card is not None:
            if sum(winners) < 1:
                for idx, player in enumerate(players):
                    if player.hand.rank == public_card.rank:
                        winners[idx] = 1
                        break
        
        # If non of the above conditions, the winner player is the one with the highest card rank
        if sum(winners) < 1:
            max_rank = max(ranks)
            max_index = [i for i, j in enumerate(ranks) if j == max_rank]
            for idx in max_index:
                winners[idx] = 1

        # Compute the total chips
        total = 0
        for p in players:
            total += p.in_chips

        # each_win = float(total) / sum(winners)
        if sum(winners) > 0:
            each_win = float(total) / sum(winners)
        else:
            each_win = 0.0

        payoffs = []
        for i, _ in enumerate(players):
            if winners[i] == 1:
                payoffs.append(each_win - players[i].in_chips)
            else:
                payoffs.append(float(-players[i].in_chips))

        return payoffs

# %% [markdown]
# ### Round

# %% [markdown]
# **Round can call other Classes' functions to keep the game running.**

# %%
class LeducholdemRound:
    
    """Initialize the round class"""
    def __init__(self, raise_amount, allowed_raise_num, num_players, np_random):
        '''
        Args:
            raise_amount (int): the raise amount for each raise
            allowed_raise_num (int): The number of allowed raise num
            num_players (int): The number of players
        '''
        self.np_random = np_random
        self.game_pointer = None
        self.raise_amount = raise_amount
        self.allowed_raise_num = allowed_raise_num
        self.num_players = num_players

        # Count the number of raise
        self.have_raised = 0

        # Count the number of player without raise
        # If every player agree to not raise, the round is over
        self.not_raise_num = 0

        # Raised amount for each player
        self.raised = [0 for _ in range(self.num_players)] # in leduc, [0,0]
        self.player_folded = None

    """Start a new bidding round"""
    def start_new_round(self, game_pointer, raised=None): 
        '''
        Args:
            game_pointer (int): The game_pointer that indicates the next player
            raised (list): Initialize the chips for each player
        Note: For the first round of the game, we need to setup the big/small blind
        '''       
        self.game_pointer = game_pointer
        self.have_raised = 0
        self.not_raise_num = 0
        self.have_folded = 0
        if raised:
            self.raised = raised
        else:
            self.raised = [0 for _ in range(self.num_players)]

    """Call Player Class functions to keep one round running"""
    def proceed_round(self, players, action):
        '''
        Args:
            players (list): The list of players that play the game
            action (str): An legal action taken by the player
        Returns:
            (int): The game_pointer that indicates the next player
        '''
        
        if action not in self.get_legal_actions():
            raise Exception('{} is not legal action. Legal actions: {}'.format(action, self.get_legal_actions()))

        if action == 'call':
            diff = max(self.raised) - self.raised[self.game_pointer] # the chips that should be put by the game pointer if he calls
            self.raised[self.game_pointer] = max(self.raised) # update the present maximum chip amount
            players[self.game_pointer].in_chips += diff # update the total chips put by the game pointer
            self.not_raise_num += 1 # number of players without raise

        elif action == 'raise':
            diff = max(self.raised) - self.raised[self.game_pointer] + self.raise_amount # the present maximum betting chips + raise chips - the game pointer's betting chips
            self.raised[self.game_pointer] = max(self.raised) + self.raise_amount # update the present maximum chip amount
            players[self.game_pointer].in_chips += diff # update the total chips put by the game pointer
            self.have_raised += 1 # number of players that raise
            self.not_raise_num = 1 # reset

        elif action == 'fold':
            players[self.game_pointer].status = 'folded'
            self.have_folded += 1
            self.player_folded = True

        elif action == 'check':
            self.not_raise_num += 1

        self.game_pointer = (self.game_pointer + 1) % self.num_players

        # Skip the folded players
        while players[self.game_pointer].status == 'folded':
            self.game_pointer = (self.game_pointer + 1) % self.num_players

        return self.game_pointer

    """Obtain the legal actions for the current player""" 
    def get_legal_actions(self):
        '''
        Returns:
           (list):  A list of legal actions
        '''
        full_actions = ['call', 'raise', 'fold', 'check']

        # If the the number of raises already reaches the maximum number raises, we can not raise any more
        if self.have_raised >= self.allowed_raise_num:
            full_actions.remove('raise')

        # If the current chips are less than that of the highest one in the round, we can not check
        if self.raised[self.game_pointer] < max(self.raised):
            full_actions.remove('check')

        # If the current player has put in the chips that are more than others, we can not call
        if self.raised[self.game_pointer] == max(self.raised):
            full_actions.remove('call')
        return full_actions
    
    """Check whether the round is over"""
    def is_over(self):
        '''
        Returns:
            (boolean): True if the current round is over
        '''
        # call, check -> not_raise_num + 1    
        if self.not_raise_num >= self.num_players or self.have_folded == 1:
            return True   
        return False

# %% [markdown]
# ### Game

# %%
class LeducholdemGame():
    
    """Initialize the class leducholdem Game, Set game rules"""
    def __init__(self, config):
        ''' 
        Configs:
            num_players (int): the number of game players
            small_blind (int): The amount of small blind
            allowed_raise_num (int): the maximum raise chances
            allow_step_back (Boolean)
            seed (int)
        '''
        self.allow_step_back = config.get('allow_step_back', True)
        self.np_random = np.random.RandomState(config.get('seed', None))
        
        self.num_players = config.get('game_num_players', 2)

        # Small blind and big blind
        self.small_blind = config.get('small_blind', 1)
        self.big_blind = 2 * self.small_blind

        # Raise amount and allowed times
        self.raise_amount = self.big_blind
        self.allowed_raise_num = config.get('allowed_raise_num', 2)
    
    ''' Specifiy some game specific parameters'''
    def configure(self, game_config):
        self.num_players = game_config.get('game_num_players', self.num_players)
        self.small_blind = game_config.get('small_blind', self.small_blind)
        self.big_blind = 2 * self.small_blind
        self.raise_amount = self.big_blind
        self.allowed_raise_num = game_config.get('allowed_raise_num', self.allowed_raise_num)

    """Initialilze the game of Limit Texas Hold'em"""
    def init_game(self):
        ''' 
        This version supports two-player limit texas hold'em
        Returns:
            (tuple): Tuple containing:
                (dict): The first state of the game
                (int): Current player's id
        '''
        # Initilize a dealer that can deal cards
        self.dealer = LeducholdemDealer(self.np_random)

        # Initilize two players to play the game
        self.players = [LeducholdemPlayer(i, self.np_random) for i in range(self.num_players)]

        # Initialize a judger class which will decide who wins in the end
        self.judger = LeducholdemJudger(self.np_random)

        # Prepare for the first round
        for i in range(self.num_players):
            self.players[i].hand = self.dealer.deal_card()
        
        # Randomly choose a small blind(player1) and a big blind(player2)
        s = self.np_random.randint(0, self.num_players) # in leduc, either 0 or 1
        b = (s + 1) % self.num_players
        self.players[b].in_chips = self.big_blind
        self.players[s].in_chips = self.small_blind
        self.public_card = None
        
        # The player with small blind plays the first
        self.game_pointer = s

        # Initilize a betting round, in the first round, the big blind and the small blind needs to
        # be passed to the round for processing.
        self.round = LeducholdemRound(raise_amount=self.raise_amount,
                           allowed_raise_num=self.allowed_raise_num,
                           num_players=self.num_players,
                           np_random=self.np_random)

        self.round.start_new_round(game_pointer=self.game_pointer, raised=[p.in_chips for p in self.players])

        # Count the round. There are 2 rounds in each game.
        self.round_counter = 0

        # Save the hisory for stepping back to the last state.
        self.history = []

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    """Return the legal actions for current player"""
    def get_legal_actions(self):
        '''
        legal actions vary in different rounds (call Round Class to get), follow different actions of opponents
        Returns:
            (list): A list of legal actions
        '''
        return self.round.get_legal_actions()
    
    """Return the current player's id"""
    def get_player_id(self):
        '''
        Returns:
            (int): current player's id
        '''
        return self.game_pointer
    
    """Return the number of applicable actions"""
    @staticmethod
    def get_num_actions():
        '''
        Returns:
            (int): The number of actions. There are 4 actions (call, raise, check and fold)
        '''
        return 4
    
    """Return the number of players in limit texas holdem"""
    def get_num_players(self):
        '''
        Returns:
            (int): The number of players in the game
        '''
        return self.num_players
    
    """Get the next state"""
    def step(self, action):
        ''' 
        Args:
            action (str): a specific action. (call, raise, fold, or check)
        Returns:
            (tuple): Tuple containing:
                (dict): next player's state
                (int): next plater's id
        '''
        if self.allow_step_back:
            # First snapshot the current state
            r = copy(self.round)
            r_raised = copy(self.round.raised)
            gp = self.game_pointer
            r_c = self.round_counter
            d_deck = copy(self.dealer.deck)
            p = copy(self.public_card)
            ps = [copy(self.players[i]) for i in range(self.num_players)]
            ps_hand = [copy(self.players[i].hand) for i in range(self.num_players)]
            self.history.append((r, r_raised, gp, r_c, d_deck, p, ps, ps_hand))

        # Then we proceed to the next round
        self.game_pointer = self.round.proceed_round(self.players, action)

        # If a round is over, we deal more public cards
        if self.round.is_over():
            # For the first round, we deal 1 card as public card. Double the raise amount for the second round
            if self.round_counter == 0:
                self.public_card = self.dealer.deal_card()
                self.round.raise_amount = 2 * self.raise_amount

            self.round_counter += 1
            self.round.start_new_round(self.game_pointer)
        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    """Return to the previous state of the game"""
    def step_back(self):
        ''' 
        Returns:
            (bool): True if the game steps back successfully
        '''    
        if len(self.history) > 0:
            self.round, r_raised, self.game_pointer, self.round_counter, d_deck, self.public_card, self.players, ps_hand = self.history.pop()
            self.round.raised = r_raised
            self.dealer.deck = d_deck
            for i, hand in enumerate(ps_hand):
                self.players[i].hand = hand
            return True
        return False
    
    """Return player's state"""
    def get_state(self, player): 
        ''' 
        Args:
            player_id (int): player id
        Returns:
            (dict): The state of the player
        '''   
        chips = [self.players[i].in_chips for i in range(self.num_players)]
        legal_actions = self.get_legal_actions()
        state = self.players[player].get_state(self.public_card, chips, legal_actions) # class Player.get_state
        state['current_player'] = self.game_pointer
        
        return state

    """Check if the game is over"""
    def is_over(self):
        ''' 
        Returns:
            (boolean): True if the game is over
        '''    
        alive_players = [1 if p.status=='alive' else 0 for p in self.players]
        # If only one player is alive, the game is over.
        if sum(alive_players) == 1:
            return True

        # If all rounds are finshed
        if self.round_counter == 2:
            return True
        return False

    """Return the payoffs of the game"""
    def get_payoffs(self):    
        '''
        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''
        chips_payoffs = self.judger.judge_game(self.players, self.public_card)
        payoffs = np.array(chips_payoffs) # payoffs = np.array(chips_payoffs) / (self.big_blind)
        return payoffs
    
    def set_small_blind(self, small_blind):
        self.small_blind = small_blind
        # 更新其他与small blind相关的设置
        print(f"Small blind set to: {self.small_blind}")



