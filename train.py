from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.game import setup_config, start_poker
from collections import deque
import torch
import torch.nn as nn
import numpy as np
import random
import copy
import os

from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai

SUITS = ('C', 'D', 'H', 'S')
RANKS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K')
STREETS = ('preflop', 'flop', 'turn', 'river')

SAVE_PATH = './checkpoint.pt'
NUM_STATES = 379
"""
1. pot
2. max_bet
3. max_aggresion
4. stack_size
5. hand_strength
6. preflop
7. flop
8. turn
9. river
10  ~  61. hole_cards[0]
62  ~ 113. hole_cards[1]
114 ~ 165. community_cards[0]
166 ~ 217. community_cards[1]
218 ~ 269. community_cards[2]
270 ~ 321. community_cards[3]
322 ~ 373. community_cards[4]
374 ~ 379. position (at most 6)
"""

NUM_ACTIONS = 7
action_set = {
    0: 'fold',
    1: 'call',
    2: 'raise', # small raise
    3: 'raise',
    4: 'raise',
    5: 'raise',
    6: 'raise'
}

class Player(BasePokerPlayer):
    def __init__(self, agent):
        self.agent = agent
        self.hole_cards = np.zeros((2, 52))
        self.community_cards = np.zeros((5, 52))
        self.position = np.zeros(6)
        self.street = np.zeros(4)
        self.stack_size = 0
        self.max_bet = 0
        self.history = [] # (state, action, reward, next_state, legal_actions, done)
        
    def get_history(self):
        return copy.deepcopy(self.history)
        
    def __card_to_value(self, card):
        return (13 * SUITS.index(card[0])) + RANKS.index(card[1])
        
    def declare_action(self, valid_actions, hole_card, round_state):
        self.stack_size = next(player['stack'] for player in round_state['seats'] if player['uuid'] == self.uuid)
        legal_actions = [0, 1]
        increment = 0.7 # increase in each level of raise
        if len(valid_actions) == 3:
            min_amount = valid_actions[2]['amount']['min']
            max_amount = valid_actions[2]['amount']['max']
            for i in range(2, NUM_ACTIONS):
                if min_amount <= increment * i * self.max_bet <= max_amount:
                    legal_actions.append(i)
        
        hand_strength = HandEvaluator.eval_hand(hole_card, round_state['community_card'])
        
        pot_size = round_state['pot']['main']['amount']
        if self.uuid in round_state['pot']['side']['eligibles']:
            pot_size += round_state['pot']['side']['amount']
            
        state = np.concatenate([
            [pot_size],
            [self.max_bet],
            [self.stack_size],
            [hand_strength],
            self.street,
            self.hole_cards.flatten(),
            self.community_cards.flatten(),
            self.position,
            self.stack
        ])
        best_action = self.agent.step(state, legal_actions)
        if best_action >= 2:
            amount = best_action * increment
        
        if self.history:
            self.history[-1][3] = state
            
        self.history.append((state, best_action, 0, None, legal_actions, False))
        return action_set[best_action], amount

    def receive_game_start_message(self, game_info):
        self.stack = game_info['rule']['initial_stack']
        self.num_players = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.done = False
        values = sorted([self.__card_to_value(hole_card[0]), self.__card_to_value(hole_card[1])])
        self.hole_cards[0][values[0]] = 1
        self.hole_cards[1][values[1]] = 1
        
        for i in range(self.num_players):
            if seats[i]['uuid'] == self.uuid:
                self.position[i] = 1
            else:
                self.position[i] = 0

    def receive_street_start_message(self, street, round_state):
        self.max_bet = 0
        self.street.fill(0)
        self.street[STREETS.index(street)] = 1
        
    def receive_game_update_message(self, action, round_state):
        self.max_bet = max(self.max_bet, action['amount'])
        seats = round_state['seats']

    def receive_round_result_message(self, winners, hand_info, round_state):
        if self.history:
            self.history[-1][2] = next(player['stack'] for player in round_state['seats'] if player['uuid'] == self.uuid)
            self.history[-1][3] = np.zeros(NUM_STATES)
            self.history[-1][5] = True

class Agent(object):
    def __init__(self,
                 replay_size=15000,
                 update_target_freq=1000,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=0.995,
                 discount=0.9,
                 batch_size=150,
                 train_freq=1,
                 mlp_layers=None,
                 learning_rate=0.001,
                 save_path=None,
                 save_freq=1000):
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.mlp_layers = mlp_layers
        self.estimator = Estimator(num_states=NUM_STATES, mlp_layers=self.mlp_layers)
        self.target_estimator = Estimator(num_states=NUM_STATES, mlp_layers=self.mlp_layers)
        
        self.update_target_freq = update_target_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.batch_size = batch_size
        self.train_freq = train_freq      
        self.learning_rate = learning_rate  
        
        self.replay = deque(maxlen=replay_size)
        
        self.train_t = 0
        self.total_t = 0
        
        self.save_path = save_path
        self.save_freq = save_freq
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.estimator.parameters(), lr=learning_rate)
    
    def __predict_nograd(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            q_values = self.estimator(state).cpu().numpy()
        return q_values
    
    def predict(self, state, legal_actions):
        q_values = self.__predict_nograd(state)
        masked_q_values = -np.inf * np.ones(NUM_ACTIONS, dtype=float)
        masked_q_values[legal_actions] = q_values[legal_actions]
        return masked_q_values[legal_actions]
    
    def step(self, state, legal_actions):
        q_values = self.predict(state)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (self.epsilon_decay ** self.total_t)
        if random.random() < epsilon:
            action = random.choice(legal_actions)
        else:
            action = np.argmax(q_values)
        return action
    
    def feed(self, transition):
        state, action, reward, next_state, legal_actions, done = tuple(transition)
        self.replay.append((state, action, reward, next_state, legal_actions, done))
        self.total_t += 1
        if self.total_t >= self.batch_size:
            self.train()
        
    def train(self):
        mini_batch = random.sample(self.replay, self.batch_size)
        state_batch = torch.cat([s1 for (s1,a,r,s2,l,d) in mini_batch])  
        next_state_batch = torch.cat([s2 for (s1,a,r,s2,l,d) in mini_batch])   
        action_batch = torch.Tensor([a for (s1,a,r,s2,l,d) in mini_batch])
        reward_batch = torch.Tensor([r for (s1,a,r,s2,l,d) in mini_batch])
        done_batch = torch.Tensor([d for (s1,a,r,s2,l,d) in mini_batch])
        
        legal_batch = [l for (s1,a,r,s2,l,d) in mini_batch]
        
        legal_actions = []
        for b in range(self.batch_size):
            legal_actions.extend([b * NUM_ACTIONS + i for i in legal_batch[b]])
        
        q_values_next = self.__predict_nograd(state_batch)
        masked_q_values = -np.inf * np.ones(self.batch_size * NUM_ACTIONS, dtype=float)
        masked_q_values[legal_actions] = q_values_next[legal_actions]
        masked_q_values = masked_q_values.reshape((self.batch_size, NUM_ACTIONS))
        actions = np.argmax(masked_q_values, axis=1)
        
        q_values_next_target = self.__predict_nograd(next_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.discount * q_values_next_target[np.arange(self.batch_size), actions]
        
        state_batch = np.array(state_batch)
        
        self.optimizer.zero_grad()
        self.estimator.train()
        
        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        action_batch = torch.from_numpy(action_batch).long().to(self.device)
        target_batch = torch.from_numpy(target_batch).float().to(self.device)
        
        Q = torch.gather(self.estimator(state_batch), dim=-1, index=action_batch.unsqueeze(-1)).squeeze(-1)
        
        loss = self.criterion(Q, target_batch)
        loss.backward()
        self.optimizer.step()
        
        if self.train_t % self.update_target_freq == 0:
            self.target_estimator = copy.deepcopy(self.estimator)
            
        self.train_t += 1
        
        if self.train_t % 100 == 0:
            print(f'iteration {self.train_t}, Loss = {loss.item()}')
        
        if self.save_path and self.train_t % self.save_freq == 0:
            self.save_checkpoint(self.save_path)
            
    @classmethod
    def from_checkpoint(cls, checkpoint: dict):
        agent = cls(
            update_target_freq=checkpoint['update_target_freq'],
            epsilon_start=checkpoint['epsilon_start'],
            epsilon_end=checkpoint['epsilon_end'],
            epsilon_decay=checkpoint['epsilon_decay'],
            discount=checkpoint['discount'],
            batch_size=checkpoint['batch_size'],
            train_freq=checkpoint['train_freq'],
            mlp_layers=checkpoint['mlp_layers'],
            learning_rate=checkpoint['learning_rate'],
            save_path=checkpoint['save_path'],
            save_freq=checkpoint['save_freq']            
        )
        agent.device = checkpoint['device']
        
        agent.total_t = checkpoint['total_t']
        agent.train_t = checkpoint['train_t']
        
        agent.estimator = Estimator(NUM_STATES, checkpoint['mlp_layers'])
        agent.estimator.load_state_dict(checkpoint['estimator'])
        agent.target_estimator = Estimator(NUM_STATES, checkpoint['mlp_layers'])
        agent.target_estimator.load_state_dict(checkpoint['target_estimator'])
        
        agent.replay = checkpoint['replay']
        agent.optimizer.load_state_dict(checkpoint['optimzier'])
        
    def save_checkpoint(self, path, filename='agent.pt'):
        attr = {
            'estimator': self.estimator.state_dict(),
            'target_estimator': self.target_estimator.state_dict(),
            'replay': self.replay,
            'total_t': self.total_t,
            'train_t': self.train_t,
            'update_target_freq': self.update_target_freq,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'discount': self.discount,
            'batch_size': self.batch_size,
            'train_freq': self.train_freq,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer.state_dict(),
            'mlp_layers': self.mlp_layers,
            'device': self.device,
            'save_path': self.save_path,
            'save_freq': self.save_freq
        }
        torch.save(attr, os.path.join(path, filename))

class Estimator(nn.Module):
    def __init__(self, num_states, mlp_layers=None):
        super(Estimator, self).__init__()
        
        self.num_states = num_states
        self.mlp_layers = mlp_layers
        
        layer_dims = [num_states] + self.mlp_layers
        fc = [nn.BatchNorm1d(layer_dims[0])]
        for i in range(1, len(layer_dims)):
            fc.append(nn.Linear(layer_dims[i - 1], layer_dims[i], bias=True))
            fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], NUM_ACTIONS, bias=True))
        self.fc = nn.Sequential(*fc)
        
    def forward(self, X):
        return self.fc(X)

def train(baselines, mlp_layers=[64,64], episodes=1000, lr=0.001, batch_size=500):
    file = open('training.log', 'a')
    if os.path.isfile(SAVE_PATH) and os.access(SAVE_PATH, os.R_OK):
        agent = Agent.from_checkpoint(checkpoint=torch.load(SAVE_PATH))
    else:
        agent = Agent(learning_rate=lr, batch_size=batch_size, mlp_layers=mlp_layers, save_path=SAVE_PATH)
    
    for episode in range(episodes):
        player = Player(agent=agent)
        config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
        config.register_player(name='p0', algorithm=player)
        for i in range(len(baselines)):
            config.register_player(name=f'p{i}', algorithm=baselines[i]())
        
        game_result = start_poker(config, verbose=0)
        history = player.get_history()
        
        for i in range(len(history)):
            agent.feed(history[i])

baselines = [baseline0_ai] * 5
train(baselines=baselines)