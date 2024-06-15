# This file should be run under the same directory as start_game.py
from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.game import setup_config, start_poker
from src.hole_card_est import HoleCardEstimator
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
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from src.agent import setup_ai as my_ai

SUITS = ('C', 'D', 'H', 'S')
RANKS = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
STREETS = ('preflop', 'flop', 'turn', 'river')

SAVE_PATH = './src/checkpoint.pt'
NUM_FEATURES = 10
"""
0. pot_size
1. dealer_btn
2. aggression
3. stack_size
4. other_stacks
5. hand_strength
6. preflop
7. flop
8. turn
9. river
"""

NUM_ACTIONS = 8
action_set = {
    0: 'fold',
    1: 'call',
    2: 'raise',
    3: 'raise',
    4: 'raise',
    5: 'raise',
    6: 'raise',
    7: 'raise'
}

class DDDQNPlayer(BasePokerPlayer):
    def __init__(self, agent):
        self.agent = agent
        self.start_stack = 0
        self.opponent_raise_count = 0
        self.opponent_move_count = 0
        self.street = np.zeros(4)
        self.hand_strength = 0
        self.last_image = None
        self.last_action = None
        self.last_features = None
        self.last_legal_actions = None
        self.history = [] # (features, image, action, reward, next_features, next_images, legal_actions, done)
        
    def get_history(self):
        return copy.deepcopy(self.history)
        
    def __cards_to_image(self, hole_cards, community_cards):
        cards = hole_cards + community_cards
        images = np.zeros((8, 17, 17))
        for i in range(len(cards)):
            images[i] = np.pad(np.zeros((4, 13)), ((6, 7), (2, 2)), 'constant', constant_values=0)
            images[i][SUITS.index(cards[i][0]) + 6][RANKS.index(cards[i][1]) + 2] = 1
        images[7] = images[:7].sum(axis=0)
        return np.swapaxes(images, 0, 2)[:, :, -1:]
        
    def declare_action(self, valid_actions, hole_card, round_state):
        dealer_btn = round_state['dealer_btn']
        
        #big_blind = round_state['big_blind_pos']
        #small_blind = round_state['small_blind_pos']
        
        stack_size = next(seat['stack'] for seat in round_state['seats'] if seat['uuid'] == self.uuid)
        other_stacks = sum([seat['stack'] for seat in round_state['seats'] if seat['uuid'] != self.uuid])
        legal_actions = [0, 1]
        if len(valid_actions) == 3:
            legal_actions.extend([action for action in range(2, NUM_ACTIONS)])
        
        pot_size = round_state['pot']['main']['amount']
        for side in round_state['pot']['side']:
            if self.uuid in side['eligibles']:
                pot_size += side['amount']
        
        aggression = self.opponent_raise_count / self.opponent_move_count if self.opponent_move_count else 0
        
        image = np.reshape(self.__cards_to_image(hole_card, round_state['community_card']), (17 * 17))
        
        features = np.concatenate([
            [pot_size],
            [dealer_btn],
            [aggression],
            [stack_size],
            [other_stacks],
            [self.hand_strength],
            self.street
        ])
        features = torch.from_numpy(features).float()
        image = torch.from_numpy(image).float()
        
        best_action = self.agent.step(features, image, legal_actions)
        
        amount = 0
        if best_action >= 2:
            min_amount = valid_actions[2]['amount']['min']
            max_amount = valid_actions[2]['amount']['max']
            amount = min_amount + int((max_amount - min_amount) * ((best_action - 2) / (NUM_ACTIONS - 2)))
        
        if self.last_features != None:
            self.history.append((self.last_features, self.last_image, self.last_action, 0, features, image, self.last_legal_actions, False))
            
        self.last_features = features
        self.last_image = image
        self.last_action = best_action
        self.last_legal_actions = legal_actions
        
        return action_set[best_action], amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hand_strength = HoleCardEstimator.eval(hole_cards=hole_card)
        self.start_stack = next(seat['stack'] for seat in seats if seat['uuid'] == self.uuid)

    def receive_street_start_message(self, street, round_state):
        self.street.fill(0)
        if street in STREETS:
            self.street[STREETS.index(street)] = 1
        
    def receive_game_update_message(self, action, round_state):
        if action['player_uuid'] != self.uuid:
            self.opponent_move_count += 1
            if action['action'] == 'raise':
                self.opponent_raise_count += 1

    def receive_round_result_message(self, winners, hand_info, round_state):        
        if self.last_features is not None:
            current_stack = next(seat['stack'] for seat in round_state['seats'] if seat['uuid'] == self.uuid)
            reward = float(current_stack - self.start_stack)
            features = torch.zeros(NUM_FEATURES).float()
            image = torch.zeros(289).float()
            self.history.append((self.last_features, self.last_image, self.last_action, reward, features, image, self.last_legal_actions, True))

class Agent(object):
    def __init__(self,
                 replay_size=5000,
                 update_target_freq=100,
                 pretrain_steps=512,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=0.995,
                 discount=0.9,
                 batch_size=128,
                 train_freq=4,
                 learning_rate=0.001,
                 save_path=None,
                 save_freq=1000):
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.estimator = Estimator(features_shape=NUM_FEATURES)
        self.estimator.eval()
        
        self.target_estimator = Estimator(features_shape=NUM_FEATURES)
        self.target_estimator.eval()
        
        self.replay = ReplayBuffer(replay_size)
        self.replay_size = replay_size
        self.update_target_freq = update_target_freq
        self.pretrain_steps = pretrain_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.learning_rate = learning_rate
        
        self.train_t = 0
        self.total_t = 0
        
        self.save_path = save_path
        self.save_freq = save_freq
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.estimator.parameters(), lr=learning_rate)
        
        self.loss_value = 0
    
    def __predict_nograd(self, features, image, use_target=False):
        with torch.no_grad():
            features = features.float().to(self.device)
            image = image.float().to(self.device)
            if use_target == True:
                q_values = self.target_estimator(features=features, image=image).cpu()
            else:
                q_values = self.estimator(features=features, image=image).cpu()
        return q_values
    
    def predict(self, features, image, legal_actions):
        q_values = self.__predict_nograd(features, image)
        masked_q_values = -np.inf * np.ones(NUM_ACTIONS, dtype=float)
        masked_q_values[legal_actions] = q_values[legal_actions]
        return masked_q_values
    
    def step(self, features, image, legal_actions):
        q_values = self.predict(features, image, legal_actions)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (self.epsilon_decay ** self.total_t)
        if random.random() < epsilon:
            action = random.choice(legal_actions)
        else:
            action = np.argmax(q_values)
        return action
    
    def feed(self, transition: tuple):
        self.replay.save(transition)
        self.total_t += 1
        if self.total_t >= self.pretrain_steps and self.total_t % self.train_freq == 0:
            self.train()
        
    def train(self):
        mini_batch = self.replay.sample(batch_size=self.batch_size)
        feature_batch = torch.stack([f1 for (f1,i1,a,r,f2,i2,l,d) in mini_batch]).to(self.device)
        image_batch = torch.stack([i1 for (f1,i1,a,r,f2,i2,l,d) in mini_batch]).to(self.device)
        next_feature_batch = torch.stack([f2 for (f1,i1,a,r,f2,i2,l,d) in mini_batch]).to(self.device)
        next_image_batch = torch.stack([i2 for (f1,i1,a,r,f2,i2,l,d) in mini_batch]).to(self.device)
        action_batch = torch.tensor([a for (f1,i1,a,r,f2,i2,l,d) in mini_batch], dtype=torch.long, device=self.device)
        reward_batch = torch.tensor([r for (f1,i1,a,r,f2,i2,l,d) in mini_batch], dtype=torch.float, device=self.device)
        done_batch = torch.tensor([float(d) for (f1,i1,a,r,f2,i2,l,d) in mini_batch], dtype=torch.float, device=self.device)
        
        legal_batch = [l for (f1,i1,a,r,f2,i2,l,d) in mini_batch]
        
        legal_actions = []
        for b in range(self.batch_size):
            legal_actions.extend([b * NUM_ACTIONS + i for i in legal_batch[b]])
        
        q_values_next = self.__predict_nograd(feature_batch, image_batch)
        masked_q_values = -np.inf * np.ones(self.batch_size * NUM_ACTIONS, dtype=float)
        masked_q_values[legal_actions] = q_values_next.flatten()[legal_actions]
        masked_q_values = torch.tensor(masked_q_values, device=self.device).reshape((self.batch_size, NUM_ACTIONS))
        actions = torch.argmax(masked_q_values, dim=1)
        
        q_values_next_target = self.__predict_nograd(next_feature_batch, next_image_batch, use_target=True)
        target_batch = reward_batch + (1.0 - done_batch) * self.discount * q_values_next_target[torch.arange(self.batch_size), actions].detach()
        
        self.optimizer.zero_grad()
        self.estimator.train()
        
        feature_batch = feature_batch.float().to(self.device)
        action_batch = action_batch.long().to(self.device)
        target_batch = target_batch.float().to(self.device)
        
        Q = torch.gather(self.estimator(feature_batch, image_batch), dim=-1, index=action_batch.unsqueeze(-1)).squeeze(-1)
        
        loss = self.criterion(Q, target_batch)
        loss.backward()
        self.optimizer.step()
        self.estimator.eval()
        self.loss_value += loss.item()
        
        self.train_t += 1
        
        if self.train_t % self.update_target_freq == 0:
            self.target_estimator.load_state_dict(self.estimator.state_dict())
        
        if self.train_t % 500 == 0:
            print(f'iteration {self.train_t}, Loss = {self.loss_value / 500}')
            self.loss_value = 0
        
        if self.save_path and self.train_t % self.save_freq == 0:
            self.save_checkpoint(self.save_path)
            
    @classmethod
    def from_checkpoint(cls, checkpoint: dict, replay_buffer_file='./src/replay.npy', save_freq=None):
        if save_freq is None:
            save_freq = checkpoint['save_freq']
            
        agent = cls(
            pretrain_steps=checkpoint['pretrain_steps'],
            update_target_freq=checkpoint['update_target_freq'],
            epsilon_start=checkpoint['epsilon_start'],
            epsilon_end=checkpoint['epsilon_end'],
            epsilon_decay=checkpoint['epsilon_decay'],
            discount=checkpoint['discount'],
            batch_size=checkpoint['batch_size'],
            train_freq=checkpoint['train_freq'],
            learning_rate=checkpoint['learning_rate'],
            save_path=checkpoint['save_path'],
            save_freq=save_freq
        )
        agent.device = checkpoint['device']
        
        agent.total_t = checkpoint['total_t']
        agent.train_t = checkpoint['train_t']
        
        agent.replay = ReplayBuffer(checkpoint['replay_size'])
        agent.replay.flag = checkpoint['replay_flag']
        agent.replay.index = checkpoint['replay_index']
        agent.replay.replay_buffer = np.load(replay_buffer_file, allow_pickle=True)
        
        agent.estimator = Estimator(features_shape=NUM_FEATURES)
        agent.estimator.load_state_dict(checkpoint['estimator'])
        agent.estimator.eval()
        
        agent.target_estimator = Estimator(features_shape=NUM_FEATURES)
        agent.target_estimator.load_state_dict(checkpoint['target_estimator'])
        agent.target_estimator.eval()
        
        agent.loss_value = checkpoint['loss']
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        
        agent.criterion = nn.MSELoss()
        
        return agent
        
    def save_checkpoint(self, filename=SAVE_PATH, replay_buffer_file='./src/replay.npy'):
        attr = {
            'estimator': self.estimator.state_dict(),
            'target_estimator': self.target_estimator.state_dict(),
            'total_t': self.total_t,
            'train_t': self.train_t,
            'update_target_freq': self.update_target_freq,
            'pretrain_steps':self.pretrain_steps,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'discount': self.discount,
            'batch_size': self.batch_size,
            'train_freq': self.train_freq,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer.state_dict(),
            'device': self.device,
            'save_path': self.save_path,
            'save_freq': self.save_freq,
            'loss': self.loss_value,
            'replay_flag': self.replay.flag,
            'replay_size': self.replay.replay_buffer_size,
            'replay_index': self.replay.index
        }
        torch.save(attr, filename)
        np.save(replay_buffer_file, self.replay.replay_buffer)

class Estimator(nn.Module):
    def __init__(self, features_shape=NUM_FEATURES, image_shape=[17,17]):
        super(Estimator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        
        self.fc1 = nn.Linear(features_shape, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        
        self.advantage_fc = nn.Linear(128, 64)
        self.advantage = nn.Linear(64, NUM_ACTIONS)
        
        self.value_fc = nn.Linear(128, 64)
        self.value = nn.Linear(64, 1)
        
    def forward(self, features, image):
        x = image.reshape(image.shape[:-1] + (1, 17, 17))
        
        x = nn.functional.elu(self.conv1(x))
        x = nn.functional.elu(self.conv2(x))
        x = nn.functional.elu(self.conv3(x))
        
        x = x.reshape(x.shape[:-3] + (128,))
        
        y = nn.functional.elu(self.fc1(features))
        y = nn.functional.elu(self.fc2(y))
        
        z = torch.cat((x, y), dim=-1)
        z = nn.functional.elu(self.fc3(z))
        z = nn.functional.elu(self.fc4(z))
        
        adavantage = nn.functional.elu(self.advantage_fc(z))
        adavantage = self.advantage(adavantage)
        
        value = nn.functional.elu(self.value_fc(z))
        value = self.value(value)
        return value + (adavantage - adavantage.mean(dim=-1, keepdim=True))

class ReplayBuffer(object):
    def __init__(self, replay_buffer_size):
        self.replay_buffer = np.empty(replay_buffer_size, dtype=object)
        self.replay_buffer_size = replay_buffer_size
        self.index = 0
        self.flag = 0
        
    def save(self, transition):
        self.replay_buffer[self.index] = transition
        self.index = (self.index + 1) % self.replay_buffer_size
        if self.index == 0:
            self.flag = 1
            
    def sample(self, batch_size):
        n = self.replay_buffer_size if self.flag else self.index
        indices = np.random.choice(n, size=batch_size, replace=True)
        return self.replay_buffer[indices]

def train(baselines, prob=None, episodes=5000, lr=0.001, batch_size=128):
    if os.path.isfile(SAVE_PATH) and os.access(SAVE_PATH, os.R_OK):
        agent = Agent.from_checkpoint(checkpoint=torch.load(SAVE_PATH))
    else:
        agent = Agent(learning_rate=lr, batch_size=batch_size, save_path=SAVE_PATH)
    
    total_wins = 0
    for episode in range(1, episodes + 1):
        player = DDDQNPlayer(agent=agent)
        config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
        config.register_player(name='p0', algorithm=player)
        opponent = np.random.choice(baselines, 1, p=prob)[0]
        config.register_player(name=f'p1', algorithm=opponent())
        
        game_result = start_poker(config, verbose=0)
        stack1 = [p['stack'] for p in game_result['players'] if p['name'] == 'p0'][0]
        stack2 = [p['stack'] for p in game_result['players'] if p['name'] == 'p1'][0]
        
        if stack1 > stack2:
            total_wins += 1
            
        history = player.get_history()
        
        for i in range(len(history)):
            agent.feed(tuple(history[i]))
            
        if episode % 100 == 0:
            print(f'episode {episode} done')
            print(f'Winning rate: {total_wins / episode}')
    
    agent.save_checkpoint(filename=SAVE_PATH)

baselines = [baseline1_ai, baseline2_ai, baseline3_ai, baseline4_ai, baseline5_ai, baseline6_ai]
prob = [0.15, 0.2, 0.2, 0.2, 0.15, 0.1]
train(baselines=baselines, prob=prob, episodes=1000)