from game.players import BasePokerPlayer
from .hole_card_est import HoleCardEstimator
import numpy as np
import torch
import torch.nn as nn
from os.path import dirname

SUITS = ('C', 'D', 'H', 'S')
RANKS = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
STREETS = ('preflop', 'flop', 'turn', 'river')
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

class Player(BasePokerPlayer):
    NUM_FEATURES = 10
    PATH = f'{dirname(__file__)}/parameters.pt'
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opponent_raise_count = 0
        self.opponent_move_count = 0
        self.street = np.zeros(4)
        self.hand_strength = 0
        self.estimator = Estimator(features_shape=self.NUM_FEATURES)
        self.estimator.load_state_dict(torch.load(self.PATH))
        self.estimator.to(self.device)
        self.estimator.eval()
        
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
        
        with torch.no_grad():
            q_values = self.estimator(features=features, image=image)
            masked_q_values = -np.inf * np.ones(NUM_ACTIONS)
            masked_q_values[legal_actions] = q_values[legal_actions]
            best_action = np.argmax(masked_q_values)
        
        amount = 0
        if best_action >= 2:
            min_amount = valid_actions[2]['amount']['min']
            max_amount = valid_actions[2]['amount']['max']
            amount = min_amount + int((max_amount - min_amount) * ((best_action - 2) / (NUM_ACTIONS - 2)))
            
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
        pass
    
class Estimator(nn.Module):
    def __init__(self, features_shape, image_shape=[17,17]):
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

def setup_ai():
    return Player()