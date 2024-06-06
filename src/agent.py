from game.players import BasePokerPlayer
from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn

from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai

class Player(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

class Agent(nn.modules):
    
# DQN, Deep Q-Network, using Reinforcement Learning
def train(env, agent, baselines, episodes=1000, lr=0.001, batch_size=200, epsilon=1.0):
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    criterion = nn.MSELoss()
    replay = deque()
    
    for episode in episodes:
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        epsiode_reward = 0
        
        while not done:
            qval = agent(state)
            if random.random() < epsilon:
                action_idx = np.random.randint(0, len(action_set))
            else:
                action_idx = np.argmax(qval.data.numpy())
            action = action_set[action_idx]

            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            epsiode_reward += reward
            
            replay.append((state, action, reward, next_state, done))
            
            state = next_state
            
            if len(replay) >= batch_size:
                miniBatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in miniBatch])  
                state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in miniBatch])   
                action_batch = torch.Tensor([a for (s1,a,r,s2,d) in miniBatch])
                reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in miniBatch])
                done_batch = torch.Tensor([d for (s1,a,r,s2,d) in miniBatch])
                
                qvals = agent(state1_batch)
                next_qvals = agent(state2_batch).detach()
                max_next_qval = torch.max(next_qvals, dim=1)[0]
                targets = reward_batch + (1 - done_batch) * max_next_qval
                
                actoin_qvals = qvals.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                loss = criterion(actoin_qvals, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        if episode % 100 == 0:
            print(f'Episode {episode}: Reward={epsiode_reward}')

def setup_ai():
    return Player()