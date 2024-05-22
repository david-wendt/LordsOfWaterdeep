''' TODO: Deep Q-network implementation '''

# TODO - config for DQN (or change gradeint file to not use config)

import numpy as np
import torch
import torch.nn as nn
from .network_utils import build_mlp, device, np2torch
import torch.optim as optim
import random
from collections import namedtuple
from collections import deque
import math

from agents.agent import Agent
from features import featurize

class DeepQNet(nn.Module):
    def __init__(self, observation_dim, n_actions, 
                 n_hidden_layers=2, hidden_layer_size=64,
                 discount_factor=1., replay_capacity=1000, batch_size=128):
        super(DeepQNet, self).__init__()

        self.policy_network = build_mlp(observation_dim, n_actions, n_hidden_layers, hidden_layer_size)
        # use this line periodically (every sum number of episodes)
        self.target_network = build_mlp(observation_dim, n_actions, n_hidden_layers, hidden_layer_size)
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def forward(self, x):
        return self.policy_network(x)
    
    def update_to_target(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
    
    


# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# for setting up replay memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'action_mask'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        

class DQNAgent(Agent):
    # can consider what other params to include
    def __init__(self, dqn: DeepQNet, eps_start, eps_decay, n_actions, learning_rate=0.001, replay_capacity=1000, batch_size=128, discount_factor=1):
        self.dqn = dqn 
        self.optimizer = optim.AdamW(self.dqn.policy_network.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_capacity)
        self.batch_size=batch_size

        self.eps = eps_start
        self.eps_decay = eps_decay
        self.n_actions = n_actions
        self.prev_score = 0.0
        self.prev_state = None
        self.prev_action = None

        self.discount_factor = discount_factor

        # to know how often to go to target
        self.episode = 0
        self.target_reset_freq = 1000

    # started from code from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html and adjusted
    def optimize_model(self):
        # get BATCH_SIZE
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.stack(batch.state) # should be cat?

        action_batch = torch.stack(batch.action) # should be cat?
        reward_batch = torch.stack(batch.reward).squeeze() # added .squeeze()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        #print("seeking issue")
        state_action_values = self.dqn.policy_network(state_batch).gather(1, action_batch)
        #print("seeking issue done")
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)

        with torch.no_grad():
            # TODO - need to mask with action_mask
            next_state_values[non_final_mask] = self.dqn.target_network(non_final_next_states).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch
        expected_state_action_values.unsqueeze(1)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.dqn.policy_network.parameters(), 100)
        self.optimizer.step()

    def end_game(self, score):
        reward = torch.tensor([score - self.prev_score], dtype=torch.float32)
        self.memory.push(self.prev_state, self.prev_action, None, reward, None)
        self.optimize_model()
        
        self.prev_state = None
        self.prev_action = None
    
    # epsilon greedy policy
    def act(self, gameState, playerState, actions, score):
        # at the end of the game method called end game
        # on every agent do a final update
        # at start of act, only do push if prev_state is not None else move on

        state_tensor, action_mask = featurize.featurize(gameState, playerState, actions)

        if self.prev_state is not None:
            reward = torch.tensor([score - self.prev_score], dtype=torch.float32)
            self.memory.push(self.prev_state, self.prev_action, state_tensor, reward, action_mask)
            self.optimize_model()

        self.prev_state = state_tensor
        self.prev_score = score

        self.episode += 1
        if self.episode % self.target_reset_freq == 0:
            self.dqn.update_to_target()

        # call featurize (state, list of actions)
        # gives features, binary encoding of available actions
        self.eps = self.eps * math.exp(-1. / self.eps_decay)
        sample = random.random()
        if sample > self.eps:
            with torch.no_grad():
                action = torch.argmax(self.dqn(state_tensor) + 1e10 * (action_mask - 1)) # mask with actions_mask
        else:
            action = torch.tensor(random.randrange(self.n_actions), dtype=torch.long)

        self.prev_action = action.unsqueeze(0) # add dim

        return 0
        return action.item()