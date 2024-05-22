''' TODO: Deep Q-network implementation '''

# TODO - config for DQN (or change gradeint file to not use config)

import numpy as np
import torch
import torch.nn as nn
from network_utils import build_mlp, device, np2torch
import torch.optim as optim
import random
from collections import namedtuple
from collections import deque
from agents.agent import Agent
import math

class DeepQNet(nn.Module):
    def __init__(self, observation_dim, n_actions, 
                 n_hidden_layers=2, hidden_layer_size=64,
                 learning_rate=0.001, discount_factor=1., replay_capacity=1000, batch_size=128):
        super(DeepQNet, self).__init__()

        self.discount_factor = discount_factor

        self.batch_size = batch_size

        # TODO - decide between passing in params and config
        self.policy_network = build_mlp(observation_dim, n_actions, n_hidden_layers, hidden_layer_size)

        # use this line periodically (every sum number of episodes)
        self.target_network = self.policy_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = optim.Adam(self.mlp.parameters(), lr=learning_rate)

        # not sure if this should be a member here
        self.memory = ReplayMemory(replay_capacity)

    def forward(self, x):
        return self.policy_network(x)
    
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
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# for setting up replay memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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
    def __init__(self, dqn: DeepQNet, eps_start, eps_end, eps_decay, n_actions):
        self.dqn = dqn 
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.n_actions = n_actions
    
    # epsilon greedy policy
    def act(self, state, actions):
        # call featurize (state, list of actions)
        # gives features, binary encoding of available actions
        self.eps = self.eps_end + self.eps * math.exp(-1. / self.eps_decay)
        sample = random.random()
        if sample > self.eps:
            with torch.no_grad():
                return self.dqn(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)
        
"""
here is the framework I think for training with this memory thing -- need to fill details
for game in range(NUM_GAMES): # where to specify
    # set up game
    while(game):
        action = agent.act(...)
        next_state, reward, done, _ = ...
        reward = torch.tensor([reward], dtype=torch.float32)
        if done:
            next_state = None
        else:
            next_state = torch.tensor([next_state], dtype=torch.float32)

        ().memory.push(state, action, next_state, reward)

        state = next_state

        # how often do we call this - is it every time
        ().optimize_model(...)
        if done:
            break

        if episode % HOW_OFTEN_TO_UPDATE_TARGET == 0:
            ().target_net.load_state_dict(().policy_net.state_dict())
"""
