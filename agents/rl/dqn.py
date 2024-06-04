''' TODO: Deep Q-network implementation '''

# TODO - config for DQN (or change gradeint file to not use config)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple
from collections import deque
import math

from agents.agent import Agent
from features import featurize

DEVICE = 'cpu'
USE_GPU = False # Way faster on cpu than on apple silicon for me, idk why - DW
if USE_GPU:
    if torch.cuda.is_available():
        DEVICE = torch.cuda.current_device()
    elif torch.backends.mps.is_available(): # Run on apple silicon gpu (for M-series MacBooks)
        DEVICE = 'mps'

ACTIVATION_FNS = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU()
}

class DeepQNet(nn.Module):
    def __init__(self, 
            input_dim, 
            output_dim, 
            hidden_layer_sizes=[256, 128],
            layernorm='layernorm',
            activation='LeakyReLU'
        ):

        super(DeepQNet, self).__init__()

        print("INITIALIZING DEEP Q NETWORK")
        print("\tInput size (state dim):", input_dim)
        print("\tOutput size (action dim):", output_dim)
        print("\tHidden layer sizes:", hidden_layer_sizes)
        print() # Newline

        layer_sizes = hidden_layer_sizes + [output_dim]
        n_layers = len(layer_sizes)

        layers = [nn.Linear(input_dim, hidden_layer_sizes[0])]
        activation_fn = ACTIVATION_FNS[activation]

        for i in range(n_layers-1):
            if layernorm == 'layernorm' or layernorm is True:
                layers.append(nn.LayerNorm(layer_sizes[i]))
            elif layernorm == 'batchnorm':
                layers.append(nn.BatchNorm1d(layer_sizes[i]))
            else:
                if layernorm is not None:
                    raise ValueError(f'Unknown layer norm: {layernorm}')
                
            layers.append(activation_fn)
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


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
    def __init__(
        self, 
        state_dim=featurize.STATE_DIM, 
        action_dim=featurize.ACTION_DIM, 
        hidden_layer_sizes=[256, 128],
        layernorm='layernorm',
        activation='LeakyReLU',
        eps_start=0.5, 
        eps_decay=0.99, 
        learning_rate=0.001, 
        replay_capacity=1000, 
        batch_size=128, 
        discount_factor=1
    ):
        super().__init__()

        # TODO: Package DQN params in a dict that gets passed in after being read from a config
        self.q_net = DeepQNet(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            layernorm=layernorm,
            activation=activation
        ).to(DEVICE)

        self.target_net = DeepQNet(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            layernorm=layernorm,
            activation=activation
        ).to(DEVICE)

        self.update_target()

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_capacity)
        self.batch_size=batch_size

        self.eps = eps_start
        self.eps_decay = eps_decay
        self.action_dim = action_dim
        self.prev_score = 0.0
        self.prev_state = None
        self.prev_action = None

        self.discount_factor = discount_factor

        # to know how often to go to target
        self.episode = 0
        self.target_reset_freq = 1000

    def train(self):
        super().train()
        self.q_net.train()

    def eval(self):
        super().eval()
        self.q_net.eval()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.requires_grad_(False) # Freeze the target net
        self.target_net.eval()

    # started from code from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html and adjusted
    def optimize_model(self):
        if not self.trainMode:
            return
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
                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None]).to(DEVICE)
        state_batch = torch.stack(batch.state).to(DEVICE) # should be cat?

        action_batch = torch.stack(batch.action).to(DEVICE) # should be cat?
        reward_batch = torch.stack(batch.reward).squeeze().to(DEVICE) # added .squeeze()
        action_mask_batch = torch.stack(batch.action_mask).to(DEVICE)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        #print("seeking issue")
        state_action_values = self.q_net(state_batch).gather(1, action_batch)
        #print("seeking issue done")
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=DEVICE)

        with torch.no_grad():
            next_state_values[non_final_mask] = (self.target_net(non_final_next_states) + 1e10 * (action_mask_batch[non_final_mask] - 1)).max(1).values


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
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()

    def end_game(self, score):
        reward = torch.tensor([score - self.prev_score], dtype=torch.float32).to(DEVICE)
        self.memory.push(self.prev_state, self.prev_action, None, reward, torch.zeros(self.action_dim))
        self.optimize_model()
        
        self.prev_state = None
        self.prev_action = None
    
    # epsilon greedy policy
    def act(self, gameState, playerState, actions, score):
        # at the end of the game method called end game
        # on every agent do a final update
        # at start of act, only do push if prev_state is not None else move on

        state_tensor, action_mask = featurize.featurize(gameState, playerState, actions)
        state_tensor = state_tensor.to(DEVICE)
        action_mask = action_mask.to(DEVICE)
        assert action_mask.sum() == len(actions)

        if self.prev_state is not None:
            reward = torch.tensor([score - self.prev_score], dtype=torch.float32).to(DEVICE)
            self.memory.push(self.prev_state, self.prev_action, state_tensor, reward, action_mask)
            self.optimize_model()

        self.prev_state = state_tensor
        self.prev_score = score

        self.episode += 1
        if self.episode % self.target_reset_freq == 0:
            self.update_target()

        if self.trainMode:
            # call featurize (state, list of actions)
            # gives features, binary encoding of available actions
            # self.eps = self.eps * math.exp(-1. / self.eps_decay) # Old version
            self.eps *= self.eps_decay # I think this is conceptually simpler - DW


        available_action_indices = np.arange(self.action_dim)[action_mask.cpu().bool()].tolist()
        sample = random.random()
        if self.trainMode and sample < self.eps:
            # Random action
            available_action_idx = torch.randint(0,len(actions),(1,)).item()
            action_idx = torch.tensor(available_action_indices[available_action_idx]).to(DEVICE)
        else:
            # Q argmax action (always take in eval mode)
            with torch.no_grad():
                action_idx = torch.argmax(self.q_net(state_tensor) + 1e10 * (action_mask - 1)) # mask with actions_mask
                available_action_idx = available_action_indices.index(action_idx)

        self.prev_action = action_idx.unsqueeze(0) # add dim

        return available_action_idx