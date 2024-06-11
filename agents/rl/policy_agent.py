import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.rl.baseline_network import BaselineNetwork
import random
from collections import namedtuple
from collections import deque

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

class PolicyNet(nn.Module):
    def __init__(self, 
            input_dim, 
            output_dim, 
            hidden_layer_sizes=[512, 256, 128],
            layernorm='layernorm',
            activation='LeakyReLU'
        ):

        super(PolicyNet, self).__init__()

        print("INITIALIZING Policy NETWORK")
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

class PolicyAgent(Agent):
    # Note: Vertical code like this is nice and clean and readable - DW
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_layer_sizes=[512, 512, 256, 128],
        layernorm='layernorm',
        activation='LeakyReLU', 
        learning_rate=0.001, 
        batch_size=2000, 
        discount_factor=1
    ):

        # TODO: Package params in a dict that gets passed in after being read from a config
        self.policy_net = PolicyNet(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            layernorm=layernorm,
            activation=activation
        ).to(DEVICE)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)
        self.batch_size=batch_size

        self.action_dim = action_dim
        self.prev_score = 0.0
        self.prev_state = None
        self.prev_action = None
        self.discount_factor = discount_factor

        self.train_episode = 0
        self.t = 0
        self.rewards = []
        self.episode_rewards = []
        self.episode_reward = 0
        self.states = []
        self.actions = []
        self.paths = []
        self.action_masks = []
        self.log_probs = []

        self.normalize_advantage = False
        self.use_baseline = True
        self.baseline_network = BaselineNetwork(state_dim)

    def train(self):
        super().train()
        self.policy_net.train()

    def eval(self):
        super().eval()
        self.policy_net.eval()

    def end_game(self, score):
        reward = score - self.prev_score
        self.rewards.append(reward)
        self.prev_state = None
        self.prev_action = None
        self.episode_reward += reward
        if self.trainMode:
            path = {
                "observation": np.array(self.states),
                "reward": np.array(self.rewards),
                "action": np.array(self.actions),
                "action_mask" : np.array(self.action_masks),
                "log_prob" : np.array(self.log_probs)
            }

            self.paths.append(path)
            self.train_episode += 1
            if self.t >= self.batch_size:
                self.t = 0
                self.update_policy()
                self.paths = []

        self.states, self.rewards, self.actions, self.log_probs, self.action_masks = [], [], [], [], []
        self.episode_reward = 0
        self.prev_score = 0
    
    def act(self, gameState, playerState, actions, score):
        state_tensor, action_mask = featurize.featurize(gameState, playerState, actions)
        state_tensor = state_tensor.to(DEVICE)
        action_mask = action_mask.to(DEVICE)

        assert action_mask.sum() == len(actions)
        
        # maybe put into own function
        logits = self.policy_net(state_tensor) + 1e10 * (action_mask - 1)
        probabilities = torch.softmax(logits, dim=-1)
        distribution = torch.distributions.Categorical(probabilities)
        action_idx = distribution.sample()

        # should probably still check that we got a valid action
        available_action_indices = np.arange(self.action_dim)[action_mask.cpu().bool()].tolist()
        available_action_idx = available_action_indices.index(action_idx)

       
        log_prob = distribution.log_prob(action_idx)

        action_idx = action_idx.numpy()
        self.actions.append(action_idx)
        # find a way to use
        log_prob = log_prob.detach().numpy()
        self.log_probs.append(log_prob)

        self.states.append(state_tensor)
        self.action_masks.append(action_mask)
        
        if self.prev_state is not None:
            reward = score - self.prev_score
            self.rewards.append(reward)
            self.episode_reward += reward
                
        self.t += 1

        self.prev_action = 0
        self.prev_state = 1
        self.prev_score = score

        return available_action_idx
    
    def get_returns(self):
        """
        Calculate the returns G_t for each timestep

        Args:
            paths: recorded sample paths. See sample_path() for details.

        Return:
            returns: return G_t for each timestep

        After acting in the environment, we record the observations, actions, and
        rewards. To get the advantages that we need for the policy update, we have
        to convert the rewards into returns, G_t, which are themselves an estimate
        of Q^π (s_t, a_t):

           G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

        where T is the last timestep of the episode.

        Note that here we are creating a list of returns for each path
        """
        all_returns = []
        for path in self.paths:
            rewards = path["reward"]

            numTimesteps = len(rewards)
            discounted_reward = 0
            returns = [0] * numTimesteps

            for t in range(numTimesteps - 1, -1, -1):
                discounted_reward *= self.discount_factor
                discounted_reward += rewards[t]
                returns[t] = discounted_reward
            
            all_returns.append(returns)
        returns = np.concatenate(all_returns)

        return returns

    def calculate_advantage(self, returns, observations):
        """
        Calculates the advantage for each of the observations
        Args:
            returns: np.array of shape [batch size]
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """
        if self.use_baseline:
            # override the behavior of advantage by subtracting baseline
            advantages = self.baseline_network.calculate_advantage(
                returns, observations
            )
        else:
            advantages = returns

        if self.normalize_advantage:
           advantages = (advantages - np.mean(advantages)) / np.std(advantages)

        return advantages

    def update_policy(self):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size]

        Perform one update on the policy using the provided data.
        To compute the loss, you will need the log probabilities of the actions
        given the observations. Note that the policy's action_distribution
        method returns an instance of a subclass of
        torch.distributions.Distribution, and that object can be used to
        compute log probabilities.
        See https://pytorch.org/docs/stable/distributions.html#distribution

        Note:
        PyTorch optimizers will try to minimize the loss you compute, but you
        want to maximize the policy's performance.
        """
        observations = np.concatenate([path["observation"] for path in self.paths])
        actions = np.concatenate([path["action"] for path in self.paths])
        rewards = np.concatenate([path["reward"] for path in self.paths])
        returns = self.get_returns()
        action_masks = np.concatenate([path["action_mask"] for path in self.paths])

        advantages = self.calculate_advantage(returns, observations)
        # print("advantages", advantages)

        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        action_masks = np2torch(action_masks)

        # involve action_mask
        logits = self.policy_net(observations) + 1e10 * (action_masks - 1)
        probabilities = torch.softmax(logits, dim=-1)
        distribution = torch.distributions.Categorical(probabilities)
        # log probabilities of the actions given the observations (should change to weight allopwable actions)
        log_probs = distribution.log_prob(actions)
        # log_probs = np2torch(np.concatenate([path["log_prob"] for path in self.paths]))


        # print(log_probs)
        # print(advantages)

        # negative because we will minimize loss
        loss = -torch.mean(log_probs * advantages)
        # print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x