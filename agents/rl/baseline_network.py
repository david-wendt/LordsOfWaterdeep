import numpy as np
import torch
import torch.nn as nn
from network_utils import build_mlp, device, np2torch


class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network
    """

    def __init__(self, env, config):
        """
        Creates self.network creates self.optimizer to
        optimize its parameters.
        """
        super().__init__()
        self.config = config
        self.env = env
        self.baseline = None
        self.lr = self.config.learning_rate
        observation_dim = self.env.observation_space.shape[0]

        # TODO - output dimension
        self.network = build_mlp(observation_dim, 1, self.config.n_layers, self.config.hidden_layer_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]
        """
        output = self.network(observations).squeeze(dim=1)

        # TODO - check this line if output should be different
        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """
        observations = np2torch(observations)
        baseline = self.forward(observations).detach().numpy()
        advantages = returns - baseline
        return advantages

    def update_baseline(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size], containing all discounted
                future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        """
        returns = np2torch(returns)
        observations = np2torch(observations)
        
        predicted = self.forward(observations)
    
        # compute the loss (MSE)
        loss = torch.nn.functional.mse_loss(predicted, returns)
    
        # backpropagate
        self.optimizer.zero_grad()
        loss.backward()
    
        # step self.optimizer once
        self.optimizer.step()
