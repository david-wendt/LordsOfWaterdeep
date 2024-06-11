import numpy as np
import torch
import torch.nn as nn


class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network
    """

    def __init__(self, state_dim):
        """
        Creates self.network creates self.optimizer to
        optimize its parameters.
        """
        super().__init__()
        self.baseline = None
        self.lr = 0.001
        observation_dim = state_dim

        layers = [nn.Linear(observation_dim, 256), nn.ReLU()]
        layers.extend([nn.Linear(256, 128), nn.ReLU()])
        layers.append(nn.Linear(128, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]
        """
        output = self.network(observations).squeeze(dim=1)

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