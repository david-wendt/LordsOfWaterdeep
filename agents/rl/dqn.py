''' TODO: Deep Q-network implementation '''

import torch 
from torch import nn

from agents.agent import Agent

class DeepQNet(nn.Module):
    def __init__(self):
        super().__init__()
    
class DQNAgent(Agent):
    def __init__(self, dqn: DeepQNet):
        self.dqn = dqn 
    
    def act(self, state, actions) -> int:
        return torch.argmax(self.dqn(state)) # Very rough