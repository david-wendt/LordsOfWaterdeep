from agents.rl.dqn import DeepQNet, DQNAgent
from agents.baseline.random_agent import RandomAgent
from game.game import GameState

def train(agents, n_games):
    for igame in range(n_games):
        game = GameState(num)
    
def main():
    deepQAgent = DQNAgent(
        DeepQNet(...),
        eps_start, # Maybe set some defaults for these?
        eps_end,
        eps_decay,
        n_actions
    )

    randomAgent = RandomAgent()

    agents = [deepQAgent, randomAgent]
