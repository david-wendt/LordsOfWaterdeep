from agents.agent import Agent
import random 

class RandomAgent(Agent):
    def act(self, gameState, playerState, actions) -> int:
        return random.randint(0,len(actions) - 1)