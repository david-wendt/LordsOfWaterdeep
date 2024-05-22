from game import game
from agents.baseline.manual_agent import ManualAgent
from agents.baseline.random_agent import RandomAgent

if __name__ == "__main__":
    agents = [ManualAgent(), RandomAgent()]
    game.main(agents)