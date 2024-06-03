from game.player import Player
from agents.baseline.random_agent import RandomAgent
from agents.baseline.manual_agent import ManualAgent
from game.buildings import CustomBuilding,CUSTOM_BUILDINGS
from game import game 
from agents.baseline import strategic_agent

if __name__ == "__main__":
    strategic_agent.main()