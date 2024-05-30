from game.player import Player
from agents.baseline.random_agent import RandomAgent
from agents.baseline.manual_agent import ManualAgent
from game.buildings import CustomBuilding,CUSTOM_BUILDINGS

if __name__ == "__main__":
    a = CUSTOM_BUILDINGS[10]
    print(a)
    b = a.purchase('Me')
    print(a)
    print(b)