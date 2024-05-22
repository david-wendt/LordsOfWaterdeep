from agents.agent import Agent
from printing_utils import header
import random 

class ManualAgent(Agent):
    def act(self, gameState, playerState, actions) -> int:
        print(header('GAME STATE') + f'\n{gameState}\n')
        print(header('PLAYER STATE') + f'\n{playerState._private_repr()}\n')
        print(header('ACTIONS AVAILABLE'))
        for i,action in enumerate(actions):
            print("\t",i,":",action)
        choice = ""
        while not choice.isdigit() or not 0 <= int(choice) < len(actions):
            choice = input("Choose an action (enter index): ")
            if choice == "": return random.randint(0,len(actions) - 1) # Random if empty
        return int(choice)
    def end_game(self, score):
        return
