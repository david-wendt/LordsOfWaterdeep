''' TODO: Write a class here to take user input to play the game. '''

from ..agent import Agent

class ManualAgent(Agent):
    def act(self, state, actions) -> int:
        print('Game state:\n', state)
        print('Actions available:\n')
        for i,action in enumerate(actions):
            print("\t",i,":",action)
        choice = ""
        while not 0 <= int(choice) < len(actions):
            choice = input("Choose an action (enter index)")
        return choice 