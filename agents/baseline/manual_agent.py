from agents.agent import Agent

class ManualAgent(Agent):
    def act(self, state, actions) -> int:
        print('Game state:\n', state.displayGame())
        print('Actions available:\n')
        for i,action in enumerate(actions):
            print("\t",i,":",action)
        choice = "-1"
        while not 0 <= int(choice) < len(actions):
            choice = input("Choose an action (enter index): ")
        return int(choice)