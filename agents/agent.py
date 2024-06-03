from game.quests import Quest, DO_NOT_COMPLETE_QUEST
from game.buildings import CustomBuilding
from game.intrigues import INTRIGUES

class Agent():
    def __init__(self) -> None:
        self.plotQuestsTaken = 0
        self.plotQuestsCompleted = 0
        self.buildingsPurchased = 0
        self.intrigueCardsPlayed = 0 
        self.trainMode = False

    def train(self):
        self.trainMode = True
    
    def eval(self):
        self.trainMode = False 

    def tookPlotQuest(self, action, actions):
        return (isinstance(action, Quest) and action.plot and actions[0] != DO_NOT_COMPLETE_QUEST)
    
    def completedPlotQuest(self, action, actions):
        return (isinstance(action, Quest) and action.plot and actions[0] == DO_NOT_COMPLETE_QUEST)
    
    def purchasedBuilding(self, action):
        return isinstance(action, CustomBuilding) and action.owner is None
    
    def playedIntrigueCard(self, action):
        return isinstance(action, str) and action in INTRIGUES
    
    def actWrapper(self, state, playerState, actions, score):
        action = self.act(state, playerState, actions, score)
        if not self.trainMode:
            if self.tookPlotQuest(action, actions):
                self.plotQuestsTaken += 1
            if self.completedPlotQuest(action, actions):
                self.plotQuestsCompleted += 1
            if self.purchasedBuilding(action):
                self.buildingsPurchased += 1
            if self.playedIntrigueCard(action):
                self.intrigueCardsPlayed += 1
            
        return action

    def act(self, state, playerState, actions, score) -> int:
        ''' Override this in subclasses'''
        raise NotImplementedError
    
    def end_game(self, score):
        return