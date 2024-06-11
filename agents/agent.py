from game.quests import Quest, DO_NOT_COMPLETE_QUEST
from game.buildings import CustomBuilding
from game.intrigues import INTRIGUES

class Agent():
    def __init__(self) -> None:
        self.plotQuestsTaken = 0
        self.plotQuestsCompleted = 0
        self.lordQuestsTaken = 0
        self.lordQuestsCompleted = 0
        self.questsTaken = 0
        self.questsCompleted = 0
        self.buildingsPurchased = 0
        self.intrigueCardsPlayed = 0 
        self.trainMode = False

    def train(self):
        self.trainMode = True
    
    def eval(self):
        self.trainMode = False 
        
    def getStats(self, ngames):
        return {
            'Plot Quests Taken Frac': self.plotQuestsTaken / self.questsTaken,
            'Plot Quests Completed Frac': self.plotQuestsCompleted / self.questsCompleted,
            'Lord Quests Taken Frac': self.lordQuestsTaken / self.questsTaken,
            'Lord Quests Completed Frac': self.lordQuestsCompleted / self.questsCompleted,
            'Buildings Purchased per Game': self.buildingsPurchased / ngames,
            'Intrigue Cards Played per Game': self.intrigueCardsPlayed / ngames,
            'Quests Taken per Game': self.questsTaken / ngames,
            'Quests Completed per Game': self.questsCompleted / ngames
        }
    
    def isLordQuest(self, quest, player):
        assert isinstance(quest, Quest)
        return quest.type in player.lordCard

    def tookQuest(self, action, actions):
        return (isinstance(action, Quest) and actions[0] != DO_NOT_COMPLETE_QUEST)
    
    def tookPlotQuest(self, action, actions):
        return self.tookQuest(action, actions) and action.plot
    
    def tookLordQuest(self, action, actions, player):
        return self.tookQuest(action, actions) and self.isLordQuest(action, player)
    
    def completedQuest(self, action, actions):
        return (isinstance(action, Quest) and actions[0] == DO_NOT_COMPLETE_QUEST)
    
    def completedPlotQuest(self, action, actions):
        return self.completedQuest(action, actions) and action.plot
    
    def completedLordQuest(self, action, actions, player):
        return self.completedQuest(action, actions) and self.isLordQuest(action, player)
    
    def purchasedBuilding(self, action):
        return isinstance(action, CustomBuilding) and action.owner is None
    
    def playedIntrigueCard(self, action):
        return isinstance(action, str) and action in INTRIGUES
    
    def actWrapper(self, state, playerState, actions, score):
        action_idx = self.act(state, playerState, actions, score)
        action = actions[action_idx]
        # print('actions:', actions)
        # print('action taken:', )
        if not self.trainMode:
            if self.tookQuest(action, actions):
                self.questsTaken += 1
            if self.tookPlotQuest(action, actions):
                self.plotQuestsTaken += 1
            if self.tookLordQuest(action, actions, playerState):
                self.lordQuestsTaken += 1
            if self.completedQuest(action, actions):
                self.questsCompleted += 1
            if self.completedPlotQuest(action, actions):
                self.plotQuestsCompleted += 1
            if self.completedLordQuest(action, actions, playerState):
                self.lordQuestsCompleted += 1
            if self.purchasedBuilding(action):
                self.buildingsPurchased += 1
            if self.playedIntrigueCard(action):
                self.intrigueCardsPlayed += 1
            
        return action_idx

    def act(self, state, playerState, actions, score) -> int:
        ''' Override this in subclasses'''
        raise NotImplementedError
    
    def end_game(self, score):
        return