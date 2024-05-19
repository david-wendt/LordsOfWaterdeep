from .game_info import *

# Player state class
class Player():
    def __init__(self, name: str, numAgents: int, 
                 lordCard: tuple[str]) -> None:
        '''
        Initialize the player's name, resources, agents, and VPs.
        
        Args:
            name: the player's name
            numAgents: the number of starting agents for the player
            lord: the lord card (i.e. secret identity) given to the player 
        '''
        self.name = name 
        self.lordCard = lordCard

        self.resources = Resources() # Inlcudes VPs!
        # NOT using Quests/Intrigues in here though

        self.activeQuests = []
        self.completedQuests = []
        # TODO (later): uncomment below
        # self.completedPlotQuests = [] # Completed plot quests 
        self.intrigues = []
        self.agents = numAgents
        self.maxAgents = numAgents # Done like this because player objects have no access to game state

    def __repr__(self):
        return f"Player `{self.name}`\n\tResources: {self.resources}"

    def getQuest(self, quest: Quest):
        '''
        Receive a quest.

        Args: 
            quest: the quest to receive.
        '''
        self.activeQuests.append(quest)

    # TODO (Later version): uncomment this
    # def getIntrigue(self, intrigue: Intrigue):
        # '''
        # Receive an intrigue card.

        # Args: 
        #     intrigue: the intrigue card to receive.
        # '''
    #     self.intrigues.append(intrigue)

    def getResources(self, resources: Resources):
        ''' Receive a resource bundle `resources` '''
        self.resources.clerics += resources.clerics
        self.resources.wizards += resources.wizards
        self.resources.rogues += resources.rogues
        self.resources.fighters += resources.fighters
        self.resources.gold += resources.gold
        self.resources.VPs += resources.VPs
        
        for _ in range(resources.intrigues):
            raise NotImplementedError # Draw intrigue 
        
        for _ in range(resources.quests):
            raise NotImplementedError # Draw quest 
    
    def removeResources(self, resources: Resources):
        if resources.intrigues != 0 or resources.quests != 0 or resources.VPs != 0:
            raise ValueError("Cannot remove intrigues or quests or VPs!")
        
        negResources = Resources(
            wizards= -resources.wizards,
            clerics= -resources.clerics,
            fighters= -resources.fighters,
            rogues= -resources.rogues,
            gold= -resources.gold,
        )

        self.getResources(negResources)

    def getAgent(self):
        '''Receive an additional agent (for future use).'''
        self.maxAgents += 1
        self.agents += 1

    def returnAgents(self):
        '''Return all of this player's agents.'''
        self.agents = self.maxAgents
        
    def isValidQuestCompletion(self, quest: Quest):
        return (
            quest.requirements.wizards <= self.resources.wizards and
            quest.requirements.clerics <= self.resources.clerics and
            quest.requirements.fighters <= self.resources.fighters and
            quest.requirements.rogues <= self.resources.rogues and
            quest.requirements.gold <= self.resources.gold 
        )
    
    def validateResources(self):
        return (
            0 <= self.resources.wizards and
            0 <= self.resources.clerics and
            0 <= self.resources.fighters and
            0 <= self.resources.rogues and
            0 <= self.resources.gold and 
            0 <= self.resources.VPs and 
            0 == self.resources.quests and 
            0 == self.resources.intrigues
        )

    def completeQuest(self, quest: Quest):
        # Make sure the agent has this quest
        if quest not in self.activeQuests:
            raise ValueError("This agent does not have this quest.")

        # Check if the quest can be completed                
        if not self.isValidQuestCompletion(quest):
            raise ValueError("Do not have enough resources to complete this quest.")
        
        self.removeResources(quest.requirements)
        self.getResources(quest.rewards)

        # TODO (future): If plot quest, append to completed plot quests
        self.completedQuests.append(quest)
        self.activeQuests.remove(quest)

        # Check that all resource counts are still nonnegative
        self.validateResources()
    
    def score(self):
        '''Compute an RL agent's score.
        
        I was intending on using this score function, rather 
        than an actual VP count, as the reward for the RL agent:
        reward(action) = score(after action) - score(before action)
        '''
        # The score used for training RL agents
        # should NOT be VP alone, but to account for 
        # endgame values of agents and gold, we should have
        # score = VP + #(agents) + #(gold)//2
        
        # More realistically though, if we 
        # are really trying to teach 
        # strategy, maybe #(white,purple) 
        # + #(black,orange)/2 + #(gold)/4 
        # to correspond to turn-value instead 
        # of VP-value at endgame? )
        # This version is implemented below.

        # TODO: Maybe include quests
        # TODO (later): Maybe include intrigues?
        # Maybe subtract sum/max/softmax of 
        #   the other player's scores? important
        #   to punish the agent for giving other players
        #   free stuff. Would need to code it up in a 
        #   non-circular way though.

        # 'Score' here represents number of turns' worth of resources acquired
        score = (
            self.resources.clerics
            + self.resources.wizards
            + self.resources.fighters / 2.
            + self.resources.rogues / 2.
            + self.resources.gold / 4.
            + self.resources.VPs * SCORE_PER_VP
        )

        # Active quests
        score += len(self.activeQuests) / 2
        # TODO: Maybe increase slightly for lord-card-aligned quests? to 0.75 or smth?

        for quest in self.completedQuests:
            if quest.type in self.lordCard:
                score += 4 * SCORE_PER_VP
            # TODO (later version): add check for lordCard = "Buildings"

        return score 