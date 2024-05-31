from game.game_info import *
from agents.agent import Agent
from agents.baseline.manual_agent import ManualAgent

# Player state class
class Player():
    def __init__(self, name: str, 
                 agent: Agent,
                 numAgents: int, 
                 lordCard: tuple[str]) -> None:
        '''
        Initialize the player's name, resources, agents, and VPs.
        
        Args:
            name: the player's name
            numAgents: the number of starting agents for the player
            lord: the lord card (i.e. secret identity) given to the player 
        '''
        self.name = name 
        self.agent = agent
        self.lordCard = lordCard

        self.resources = Resources() # Inlcudes VPs!
        # NOT using Quests/Intrigues in here though

        self.activeQuests = []
        self.completedQuests = []
        # TODO (later): uncomment below
        self.completedPlotQuests = [0] * len(QUEST_TYPES) # N completed plot quests per type 
        self.intrigues = []
        self.agents = numAgents
        self.maxAgents = numAgents # Done like this because player objects have no access to game state

        self.hasCastle = False # for Castle Waterdeep

    def _base_repr(self):
        res = f"\nPlayer `{self.name}`\n\t"
        res += f"VPs: {self.resources.VPs}\n\t"
        res += f"Agents remaining: {self.agents}\n\tResources: {self.resources}"
        if self.hasCastle:
            res += "\n\tCastle Waterdeep: Owned"
        return res
        
    def __repr__(self):
        questTypes = ", ".join([quest.type for quest in self.activeQuests])
        res = self._base_repr()
        res += f"\n\tIngrigues: {len(self.intrigues)}"
        res += f"\n\tQuests: {questTypes}" # NOTE: Full quest info
        # should be public, but I thought it would be too clunky for this repr
        return res
    
    def _private_repr(self):
        activeQuests = ""
        for quest in self.activeQuests:
            activeQuests += "\n" + str(quest)
            
        completedQuestNames = [
            f"{quest.name} ({quest.type})" 
            for quest in self.completedQuests
        ]
        return self.__repr__() \
            + f"\n\tSecret Identity: {self.lordCard}" \
            + f"\n\tIntrigues: {self.intrigues}" \
            + f"\n\tActive Quests: {activeQuests}" \
            + f"\n\tCompleted Quests: {completedQuestNames}"

    def getQuest(self, quest: Quest):
        '''
        Receive a quest.

        Args: 
            quest: the quest to receive.
        '''
        self.activeQuests.append(quest)


    def getIntrigue(self, intrigue: str):
        '''
        Receive an intrigue card.

        Args: 
            intrigue: the intrigue card to receive.
        '''
        self.intrigues.append(intrigue)

    def getResources(self, resources: Resources):
        ''' Receive a resource bundle `resources` '''
        if isinstance(resources, FixedResources):
            raise TypeError("Player can only get Resources, not FixedResources!")
        self.resources.clerics += resources.clerics
        self.resources.wizards += resources.wizards
        self.resources.rogues += resources.rogues
        self.resources.fighters += resources.fighters
        self.resources.gold += resources.gold
        self.resources.VPs += resources.VPs
        
    def removeResources(self, resources: Resources):
        if resources.VPs != 0:
            raise ValueError("Cannot remove VPs!")
        
        negResources = Resources(
            wizards= -resources.wizards,
            clerics= -resources.clerics,
            fighters= -resources.fighters,
            rogues= -resources.rogues,
            gold= -resources.gold,
        )

        self.getResources(negResources)

        # Check that all resource counts are still nonnegative
        self.validateResources()

    def removeAllResources(self):
        ''' Remove all resources except for VPs '''
        self.resources.clerics  = 0
        self.resources.wizards  = 0
        self.resources.rogues   = 0
        self.resources.fighters = 0
        self.resources.gold     = 0

    def getAgent(self):
        '''Receive an additional agent (for future use).'''
        self.maxAgents += 1
        self.agents += 1

    def returnAgents(self):
        '''Return all of this player's agents.'''
        self.agents = self.maxAgents
        
    def canRemoveResources(self, resources: Resources):
        return (
            resources.wizards <= self.resources.wizards and
            resources.clerics <= self.resources.clerics and
            resources.fighters <= self.resources.fighters and
            resources.rogues <= self.resources.rogues and
            resources.gold <= self.resources.gold 
        )

    def isValidQuestCompletion(self, quest: Quest):
        return self.canRemoveResources(quest.requirements)
    
    def completableQuests(self):
        return [quest for quest in self.activeQuests if self.isValidQuestCompletion(quest)]
    
    def validateResources(self):
        assert (
            0 <= self.resources.wizards and
            0 <= self.resources.clerics and
            0 <= self.resources.fighters and
            0 <= self.resources.rogues and
            0 <= self.resources.gold and 
            0 <= self.resources.VPs
        ),"Some resources are negative! " + str(self.resources)
    
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

        # TODO: 
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

        # Intrigues 
        score += len(self.intrigues) / 2.

        # Castle Waterdeep
        score += self.hasCastle / 2.

        # Active quests
        score += len(self.activeQuests) / 2.
        # TODO: Maybe increase slightly for lord-card-aligned quests? to 0.75 or smth?

        for quest in self.completedQuests:
            if quest.type in self.lordCard:
                score += 4 * SCORE_PER_VP
            # TODO (later version): add check for lordCard = "Buildings"

        return score 

    def selectMove(self, gameState, actions):
        if isinstance(self.agent, ManualAgent):
            print("\n\nCURRENT PLAYER:", self.name, "(manual agent) must select a move.")
            
        score = self.score()
        return self.agent.act(gameState, self, actions, score)
    
    def convertResourcesToVPs(self):
        self.resources.VPs += (
            + self.resources.clerics
            + self.resources.wizards
            + self.resources.fighters
            + self.resources.rogues
            + self.resources.gold // 2
        )
        self.removeAllResources()

    def lordCardToVPs(self):
        for quest in self.completedQuests:
            if quest.type in self.lordCard:
                self.resources.VPs += 4

    def clear(self):
        self.activeQuests = []
        self.completedQuests = []
        self.intrigues = []
        self.hasCastle = False 

    def endGame(self): 
        # Final score to eval
        score = self.score()

        # End game VPs
        self.convertResourcesToVPs()
        self.lordCardToVPs()
        self.clear()

        # TODO (later): once we implement more stuff, call score again here and check it is the same
        finalScore = self.resources.VPs * SCORE_PER_VP
        self.agent.end_game(finalScore)
        return score