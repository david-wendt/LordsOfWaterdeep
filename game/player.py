from game.game_info import *

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

        self.resources = Resources()

        self.activeQuests = []
        self.completedQuests = []
        # self.completedPlotQuests = [] # Completed plot quests
        # self.intrigues = []
        self.agents = numAgents

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

    def getResource(self, resource: str, number: int):
        '''
        Receive some number of resources of the same type.

        Args: 
            resource: the type of resource.
            number: the number of that resource type received.
        '''
        if resource not in RESOURCES:
            raise ValueError("Invalid resource type.")
        if resource in ["Q", "I"]:
            raise ValueError("Cannot receive quests or intrigue cards with this function. \
                             Use 'receiveQuest' or 'receive Intrigue', respectively.")
        if number <= 0:
            raise ValueError("Cannot receive nonnegative resource count.")
        assert isinstance(number, int)
        
        self.resources[resource] += number

    # This may need to be uncommented if we add the plot quest 
    # or building that gives an extra agent
    # def getAgent(self):
    #     '''Receive an additional agent (for future use).'''
    #     self.maxAgents += 1
    #     self.agents += 1

    def returnAgents(self):
        '''Return all of this player's agents.'''
        self.agents = self.maxAgents
        
    def completeQuest(self, quest: Quest):
        # Make sure the agent has this quest
        if quest not in self.activeQuests:
            raise ValueError("This agent does not have this quest.")

        # Check if the quest can be completed
        validCompletion = True
        for resource in quest.requirements:
            if quest.requirements[resource] > self.resources[resource]:
                validCompletion = False
                
        if not validCompletion:
            raise ValueError("Do not have enough resources to complete this quest.")
        
        for resource,number in quest.requirements.items():
            self.resources[resource] -= number
        for resource,number in quest.rewards.items():
            if resource not in RESOURCES:
                raise ValueError("Invalid resource type.")
            elif resource == "Q":
                # TODO: Implement this. somehow access gamestate 
                raise Exception("Not yet implemented.")
            elif resource == "I":
                # TODO (later version): Implement this
                raise Exception("This is impossible! Intrigue cards do not exist yet!")
            else:
                self.resources[resource] += number

        # TODO (future): If plot quest, append to completed plot quests
        self.completedQuests.append(quest)
        self.activeQuests.remove(quest)

        
        # Check that all resource counts are still nonnegative
        for resourceNumber in self.resources.values():
            assert resourceNumber >= 0
    
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
        score = 0. 
        for resource,number in self.resources.items():
            if resource in ["Purple", "White"]:
                score += number 
            elif resource in ["Orange", "Black"]:
                score += number / 2.
            elif resource == "Gold":
                score += number / 4.
            elif resource == "VP":
                score += number * SCORE_PER_VP # TODO: Estimate value of a VP
            else:
                raise ValueError("Invalid resource type.")

        # Active quests
        score += len(self.activeQuests) / 2
        # TODO: Maybe increase slightly for lord-card-aligned quests? to 0.75 or smth?

        for quest in self.completedQuests:
            if quest.type in self.lordCard:
                score += 4 * SCORE_PER_VP
            # TODO (later version): add check for lordCard = "Buildings"

        return score 