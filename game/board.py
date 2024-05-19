from random import shuffle
from .game_info import *

class BoardState():
    '''Class to represent the state of the board itself 
    along with cards, but not players.'''
    def __init__(self) -> None:
        '''
        Initialize the board state. Creates the quest stack
        and initializes all buildings states.
        '''
        # Create the quest stack
        self.questStack = list(QUESTS).copy()
        shuffle(self.questStack)

        self.intrigueStack = list(INTRIGUES)

        # Initialize building occupation states.
        # Will be None when unoccupied, player.name when occupied 
        # with one of player's agents.
        self.buildingStates = {building: None for building in DEFAULT_BUILDINGS}

        # Initialize the four available quests at Cliffwatch Inn
        self.availableQuests = [self.drawQuest() for _ in range(4)]

    def __repr__(self):
        res = "Buildings:\n"
        for building,occupier in self.buildingStates.items():
            res += f"\t{building}:\t"
            if occupier is None:
                res += "unoccupied.\n"
            else:
                res += f"{occupier}.\n"

        res += "Quests (at Cliffwatch Inn):\n"
        for quest in self.availableQuests:
            res += f"\t{quest}\n"

        return res 

    def clearBuildings(self):
        '''Clears all buildings to their unoccupied states.'''
        for building in self.buildingStates:
            self.buildingStates[building] = None
    
    def drawQuest(self) -> Quest:
        '''
        Draw the top quest from the quest stack,
        removing it from the stack in the process.
        
        Returns: 
            The top quest from the quest stack.
        '''
        return self.questStack.pop()
    
    def drawIntrigue(self) -> str: # TODO: Change return type depending on intrigue card implementation
        '''
        Draw the top quest from the intrigue card stack,
        removing it from the stack in the process.
        
        Returns: 
            The top intrigue card from the intrigue card stack.
        '''
        return self.intrigueStack.pop()
    
    def printQuestStack(self) -> None:
        '''Debug function for printing the quest stack.'''
        print("Quest stack (top first):")
        questStackCopy = self.questStack.copy()
        for i in range(len(self.questStack)):
            print(i+1, questStackCopy.pop())

    def occupyBuilding(self, building: str, playerName: str):
        '''Change the occupation state of building from 'None'
        to being occupied by the player named playerName.'''
        self.buildingStates[building] = playerName

    def chooseQuest(self, quest_idx):
        quest = self.availableQuests[quest_idx]
        self.availableQuests[quest_idx] = self.drawQuest()
        return quest 

def main():
    # Test the quest stack
    boardState = BoardState()
    for i in range(2):
        print("Drew quest:", boardState.drawQuest())
        boardState.printQuestStack()
        print()