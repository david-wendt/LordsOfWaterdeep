from random import shuffle
from game.game_info import DEFAULT_BUILDINGS, QUESTS, Quest

class BoardState():
    '''Class to represent the state of the board itself 
    along with cards, but not players.'''
    def __init__(self) -> None:
        '''
        Initialize the board state. Creates the quest stack
        and initializes all buildings states.
        '''
        # Create the quest stack
        self.questStack = QUESTS.copy()
        shuffle(self.questStack)

        # Initialize building occupation states.
        # Will be None when unoccupied, player.name when occupied 
        # with one of player's agents.
        self.buildingStates = {building: None for building in DEFAULT_BUILDINGS}

        # Initialize the four available quests at Cliffwatch Inn
        self.availableQuests = [self.drawQuest() for _ in range(4)]

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


def main():
    # Test the quest stack
    boardState = BoardState()
    for i in range(2):
        print("Drew quest:", boardState.drawQuest())
        boardState.printQuestStack()
        print()