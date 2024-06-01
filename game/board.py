from random import shuffle
from game.game_info import *

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

        self.intrigueStack = INTRIGUES * 5
        shuffle(self.intrigueStack)

        # Initialize building occupation states.
        # Will be None when unoccupied, player.name when occupied 
        # with one of player's agents.
        self.buildings = {building: None for building in DEFAULT_BUILDINGS}
        self.customBuildings = [] # Needed to keep track of ordering of custom buildings

        # Initialize the four available quests at Cliffwatch Inn
        self.availableQuests = [self.drawQuest() for _ in range(NUM_CLIFFWATCH_QUESTS)]

        # Initialize the three available buildings at Builder's Hall
        self.buildingStack = list(CUSTOM_BUILDINGS).copy()
        shuffle(self.buildingStack)
        self.availableBuildings = [self.drawBuilding() for _ in range(NUM_BUILDERS_HALL)]
        self.buildersHallVPs = [0] * NUM_BUILDERS_HALL

        # Discard pile of quests
        self.questDiscard = []

    def reprBuilding(self, building):
        return f"{building}. Occupier: {self.buildings[building]}"

    def __repr__(self):
        res = "\n".join(["Buildings:"] + [
            '\t' + self.reprBuilding(building)
            for building in self.buildings
        ])

        res += "\n\nQuests (at Cliffwatch Inn):\n"
        for quest in self.availableQuests:
            res += f"\t{quest}\n"

        return res 

    def clearBuildings(self):
        '''Clears all buildings to their unoccupied states.'''
        for building in self.buildings:
            self.buildings[building] = None
    
    def drawQuest(self) -> Quest:
        '''
        Draw the top quest from the quest stack,
        removing it from the stack in the process.
        Resuffles discard pile if stack is empty.
        
        Returns: 
            The top quest from the quest stack.
        '''
        
        # Need to reshuffle discard pile
        if len(self.questStack) == 0:
            if len(self.questDiscard) == 0:
                raise ValueError("Out of quests, and empty discard pile!")
            self.questStack = self.questDiscard
            self.questDiscard = []
            shuffle(self.questStack)

        return self.questStack.pop()
    
    def drawIntrigue(self) -> str:
        '''
        Draw the top quest from the intrigue card stack,
        removing it from the stack in the process.
        
        Returns: 
            The top intrigue card from the intrigue card stack.
        '''
        return self.intrigueStack.pop()
    
    def drawBuilding(self) -> CustomBuilding:
        '''
        Draw the top quest from the building stack,
        removing it from the stack in the process.
        
        Returns: 
            The top custom building from the building stack.
        '''
        return self.buildingStack.pop()

    def chooseQuest(self, quest_idx):
        ''' Choose available quest with index quest_idx,
        replacing it with a new quest and returning it. '''
        quest = self.availableQuests[quest_idx]
        self.availableQuests[quest_idx] = self.drawQuest()
        return quest 
    
    def purchaseBuilding(self, building: CustomBuilding, owner: str):
        ''' Choose available building with index building_idx,
        replacing it with a new building and adding it to
        self.buildings with owner `owner`. Returns the 
        number of victory points awarded to the purchaser,
        and the cost of gold to be removed from the purchaser. '''
        building_idx = self.availableBuildings.index(building)
        VPs = self.buildersHallVPs[building_idx]
        self.buildersHallVPs[building_idx] = 0
        self.availableBuildings[building_idx] = self.drawBuilding()

        purchasedBuilding = building.purchase(owner)
        self.buildings[purchasedBuilding] = None
        self.customBuildings.append(purchasedBuilding)
        # TODO: If we implement building lord card, keep track 
        # of # buildings owned in player state
        return VPs,building.cost

    def resetQuests(self):
        ''' Reset the quests at Cliffwatch Inn '''
        self.questDiscard.extend(self.availableQuests)
        self.availableQuests = [
            self.drawQuest()
            for _ in range(NUM_CLIFFWATCH_QUESTS)
        ]

def main():
    # Test the quest stack
    boardState = BoardState()
    for i in range(2):
        print("Drew quest:", boardState.drawQuest())
        boardState.printQuestStack()
        print()