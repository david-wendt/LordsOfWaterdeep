import torch
import numpy as np

from game.game_info import *
from game.player import Player
from game.board import BoardState
from game.game import GameState


# Define the maximum number of active quests a player can have.
# This determines the number of quest feature vector blocks 
# there is room for in the overall feature vectors.
N_MAX_ACTIVE_QUESTS = 10

# TODO (later): uncomment this for proper
# intrigue card featurizations once we have
# different intrigue cards
# N_MAX_INTRIGUE_CARDS = 10

N_ACTIONS = (len(DEFAULT_BUILDINGS)
    # + len(ADDITIONAL_BUILDINGS)
    + NUM_CLIFFWATCH_QUESTS # Number of quests to chose from cliffwatch
    # + N_MAX_INTRIGUE_CARDS # TODO (later)
    + N_RESOURCE_TYPES # Number of reseources to choose from the single intrigue
        # We have implemented so far
    + N_MAX_ACTIVE_QUESTS # For choosing a quest to complete
    + 1 # Choose not to complete a quest
    # + N_BUILDERS_HALL_BUILDINGS # Buildings at Builder's Hall to build
)

def featurizeResources(
        resources: Resources | FixedResources,
        includeVP: bool = False,
        includeQ: bool = False,
        includeI: bool = False
        ):
    
    '''Featurize a resource object.'''
    data = [
        resources.wizards,
        resources.clerics,
        resources.fighters,
        resources.rogues,
        resources.gold
    ]

    if includeVP:
        data.append(resources.VPs)

    if isinstance(resources, FixedResources) and includeQ:
        data += [resources.quests]

    if isinstance(resources, FixedResources) and includeI:
            resources.intrigues

    return torch.tensor(data)

def featurizeQuest(quest: Quest):
    '''Featurize a quest.'''
    # One-hot vector for quest type
    typeVector = torch.tensor(np.array(QUEST_TYPES) == quest.type)

    # Appent feature vectors for requirements and rewards
    return torch.cat([typeVector, 
                      featurizeResources(quest.requirements), 
                      featurizeResources(quest.rewards, includeVP=True, includeQ=True, includeI=True)])

# The length of a quest feature vector (for use in zero-blocks)
QUEST_FEATURE_LEN = featurizeQuest(QUESTS[0]).size(0)
for quest in QUESTS:
    shape = featurizeQuest(quest).size()
    assert len(shape) == 1
    assert shape[0] == QUEST_FEATURE_LEN

def stateDim(nPlayers: int) -> int:
    boardStateDim = (
        len(DEFAULT_BUILDINGS) * nPlayers
        + QUEST_FEATURE_LEN * NUM_CLIFFWATCH_QUESTS
    )

    privateInfoDim = len(QUEST_TYPES) # Lord card

    playerDim = (
        3 # one each for agents/intrigues/castle
        + 6 # Resources + VP
        + len(QUEST_TYPES) # Completed quests
        + N_MAX_ACTIVE_QUESTS * QUEST_FEATURE_LEN # Active quests
    )

    return ( # Comments here are lines from gameState
        1 # numRoundsLeft = torch.tensor([gameState.roundsLeft])
        + boardStateDim # boardStateFeatures = featurizeBoardState(gameState.boardState, playerNames)
        + privateInfoDim # currentPlayerPrivateInfo = featurizePrivatePlayerInfo(currentPlayer)
        + nPlayers * playerDim # playerFeatures = [featurizePlayer(player) for player in players]
    )

def featurizePlayer(player: Player):
    '''Featurize a player state.'''
    # One element each for number of agents/intrigues/castle
    agentsIntriguesCastle = torch.tensor([
        player.agents, len(player.intrigues), player.hasCastle])

    # Vector of the player's resources
    resourceVec = featurizeResources(player.resources, includeVP=True) 

    # TODO (later): add number of plot quests

    # Vector of the player's featurized active/completed quests
    activeQuestFeatures = []
    for i in range(N_MAX_ACTIVE_QUESTS):
        if i < len(player.activeQuests):
            quest = player.activeQuests[i]
            activeQuestFeatures.append(featurizeQuest(quest))
        else:
            activeQuestFeatures.append(torch.zeros(QUEST_FEATURE_LEN))

    completedQuestTypes = torch.zeros(len(QUEST_TYPES))
    for quest in player.completedQuests:
        type_idx = QUEST_TYPES.index(quest.type)
        completedQuestTypes[type_idx] += 1

    return torch.cat([agentsIntriguesCastle,
                      resourceVec, completedQuestTypes]
                      + activeQuestFeatures)

def featurizePrivatePlayerInfo(player: Player):
    lordCard = torch.zeros(len(QUEST_TYPES))
    for questType in player.lordCard:
        questTypeIdx = QUEST_TYPES.index(questType)
        lordCard[questTypeIdx] += 1
        
    # TODO (later): featurize intrigue cards properly if there are multiple types
    return lordCard

def featurizeCustomBuilding(building: CustomBuilding):
    return torch.cat([
        featurizeResources(building.rewards, False, True, True),
        featurizeResources(building.ownerRewards, True, False, True)
    ]) # Length 14

CUSTOM_BUILDING_DIM = 14 # As above, for rewards + owner rewards

def featurizeUnownedBuilding(building: CustomBuilding, VPs: int):
    # Rewards: Q,I but no VP
    # Owner rewards: I,VP but no Q
    assert building.owner == None,"Building must be unowned!"
    return torch.cat([torch.tensor([VPs, building.cost]), 
                      featurizeCustomBuilding(building)]) # Length 16

def featurizeOwnedBuilding(building: CustomBuilding, playerNames: np.ndarray):
    ownership = playerNames == building.owner
    return torch.cat([
        torch.tensor(ownership),
        featurizeCustomBuilding(building)
    ]) # Length nplayers + 14

def featurizeBoardState(boardState: BoardState, playerNames: list[str]):
    '''Featurize the state of the game board (but not the players).'''
    playerNames = np.array(playerNames)

    # Concatendated one-hot vectors for building occupations by player
    buildingStateList = []
    for building in DEFAULT_BUILDINGS:
        buildingState = playerNames == boardState.buildings[building]
        buildingStateList.append(torch.tensor(buildingState))

    for i in range(NUM_ADDITIONAL_BUILDINGS):
        if i < len(boardState.customBuildings):
            building = boardState.customBuildings[i]
            # Occupation state
            buildingStateList.append(torch.tensor(playerNames == boardState.buildings[building]))
            # Custom building description
            buildingFeat = featurizeOwnedBuilding(building, playerNames)
            assert buildingFeat.shape == (CUSTOM_BUILDING_DIM + len(playerNames))
            buildingStateList.append(buildingFeat)
        else:
            buildingStateList.append(torch.zeros(2 * len(playerNames) + CUSTOM_BUILDING_DIM))

    availableQuestFeatures = [featurizeQuest(quest) for quest in boardState.availableQuests]

    availableBuildingFeatures = [
        featurizeUnownedBuilding(boardState.availableBuildings[i], boardState.buildersHallVPs[i])
        for i in range(NUM_BUILDERS_HALL)
    ]

    return torch.cat(buildingStateList + availableQuestFeatures + availableBuildingFeatures)

def featurizeGameState(gameState: GameState, currentPlayer: Player):
    '''Featurize the game state.'''

    # Reorder players so current player is always first
    players = gameState.players.copy()
    currentPlayerIdx = players.index(currentPlayer)
    if currentPlayerIdx != 0:
        players = players[currentPlayerIdx:] + players[:currentPlayerIdx]
    assert players[0] == currentPlayer # Double-check the above
    playerNames = [player.name for player in players]

    numRoundsLeft = torch.tensor([gameState.roundsLeft])
    boardStateFeatures = featurizeBoardState(gameState.boardState, playerNames)
    currentPlayerPrivateInfo = featurizePrivatePlayerInfo(currentPlayer)

    # Featurize players in modified turn order (turn order cycle with curr player first)
    playerFeatures = [featurizePlayer(player) for player in players]

    return torch.cat([numRoundsLeft, boardStateFeatures, 
                      currentPlayerPrivateInfo]
                      + playerFeatures)
    
def getActionMask(actions):
    ''' Get a mask of which actions are currently availbale. 
    
    should have length 
    N_ACTIONS = (len(DEFAULT_BUILDINGS)
        + NUM_CLIFFWATCH_QUESTS # Number of quests to chose from cliffwatch
        + N_RESOURCE_TYPES # Number of reseources to choose from the single intrigue
            # We have implemented so far
        + N_MAX_ACTIVE_QUESTS # For choosing a quest to complete
        + 1 # Choose not to complete a quest
    )   
    so that first len(DEFAULT_BUILDINGS) entries 
        correspond to choosing a building to place an action,
    the next NUM_CLIFFWATCH_QUESTS entries 
        correspond to choosing a quest from Cliffwatch Inn
    the next NUM_BUILDERS_HALL entries correspond 
        to choosing a custom building from Builder's Hall
    the next N_RESOURCE_TYPES entries correspond to
        choosing a resource from an intrigue card or building (the resource you want)
    the next N_PLAYERS entries correspond to choosing an opponent
        to GIVE some resource to (i.e. higher Q-value means worse opponent)
    and the final 1 + N_MAX_ACTIVE_QUESTS correspond 
        to choosing to not complete a quest or 
        to choosing an active quest to complete 
        (respectively)
    '''
    # TODO: Update this once the above is finalized
    
    actionMask = torch.zeros(N_ACTIONS)
    if isinstance(actions[0], Building):
        # Move type: Choose a building in which to place an agent
        for building in actions:
            assert isinstance(building, Building)
            action_idx = DEFAULT_BUILDINGS.index(building)
            # TODO (later): also index from built additional buildings
            # (and change + len(DEFAULT_BUILDINGS) shifts below!!)
            actionMask[action_idx] = 1

    elif isinstance(actions[0], Quest):
        # Choose one of four quests from Cliffwatch Inn
        assert len(actions) == NUM_CLIFFWATCH_QUESTS
        for i,quest in enumerate(actions):
            assert isinstance(quest, Quest)
        
        actionMask[
            len(DEFAULT_BUILDINGS)
            : len(DEFAULT_BUILDINGS) + NUM_CLIFFWATCH_QUESTS
        ] = torch.ones(NUM_CLIFFWATCH_QUESTS)

    elif isinstance(actions[0], Resources):
        # Choose a resource bundle to receive from an intrigue card
        assert actions == STANDARD_RESOURCE_BUNDLES
        actionMask[
            len(DEFAULT_BUILDINGS) + NUM_CLIFFWATCH_QUESTS
            : len(DEFAULT_BUILDINGS) + NUM_CLIFFWATCH_QUESTS
             + len(STANDARD_RESOURCE_BUNDLES)
        ] = torch.ones(len(STANDARD_RESOURCE_BUNDLES))

    elif actions[0] == DO_NOT_COMPLETE_QUEST:
        for i,action in enumerate(actions):
            if i != 0: assert isinstance(action, Quest)
    
        actionMask[len(DEFAULT_BUILDINGS) 
            + NUM_CLIFFWATCH_QUESTS
            + len(STANDARD_RESOURCE_BUNDLES)
            : len(DEFAULT_BUILDINGS) 
            + NUM_CLIFFWATCH_QUESTS
            + len(STANDARD_RESOURCE_BUNDLES)
            + len(actions)
        ] = torch.ones(len(actions))

    elif isinstance(actions[0], str):
        assert set(actions).issubset(INTRIGUES)
        raise NotImplementedError("Choose an intrigue card")
    
    else:
        raise ValueError("Unknown action type:", actions)

    assert actionMask.sum() > 0
    return actionMask

def featurize(gameState, currentPlayer, actions) -> tuple[torch.Tensor, torch.Tensor]:
    stateFeatures = featurizeGameState(gameState, currentPlayer)
    actionMask = getActionMask(actions)
    stateFeatures = torch.cat([stateFeatures, actionMask])
    return stateFeatures,actionMask

# Note for self later: Although one large CNN would not work, consider forcing 
# the first layer to be the same for each quest block, for each player block, etc.
# ^^ This was from last spring. Reminds me that maybe some 1d-conv layers (or similar
# layers of our own construction) could be helpful given the repetition in the state
# space featurization - DW (now)