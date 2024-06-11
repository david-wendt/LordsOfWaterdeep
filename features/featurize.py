import torch
import numpy as np

from game.game_info import *
from game.player import Player
from game.board import BoardState
from game.game import GameState
from game import utils


# Define the maximum number of active quests a player can have.
# This determines the number of quest feature vector blocks 
# there is room for in the overall feature vectors.
N_MAX_ACTIVE_QUESTS = 10

N_MAX_CUSTOM_BUILDINGS = 8

STATE_DIM = 766 # Hardcoded for train.py
ACTION_DIM = 57 # Hardcoded for train.py

STATE_DIM = {
    2: 766,
    4: 1224
}

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
        data += [resources.intrigues]

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

# This is outdated. Maybe replace it, but for now it is just hardcoded
# def stateDim(nPlayers: int) -> int:
#     boardStateDim = (
#         len(DEFAULT_BUILDINGS) * nPlayers
#         + QUEST_FEATURE_LEN * NUM_CLIFFWATCH_QUESTS
#     )

#     privateInfoDim = len(QUEST_TYPES) # Lord card

#     playerDim = (
#         3 # one each for agents/intrigues/castle
#         + 6 # Resources + VP
#         + len(QUEST_TYPES) # Completed quests
#         + N_MAX_ACTIVE_QUESTS * QUEST_FEATURE_LEN # Active quests
#     )

#     return ( # Comments here are lines from gameState
#         1 # numRoundsLeft = torch.tensor([gameState.roundsLeft])
#         + boardStateDim # boardStateFeatures = featurizeBoardState(gameState.boardState, playerNames)
#         + privateInfoDim # currentPlayerPrivateInfo = featurizePrivatePlayerInfo(currentPlayer)
#         + nPlayers * playerDim # playerFeatures = [featurizePlayer(player) for player in players]
#     )

def featurizePlayer(player: Player):
    '''Featurize a player state.'''
    # One element each for number of agents/intrigues/castle
    agentsIntriguesCastle = torch.tensor([
        player.agents, player.numIntrigues(), player.hasCastle])

    # Vector of the player's resources
    resourceVec = featurizeResources(player.resources, includeVP=True) 

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

    completedPlotQuests = torch.zeros(len(QUEST_TYPES))
    for questType,numCompleted in player.completedPlotQuests.items():
        type_idx = QUEST_TYPES.index(questType)
        completedPlotQuests[type_idx] = numCompleted

    return torch.cat([agentsIntriguesCastle,
                      resourceVec, completedQuestTypes,
                      completedPlotQuests]
                      + activeQuestFeatures)

def featurizePrivatePlayerInfo(player: Player):
    lordCard = torch.zeros(len(QUEST_TYPES))
    for questType in player.lordCard:
        questTypeIdx = QUEST_TYPES.index(questType)
        lordCard[questTypeIdx] += 1
        
    
    return torch.cat([lordCard, torch.tensor(player.intrigues)])

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
    customBuildingFeat = featurizeCustomBuilding(building)
    return torch.cat([
        torch.tensor(ownership),
        customBuildingFeat
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
            assert buildingFeat.shape == (CUSTOM_BUILDING_DIM + len(playerNames),),(buildingFeat.shape,CUSTOM_BUILDING_DIM,len(playerNames))
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

    players = utils.reorderPlayers(gameState.players, currentPlayer)
    playerNames = [player.name for player in players]

    numRoundsLeft = torch.tensor([gameState.roundsLeft])
    boardStateFeatures = featurizeBoardState(gameState.boardState, playerNames)
    currentPlayerPrivateInfo = featurizePrivatePlayerInfo(currentPlayer)

    # Featurize players in modified turn order (turn order cycle with curr player first)
    playerFeatures = [featurizePlayer(player) for player in players]

    resourcesToGive = gameState.getResourcesToGive()
    resourcesToGiveFeat = featurizeResources(resourcesToGive)

    return torch.cat([numRoundsLeft, boardStateFeatures, 
                      currentPlayerPrivateInfo, resourcesToGiveFeat]
                      + playerFeatures)
    
def getActionMask(actions: list, gameState: GameState, currentPlayer: Player):
    ''' Get a mask of which actions are currently availbale. '''
    # For reference:
    # List of all possible actions (with counts per 100 two-random-player games):
    # 6522     Choose a building to play an agent at (MAIN ACTION)
    #     -> pass list of buildings
    # 1932     Choose which completable quest to complete
    #     -> pass list of quests, prepended with DO_NOT_COMPLETE
    # 1552     choose a quest from cliffwatch inn
    #     -> pass list of quests (in cliffwatch)
    # 698      Choose an intrigue card to play
    #     -> pass list of intrigue cards (i.e. strings, a subset of INTRIGUES)
    # 469      Choose which building from Builder's Hall to purchase
    #     -> pass list of buildings (but they are custom and unbuilt)
    # 342      Select an opponent to give a resource to. 
    #     -> pass list of players (specifically all others than self)
    # 181      Choose which owner reward to receive from a building
    #     -> pass list of resource bundles (but there should be exactly two)
    # 65       Call in a favor: select a resource bundle from standard resource bundles
    #     -> pass list of resource bundles (specifically STANDARD_RESOURCE_BUNDLES)

    default_buildings_len = len(DEFAULT_BUILDINGS) # Choose a default building to place an agent (with below)
    custom_buildings_len = N_MAX_CUSTOM_BUILDINGS # Choose a custom building to place an agent (with above)
    cliffwatch_quest_len = NUM_CLIFFWATCH_QUESTS # Choose which quest from Cliffwatch to take
    quest_complete_len = 1 + N_MAX_ACTIVE_QUESTS # Choose which active quest to complete, or to not complete any
    builders_hall_len = NUM_BUILDERS_HALL # Choose which building from Builder's Hall to purchase
    owner_reward_len = 2 # Choose which of two owner rewards to take as building owner
    play_intrigue_len = len(INTRIGUES) # Choose which intrigue card to play
    opponent_len = gameState.numPlayers - 1 # Choose which opponent to give a resource to
    call_in_favor_len = len(STANDARD_RESOURCE_BUNDLES) # Choose which standard resource bundle to 
        # get as a reward for 'Call in a Favor' intrigue card

    default_buildings_start = 0
    default_buildings_end = default_buildings_start + default_buildings_len
    custom_buildings_start = default_buildings_end
    custom_buildings_end = custom_buildings_start + custom_buildings_len
    cliffwatch_quest_start = custom_buildings_end
    cliffwatch_quest_end = cliffwatch_quest_start + cliffwatch_quest_len
    quest_complete_start = cliffwatch_quest_end
    quest_complete_end = quest_complete_start + quest_complete_len
    builders_hall_start = quest_complete_end
    builders_hall_end = builders_hall_start + builders_hall_len
    owner_reward_start = builders_hall_end
    owner_reward_end = owner_reward_start + owner_reward_len
    play_intrigue_start = owner_reward_end
    play_intrigue_end = play_intrigue_start + play_intrigue_len
    opponent_start = play_intrigue_end
    opponent_end = opponent_start + opponent_len
    call_in_favor_start = opponent_end
    call_in_favor_end = call_in_favor_start + call_in_favor_len

    action_len = call_in_favor_end
    assert action_len == (default_buildings_len + custom_buildings_len + cliffwatch_quest_len 
                          + quest_complete_len + builders_hall_len + owner_reward_len 
                          + play_intrigue_len + opponent_len + call_in_favor_len
                          ), "The total length does not match the sum of individual lengths"
    assert action_len == ACTION_DIM

    actionMask = torch.zeros(action_len)

    if isinstance(actions[0], Building) or (
        isinstance(actions[0], CustomBuilding) 
            and actions[0].owner is not None):
        # Action type: Choose a building in which to place an agent
        for building in actions:
            if isinstance(building, Building):
                default_idx = DEFAULT_BUILDINGS.index(building)
                actionMask[default_buildings_start + default_idx] = 1
            elif isinstance(building, CustomBuilding):
                assert building.owner is not None 
                custom_idx = gameState.boardState.customBuildings.index(building)
                actionMask[custom_buildings_start + custom_idx] = 1
    
    elif isinstance(actions[0], Quest):
        # Action type: Choose which quest from Cliffwatch to take
        assert len(actions) == NUM_CLIFFWATCH_QUESTS
        for i,quest in enumerate(actions):
            assert isinstance(quest, Quest)
        
        actionMask[cliffwatch_quest_start:cliffwatch_quest_end] = torch.ones(NUM_CLIFFWATCH_QUESTS)

    elif actions[0] == DO_NOT_COMPLETE_QUEST:
        # Action type: Choose which active quest to complete, or to not complete any
        for i,action in enumerate(actions):
            if i != 0: assert isinstance(action, Quest)
        
        actionMask[quest_complete_start:
                   quest_complete_start + len(actions)] = torch.ones(len(actions))

    elif isinstance(actions[0], CustomBuilding) and actions[0].owner is None:
        # Action type: Choose which building from Builder's Hall to purchase
        # (out of the affordable subset)
        assert len(actions) <= NUM_BUILDERS_HALL
        for building in actions:
            assert isinstance(building, CustomBuilding) and building.owner is None
            building_idx = gameState.boardState.availableBuildings.index(building)
            actionMask[builders_hall_start + building_idx] = 1

    elif isinstance(actions[0], FixedResources):
        # Action type: Choose which of two owner rewards to take as building owner
        assert len(actions) == 2 and isinstance(actions[1], FixedResources)
        actionMask[owner_reward_start:owner_reward_end] = torch.ones(len(actions))

    elif isinstance(actions[0], str):
        # Action type: Choose which intrigue card to play
        for action in actions:
            intrigue_idx = INTRIGUES.index(action)
            actionMask[play_intrigue_start + intrigue_idx] = 1

    elif isinstance(actions[0], Player):
        # Action type: Choose which opponent to give a resource to
        opponents = utils.getOpponents(gameState.players, currentPlayer)
        assert actions == opponents and len(actions) == gameState.numPlayers - 1
        actionMask[opponent_start:opponent_end] = torch.ones(len(actions))

    elif isinstance(actions[0], Resources):
        # Action type: Choose which standard resource bundle to get as a reward for 'Call in a Favor' intrigue card
        assert actions == STANDARD_RESOURCE_BUNDLES
        actionMask[call_in_favor_start:call_in_favor_end] = torch.ones(len(actions))

    else:
        raise ValueError("Unknown action type:", actions)

    assert actionMask.sum() > 0
    return actionMask

def featurize(gameState, currentPlayer, actions) -> tuple[torch.Tensor, torch.Tensor]:
    stateFeatures = featurizeGameState(gameState, currentPlayer)
    actionMask = getActionMask(actions, gameState, currentPlayer)
    assert len(actionMask) == ACTION_DIM,(len(actionMask),ACTION_DIM)
    stateFeatures = torch.cat([stateFeatures, actionMask])
    assert len(stateFeatures) == STATE_DIM,(len(stateFeatures),STATE_DIM)
    return stateFeatures,actionMask

# Note for self later: Although one large CNN would not work, consider forcing 
# the first layer to be the same for each quest block, for each player block, etc.
# ^^ This was from last spring. Reminds me that maybe some 1d-conv layers (or similar
# layers of our own construction) could be helpful given the repetition in the state
# space featurization - DW (now)