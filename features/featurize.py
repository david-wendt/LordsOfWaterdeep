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
        includeVP: bool,
        includeQI: bool = False
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

    if isinstance(resources, FixedResources) and includeQI:
        data += [
            resources.quests,
            resources.intrigues
        ]

    return torch.tensor(data)

def featurizeQuest(quest: Quest):
    '''Featurize a quest.'''
    # One-hot vector for quest type
    typeVector = torch.tensor(np.array(QUEST_TYPES) == quest.type)

    # Appent feature vectors for requirements and rewards
    return torch.cat([typeVector, 
                      featurizeResources(quest.requirements, includeVP=False), 
                      featurizeResources(quest.rewards, includeVP=True, includeQI=True)])

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

# TODO (later): uncomment the below and complete
# def featurizeBuilding(rewards: dict[str,int], ownerRewards: dict[str,int],
#                       cost: int, state: int):
#     '''
#     Featurize one non-default building, either built already or not yet built.

#     Args: 
#         rewards: the reward dictionary a player gets when placing 
#             an agent at the building
#         ownerRewards: the reward dictionary the building owner receives
#             when a player places an agent there
#         cost: the cost (in gold) for buying a building if it is unbuilt.
#             Set to 'None' when it is already built.
#         state: the occupation state of the building. None if unoccupied,
#             otherwise a player's name.
#     '''
#     # Cost should be None when the building is already built
#     if cost == None:
#         # Featurize a built building
#         pass 
#     else:
#         # Featurize an unbuilt building
#         pass
#
#     # Hint: for reward dicts, use featurizeRewards
#     # (check all buildings to see whether intrigue/quests/VP can be both
#     #  player and owner rewards or not.)
#
#     # For builidngs that gather rewards over time, maybe
#     # put unbuilt rewards as one rounds worth? or 1.5 or 1.25 or something?
#     raise Exception("Not yet implemented.")

def featurizeBoardState(boardState: BoardState, playerNames: list[str]):
    '''Featurize the state of the game board (but not the players).'''
    # Concatendated one-hot vectors for building occupations by player
    buildingStateList = []
    for building in DEFAULT_BUILDINGS:
        buildingState = np.array(playerNames) == boardState.buildings[building]
        buildingStateList.append(torch.tensor(buildingState))

    # TODO (later): do the same but for all possible building spots

    availableQuestFeatures = [featurizeQuest(quest) for quest in boardState.availableQuests]

    # TODO (later): put featurized available buildings (i.e. to build) here

    return torch.cat(buildingStateList + availableQuestFeatures)

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

# Deprecated (from last spring)
# def featurizeAction(gameState: GameState, action: str):
#     # The first four elements of the action feature vector 
#     # will correspond to how much the agent wants each 
#     # of the four quests available at Cliffwatch Inn. 
#     # Then, if the agent chooses the quest spot, we
#     # can also get the argmax of the first four q-values
#     # to choose a quest.
#     #      Actually, I'm not sure that this action-choosing
#     # framework is correct. Don't we need to compute the q-value
#     # for each action vector by feeding the vector into the q-network?
#     # It makes more sense to feed in a list of action vectors then
#     # instead of taking an argmax over elements. What elements will
#     # we have in a q-value vector to argmax over other than the outputs
#     # of the q-network on the action vectors we feed in?
#     numActions = 4

#     # One possible action per building space on the game board
#     numActions += NUM_POSSIBLE_BUILDINGS

#     # One possible action per possible active quest to complete.
#     numActions += MAX_QUESTS

#     # TODO (later): action choices for:
#     #   - choosing a building to build
#     #   - choosing an intrigue card from one's hand
#     #   - giving intrigue card rewards to another player 
#     #   - anything else?

#     actionFeatures = np.zeros(numActions)
#     if action in DEFAULT_BUILDINGS:
#         actionFeatures[4 + DEFAULT_BUILDINGS.index(action)] = 1
#     # elif action[:8] == "BUILDING": # One of the built buildings
#         # actionFeatures[4 + len(DEFAULT_BUILDINGS) + (INDEX OF BUILT BUILDING)] = 1
#     elif action[:8] == "COMPLETE": 
#         questIndex = int(action[8:])
#         actionFeatures[4 + NUM_POSSIBLE_BUILDINGS + questIndex] = 1
#     else:
#         raise ValueError("No other possible actions.")
    
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
        correspond to choosing a quest from cliffwatch
    the next N_RESOURCE_TYPES entries correspond to
        choosing a resource from an intrigue card
    and the final 1 + N_MAX_ACTIVE_QUESTS correspond 
        to choosing to not complete a quest or 
        to choosing an active quest to complete 
        (respectively)
    '''

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