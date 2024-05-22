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
                      featurizeResources(quest.rewards, includeVP=True, includeQ=True)])

# The length of a quest feature vector (for use in zero-blocks)
QUEST_FEATURE_LEN = featurizeQuest(QUESTS[0]).size(0)
for quest in QUESTS:
    shape = featurizeQuest(quest).size()
    assert len(shape) == 1
    assert shape[0] == QUEST_FEATURE_LEN

def featurizePlayer(player: Player):
    '''Featurize a player state.'''
    # One element for number of agents
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
    playerNames = map(lambda x: x.name, players)

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
    raise NotImplementedError

def featurize(gameState, currentPlayer, actions) -> tuple[torch.Tensor, torch.Tensor]:
    stateFeatures = featurizeGameState(gameState, currentPlayer)
    actionMask = getActionMask(actions)
    return torch.cat([stateFeatures, actionMask]), actionMask

# Note for self later: Although one large CNN would not work, consider forcing 
# the first layer to be the same for each quest block, for each player block, etc.
# ^^ This was from last spring. Reminds me that maybe some 1d-conv layers (or similar
# layers of our own construction) could be helpful given the repetition in the state
# space featurization - DW (now)