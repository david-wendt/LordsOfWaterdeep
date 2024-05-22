import torch

from game.game_info import RESOURCES, QUESTS, QUEST_TYPES, DEFAULT_BUILDINGS, Quest, NUM_POSSIBLE_BUILDINGS
from game.player import Player
from game.board import BoardState
from game.game import GameState

def featurizeResources(resources: dict[str, int], includeVP: bool = False, includeQ: bool = False):
    '''Featurize a resource vector.'''
    # TODO (later): change "includeQVP" to "includeQI"
    possibleResources = RESOURCES.copy() # All resources, incluidng VP, I, and Q
    if not includeVP:
        possibleResources.remove("VP")
    if not includeQ:
        possibleResources.remove("Q")

    # Each element of the feature vector corresponds to one resource type
    return np.array([resources.get(resource, 0) for resource in possibleResources])

def featurizeQuest(quest: Quest):
    '''Featurize a quest.'''
    # One-hot vector for quest type
    typeVector = (np.array(QUEST_TYPES) == quest.type).astype(int)

    # Appent feature vectors for requirements and rewards
    return np.hstack([typeVector, 
                      featurizeResources(quest.requirements), 
                      featurizeResources(quest.rewards, includeVP=True, includeQ=True)])

# The length of a quest feature vector (for use in zero-blocks)
QUEST_FEATURE_LEN = len(featurizeQuest(QUESTS[0]))

# Define the maximum number of quests a player can have.
# This determines the number of quest feature vector blocks 
# there is room for in the overall feature vectors.
MAX_QUESTS = 15

def featurizePlayer(player: Player):
    '''Featurize a player state.'''
    # One element for number of agents
    agentsVec = np.array([player.agents,])

    # Vector of the player's resources
    resourceVec = featurizeResources(player.resources, includeVP=True)

    # TODO: Add number of Intrigue cards 
    # TODO (later): add number of plot quests

    # Vector of the player's featurized active/completed quests
    questFeaturesList = []
    for playerQuests in [player.activeQuests, player.completedQuests]:
        for i in range(MAX_QUESTS):
            if i < len(playerQuests):
                questFeaturesList.append(featurizeQuest(playerQuests[i]))
            else:
                questFeaturesList.append(np.zeros(QUEST_FEATURE_LEN))

    # TODO: Maybe reorder based on score? 
    # You first, then main rival,
    # then second rival, etc (decreasing order of otherPlayer.score())
    # Originally I was fine with featurizing in turn order, but
    # realized it is not so simple during reassignment phase
    # when you are playing out of turn order

    return np.hstack([agentsVec, resourceVec] + questFeaturesList)

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
        buildingStateVec = np.array(playerNames) == boardState.buildingStates[building]
        buildingStateList.append(buildingStateVec.astype(int))
    buildingFeatures = np.hstack(buildingStateList)

    # TODO (later): do the same but for all possible building spots

    availableQuestFeatures = np.hstack([featurizeQuest(quest) for quest in boardState.availableQuests])

    # TODO (later): put featurized available buildings (i.e. to build) here

    return np.append(buildingFeatures, availableQuestFeatures)

def featurizeGameState(gameState: GameState):
    '''Featurize the game state.'''
    numRoundsLeftArr = np.array([gameState.roundsLeft,])
    boardStateFeatures = featurizeBoardState(gameState.boardState, gameState.playerNames)

    # Featurize players in turn order.
    playerFeatures = [featurizePlayer(player) for player in gameState.players]
    # TODO: Check if there is anything else that needs to be included as a feature.
    
    return np.hstack([numRoundsLeftArr, boardStateFeatures] + playerFeatures)

def featurizeAction(gameState: GameState, action: str):
    # The first four elements of the action feature vector 
    # will correspond to how much the agent wants each 
    # of the four quests available at Cliffwatch Inn. 
    # Then, if the agent chooses the quest spot, we
    # can also get the argmax of the first four q-values
    # to choose a quest.
    #      Actually, I'm not sure that this action-choosing
    # framework is correct. Don't we need to compute the q-value
    # for each action vector by feeding the vector into the q-network?
    # It makes more sense to feed in a list of action vectors then
    # instead of taking an argmax over elements. What elements will
    # we have in a q-value vector to argmax over other than the outputs
    # of the q-network on the action vectors we feed in?
    numActions = 4

    # One possible action per building space on the game board
    numActions += NUM_POSSIBLE_BUILDINGS

    # One possible action per possible active quest to complete.
    numActions += MAX_QUESTS

    # TODO (later): action choices for:
    #   - choosing a building to build
    #   - choosing an intrigue card from one's hand
    #   - giving intrigue card rewards to another player 
    #   - anything else?

    actionFeatures = np.zeros(numActions)
    if action in DEFAULT_BUILDINGS:
        actionFeatures[4 + DEFAULT_BUILDINGS.index(action)] = 1
    # elif action[:8] == "BUILDING": # One of the built buildings
        # actionFeatures[4 + len(DEFAULT_BUILDINGS) + (INDEX OF BUILT BUILDING)] = 1
    elif action[:8] == "COMPLETE": 
        questIndex = int(action[8:])
        actionFeatures[4 + NUM_POSSIBLE_BUILDINGS + questIndex] = 1
    else:
        raise ValueError("No other possible actions.")

def featurize(gameState, playerState, actions) -> tuple[torch.Tensor, torch.Tensor]:
    stateFeatures = torch.cat([
        featurizeGameState(gameState),
        featurizePlayer(playerState)
    ])

    actionMask = getActionMask(actions)
    return torch.cat([stateFeatures, actionMask]), actionMask

# Note for self later: Although one large CNN would not work, consider forcing 
# the first layer to be the same for each quest block, for each player block, etc.

# TODO: Write test functions which print the game state and actions 
# in words to compare with the displayed game state before featurization

def main():
    # Test the quest featurization
    quest = QUESTS[np.random.randint(len(QUESTS))]
    print(quest, featurizeQuest(quest))