from collections import namedtuple

SCORE_PER_VP = 0.5 
# TODO: Make sure this is not too low.
# 'score' here means 'one turn's worth of value'
# and at the end of the game you can get 2VP/turn.
# However, we need to make sure that completing a quest
# increases total score

# Seems like the intended value is 0.25 (based on analysis here https://docs.google.com/spreadsheets/d/1rGbUNVHCKTy-D7s4yezfK91vA1HQFlCmWWY78Ize7xY/edit#gid=0
# to make the numbers work out for various quest categories), so 
# I think I'm happy upping it to 0.5 to incentivize quest completion

# Define resources
RESOURCES = [
    "VP", # Victory points
    "Q", # Quest cards (as rewards to be obtained)
    # "I", # Intrigue cards (as rewards to be obtained) # TODO (later versions): Uncomment this
    "Purple", # Wizard item, i.e. purple cubes
    "White", # Cleric item, i.e. white cubes
    "Black", # Rogue item, i.e. black cubes
    "Orange", # Fighter item, i.e. orange cubes
    "Gold", # Gold
]

# Define all quest types
QUEST_TYPES = ["Arcana", "Piety", "Skullduggery", "Warfare", "Commerce"]


# Define some constants for use in creating fake quests
QUEST_TYPE_RESOURCES = ["Purple", "White", "Black", "Orange", "Gold"]
ONE_MOVE_RESOURCES = [1, 1, 2, 2, 4]


# Define the quest type (i.e. Python type not in-game type) as a namedtuple
Quest = namedtuple("Quest", "name type requirements rewards")


# Fake quests which can be completed in one move
# QUESTS = [
#     Quest("Simple Warfare", "Warfare", {"Orange": 2}, {"VP": 4}),
#     Quest("Simple Skullduggery", "Skullduggery", {"Black": 2}, {"VP": 4}),
#     Quest("Simple Piety", "Piety", {"White": 1}, {"VP": 4}),
#     Quest("Simple Arcana", "Arcana", {"Purple": 1}, {"VP": 4}),
#     Quest("Simple Commerce", "Commerce", {"Gold": 4}, {"VP": 4}),
# ]

# Fake quests which can be completed in two moves
QUESTS = []
for i,type1 in enumerate(QUEST_TYPES):
    for j,type2 in enumerate(QUEST_TYPES):  
        if i == j:
            resources = {QUEST_TYPE_RESOURCES[i]: 2 * ONE_MOVE_RESOURCES[i]}
        else:
            resources = {QUEST_TYPE_RESOURCES[i]: ONE_MOVE_RESOURCES[i],
                         QUEST_TYPE_RESOURCES[j]: ONE_MOVE_RESOURCES[j]}
        QUESTS.append(Quest(
            type1 + " + " + type2,
            type1, resources, 
            {"VP": 8}
        ))


# Define number of agents per player as a function of number of players
def agentsPerPlayer(numPlayers: int):
    '''
    Return the number of agents per player
    as a function of the number of players in 
    the game.
    
    Args:
        numPlayers: the total number of players in the game.
        
    Returns:
        the number of agents per player.
    '''
    if numPlayers == 2: return 4
    if numPlayers == 3: return 3
    if numPlayers == 4: return 2
    if numPlayers == 5: return 2 
    else: 
        raise ValueError("Number of players must be an integer between 2 and 5, inclusive.")

# Define the Lord cards (i.e. secret identities)
# TODO (later version): uncomment below
LORD_CARDS = []
for i,type1 in enumerate(QUEST_TYPES):
    for type2 in QUEST_TYPES[i+1:]:
        LORD_CARDS.append((type1, type2))
# LORD_CARDS.append("Buildings")


# Define all buildings
DEFAULT_BUILDINGS = [ 
    # TODO: Temporary 3 cliffwatch spaces should all give 2 gold?
    #       Or maybe for now, one spot which draws a quest, and 
    #       a different one which resets then draws?

    # TODO (later) uncomment the below
    # TODO (later) make the cliffwatch spaces correct (i.e. add intrigue)
    "Purple", # Blackstaff Tower (for Wizards)
    "Orange", # Field of Triumph (for Fighters)
    "White", # The Plinth (for Clerics)
    "Black", # The Grinning Lion Tavern (for Rogues)
    "Gold", # Aurora's Realms Shop (for Gold)
    "Quest", # Cliffwatch Inn (for Quests)
    # "Castle", # Castle Waterdeep (for Castle + Intrigue)
    # "Builder", # Builder's Hall (for buying Buildings)
    # "Waterdeep1", # Waterdeep Harbor, first slot (for playing Intrigue)
    # "Waterdeep2", # Waterdeep Harbor, second slot (for playing Intrigue)
    # "Waterdeep3", # Waterdeep Harbor, third slot (for playing Intrigue)
]

# TODO (later): change the below to add all empty building slots
NUM_POSSIBLE_BUILDINGS = len(DEFAULT_BUILDINGS)

def main():
    # Verify correctness of quest attributes
    for quest in QUESTS:
        for rewardKey in quest.rewards:
            assert rewardKey in RESOURCES, quest
        for requirementKey in quest.requirements:
            assert requirementKey in RESOURCES, quest
        assert quest.type in QUEST_TYPES

    # Demonstrate how to access quest data
    print(QUESTS)
    orangeQuest = QUESTS[0]
    print(orangeQuest)
    print(orangeQuest.type)
    print(orangeQuest.rewards)
    print(orangeQuest.rewards["VP"])

    # Demonstrate that quests are immutable
    orangeQuest.rewards = {"black": 2} # Results in AttributeError