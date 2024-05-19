from collections import namedtuple
from dataclasses import dataclass

SCORE_PER_VP = 0.5 
# TODO: Make sure this is not too low.
# 'score' here means 'one turn's worth of value'
# and at the end of the game you can get 2VP/turn.
# However, we need to make sure that completing a quest
# increases total score

# Seems like the intended value is 0.25 (based on analysis here https://docs.google.com/spreadsheets/d/1rGbUNVHCKTy-D7s4yezfK91vA1HQFlCmWWY78Ize7xY/edit#gid=0
# to make the numbers work out for various quest categories), so 
# I think I'm happy upping it to 0.5 to incentivize quest completion

# Define all quest types
# (setting as constants for autocomplete + limiting magic constants)
ARCANA = "Arcana"
PIETY = "Piety"
SKULLDUGGERY = "Skullduggery"
WARFARE = "Warfare"
COMMERCE = "Commerce"

QUEST_TYPES = [ARCANA, PIETY, SKULLDUGGERY, WARFARE, COMMERCE]

# Define the Lord cards (i.e. secret identities)
LORD_CARDS = []
for i,type1 in enumerate(QUEST_TYPES):
    for type2 in QUEST_TYPES[i+1:]:
        LORD_CARDS.append((type1, type2))
# TODO (later version): uncomment building lord card
# LORD_CARDS.append("Buildings")

@dataclass 
class Resources: 
    ''' Class representing a resource bundle '''
    wizards: int = 0
    clerics: int = 0
    fighters: int = 0
    rogues: int = 0
    gold: int = 0
    VPs: int = 0
    quests: int = 0
    intrigues: int = 0

@dataclass
class Quest:
    ''' Class representing a quest '''
    name: str 
    type: str 
    requirements: Resources 
    rewards: Resources 

# Define all buildings
DEFAULT_BUILDINGS = {
    "Blackstaff Tower": Resources(wizards=1), # Blackstaff Tower (for Wizards)
    "Field of Triumph": Resources(fighters=2), # Field of Triumph (for Fighters)
    "The Plinth": Resources(clerics=1), # The Plinth (for Clerics)
    "The Grinning Lion Tavern": Resources(rogues=2), # The Grinning Lion Tavern (for Rogues)
    "Aurora's Realms Shop": Resources(gold=4), # Aurora's Realms Shop (for Gold)
    "Cliffwatch Inn, gold": Resources(gold=2, quests=1), # Cliffwatch Inn, gold spot (for Quests)
    "Cliffwatch Inn, intrigue": Resources(quests=1, intrigues=1), # Cliffwatch Inn, intrigue spot (for Quests)
    "Cliffwatch Inn, reset": Resources(quests=1), # Cliffwatch Inn, reset quest spot (for Quests)
    "Castle": Resources(intrigues=1), # Castle Waterdeep (for Castle + Intrigue)
    "Waterdeep, 1": Resources(), # Waterdeep Harbor, first slot (for playing Intrigue)
    "Waterdeep, 2": Resources(), # Waterdeep Harbor, second slot (for playing Intrigue)
    "Waterdeep, 3": Resources(), # Waterdeep Harbor, third slot (for playing Intrigue)
    # TODO (later) uncomment Builder's Hall
    # "Builder's Hall": Resources(), # Builder's Hall (for buying Buildings)
}

# TODO (later): change the below to add all empty building slots
NUM_POSSIBLE_BUILDINGS = len(DEFAULT_BUILDINGS)

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
    
QUESTS = [
    Quest('Train Bladesingers', WARFARE, 
          requirements=Resources(fighters=3, wizards=1), 
          rewards=Resources(fighters=1, wizards=1, VPs=4)),
    Quest('Spy on the House of Light', COMMERCE,
        requirements=Resources(fighters=3, rogues=2),
        rewards=Resources(gold=6, VPs=6)),
    Quest('Safeguard Eltorchul Mage', COMMERCE,
        requirements=Resources(fighters=1, rogues=1, wizards=1, gold=4),
        rewards=Resources(wizards=2, VPs=4)),
    Quest('Expose Cult Corruption', SKULLDUGGERY,
        requirements=Resources(clerics=1, rogues=4),
        rewards=Resources(clerics=2, VPs=4)),
    Quest('Domesticate Owlbears', ARCANA,
        requirements=Resources(clerics=1, wizards=2),
        rewards=Resources(fighters=1, gold=2, VPs=8)),
    Quest('Procure Stolen Goods', SKULLDUGGERY,
        requirements=Resources(rogues=3, gold=6),
        rewards=Resources(intrigues=2, VPs=8)),
    Quest('Ambush Artor Modlin', WARFARE,
        requirements=Resources(clerics=1, fighters=3, rogues=1),
        rewards=Resources(gold=4, VPs=8)),
    Quest('Raid Orc Stronghold', WARFARE,
        requirements=Resources(fighters=4, rogues=2),
        rewards=Resources(gold=4, VPs=8)),
    Quest('Build a Reputation in Skullport', SKULLDUGGERY,
        requirements=Resources(),
        rewards=Resources()),
    Quest('Convert a Noble to Lathander EDITED', PIETY,
        requirements=Resources(clerics=2, wizards=1),
        rewards=Resources(quests=1, VPs=10)), # Changed from 8 to 10 to put on equal footing with the next two
    Quest('Discover Hidden Temple of Lolth', PIETY,
        requirements=Resources(clerics=2, fighters=1, rogues=1),
        rewards=Resources(quests=1, VPs=10)),
    Quest('Form an Alliance with the Rashemi', PIETY,
        requirements=Resources(clerics=2, wizards=1),
        rewards=Resources(quests=1, VPs=10)),
    Quest('Thin the City Watch EDITED', COMMERCE,
        requirements=Resources(clerics=1, fighters=1, rogues=1, gold=4),
        rewards=Resources(rogues=2, VPs=8)), # Seems to be OP otherwise (see spreadsheet)
    Quest('Steal Spellbook from Silverhand', ARCANA,
        requirements=Resources(fighters=1, rogues=2, wizards=2),
        rewards=Resources(gold=4, intrigues=2, VPs=7)),
    Quest('Investigate Aberrant Infestation', ARCANA,
        requirements=Resources(clerics=1, fighters=1, wizards=2),
        rewards=Resources(intrigues=1, VPs=13))
    # TODO: FIll this with the other quests (except for plot quests)
    # TODO (later): add plot quests
]

INTRIGUES = [
    'Choice of any resource', # 4 gold or 2 F/R or 1 W/C
] * 20
# TODO: figure out what to do for intrigue cards

def main():
    pass 