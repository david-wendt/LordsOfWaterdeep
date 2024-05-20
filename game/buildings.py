from dataclasses import dataclass
from game.resources import FixedResources

@dataclass(frozen=True)
class Building:
    name: str 
    rewards: FixedResources
    playIntrigue: bool = False
    getCastle: bool = False

# Define all buildings
DEFAULT_BUILDINGS = [
    Building("Blackstaff Tower", FixedResources(wizards=1)), # Blackstaff Tower (for Wizards)
    Building("Field of Triumph", FixedResources(fighters=2)), # Field of Triumph (for Fighters)
    Building("The Plinth", FixedResources(clerics=1)), # The Plinth (for Clerics)
    Building("The Grinning Lion Tavern", FixedResources(rogues=2)), # The Grinning Lion Tavern (for Rogues)
    Building("Aurora's Realms Shop", FixedResources(gold=4)), # Aurora's Realms Shop (for Gold)
    Building("Cliffwatch Inn, gold", FixedResources(gold=2, quests=1)), # Cliffwatch Inn, gold spot (for Quests)
    Building("Cliffwatch Inn, intrigue", FixedResources(quests=1, intrigues=1)), # Cliffwatch Inn, intrigue spot (for Quests)
    Building("Cliffwatch Inn, reset", FixedResources(quests=1)), # Cliffwatch Inn, reset quest spot (for Quests)
    Building("Castle Waterdeep", FixedResources(intrigues=1), getCastle=True), # Castle Waterdeep (for Castle + Intrigue)
    Building("Waterdeep Harbor 1", FixedResources(), playIntrigue=True), # Waterdeep Harbor, first slot (for playing Intrigue)
    Building("Waterdeep Harbor 2", FixedResources(), playIntrigue=True), # Waterdeep Harbor, second slot (for playing Intrigue)
    Building("Waterdeep Harbor 3", FixedResources(), playIntrigue=True), # Waterdeep Harbor, third slot (for playing Intrigue)
    # TODO (later) uncomment Builder's Hall
    # "Builder's Hall": Resources(), # Builder's Hall (for buying Buildings)
]

# TODO (later): change the below to add all empty building slots
NUM_POSSIBLE_BUILDINGS = len(DEFAULT_BUILDINGS)