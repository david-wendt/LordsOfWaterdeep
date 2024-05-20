from dataclasses import dataclass
from game.resources import FixedResources

@dataclass(frozen=True)
class Building:
    name: str 
    rewards: FixedResources
    playIntrigue: bool = False
    reassign: bool = False 
    getCastle: bool = False
    resetQuests: bool = False

    def __repr__(self):
        res = f"{self.name}: rewards"
        extra_rewards = []
        if self.playIntrigue:
            extra_rewards.append("Play Intrigue")
        if self.reassign:
            extra_rewards.append("reassign")
        if self.getCastle:
            extra_rewards.append("Castle Waterdeep")
        extra_rewards = " + ".join(extra_rewards)
        
        if self.rewards:
            res += f" {self.rewards}"
            if extra_rewards:
                res += " + " + extra_rewards
        else:
            if extra_rewards:
                res += " " + extra_rewards
            else:
                raise ValueError('Building rewards nothing!')
            
        return res

# Define all buildings
DEFAULT_BUILDINGS = [
    Building("Blackstaff Tower", FixedResources(wizards=1)), # Blackstaff Tower (for Wizards)
    Building("Field of Triumph", FixedResources(fighters=2)), # Field of Triumph (for Fighters)
    Building("The Plinth", FixedResources(clerics=1)), # The Plinth (for Clerics)
    Building("The Grinning Lion Tavern", FixedResources(rogues=2)), # The Grinning Lion Tavern (for Rogues)
    Building("Aurora's Realms Shop", FixedResources(gold=4)), # Aurora's Realms Shop (for Gold)
    Building("Cliffwatch Inn, gold", FixedResources(gold=2, quests=1)), # Cliffwatch Inn, gold spot (for Quests)
    Building("Cliffwatch Inn, intrigue", FixedResources(quests=1, intrigues=1)), # Cliffwatch Inn, intrigue spot (for Quests)
    Building("Cliffwatch Inn, reset", FixedResources(quests=1), resetQuests=True), # Cliffwatch Inn, reset quest spot (for Quests)
    Building("Castle Waterdeep", FixedResources(intrigues=1), getCastle=True), # Castle Waterdeep (for Castle + Intrigue)
    Building("Waterdeep Harbor 1", FixedResources(), playIntrigue=True, reassign=True), # Waterdeep Harbor, first slot (for playing Intrigue)
    Building("Waterdeep Harbor 2", FixedResources(), playIntrigue=True, reassign=True), # Waterdeep Harbor, second slot (for playing Intrigue)
    Building("Waterdeep Harbor 3", FixedResources(), playIntrigue=True, reassign=True), # Waterdeep Harbor, third slot (for playing Intrigue)
    # TODO (later) uncomment Builder's Hall
    # "Builder's Hall": Resources(), # Builder's Hall (for buying Buildings)
]

# TODO (later): change the below to add all empty building slots
NUM_POSSIBLE_BUILDINGS = len(DEFAULT_BUILDINGS)