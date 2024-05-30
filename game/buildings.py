from dataclasses import dataclass
from game.resources import FixedResources

NUM_BUILDERS_HALL = 3

@dataclass(frozen=True)
class Building:
    name: str 
    rewards: FixedResources
    playIntrigue: bool = False
    reassign: bool = False 
    getCastle: bool = False
    resetQuests: bool = False
    buyBuilding: bool = False

    def __repr__(self):
        res = f"{self.name}: rewards"
        extra_rewards = []
        if self.playIntrigue:
            extra_rewards.append("Play Intrigue")
        if self.reassign:
            extra_rewards.append("Reassign Agent")
        if self.getCastle:
            extra_rewards.append("Castle Waterdeep")
        if self.resetQuests:
            extra_rewards.append("Reset Quests")
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
    Building("Builder's Hall", buyBuilding=True), # Builder's Hall (for buying Buildings)
]

@dataclass(frozen=True)
class CustomBuilding:
    name: str 
    rewards: FixedResources
    ownerRewards: FixedResources
    cost: int

CUSTOM_BUILDINGS = [ # Only simple ones, without any updating/spending/choosing
    # NOTE: Multiple owner resources means OR, not AND!
    CustomBuilding("Dragon Tower", FixedResources(wizards=1, intrigues=1), 
                   FixedResources(intrigues=1), cost=3),
    CustomBuilding("Fetlock Court", FixedResources(wizards=1, fighters=2), 
                   FixedResources(wizards=1, fighters=1), cost=8),
    CustomBuilding("Helmstar Warehouse", FixedResources(rogues=2, gold=2), 
                   FixedResources(rogues=1), cost=3),
    CustomBuilding("Helmstar Fighterhouse", FixedResources(fighters=2, gold=2), 
                   FixedResources(fighters=1), cost=3),
    CustomBuilding("House of Heroes", FixedResources(clerics=1, fighters=2), 
                   FixedResources(clerics=1, fighters=1), cost=8),
    CustomBuilding("House of Good Spirits, Cleric", FixedResources(fighters=1, clerics=1),
                   FixedResources(fighters=1), cost=3),
    CustomBuilding("House of Good Spirits, Wizard", FixedResources(fighters=1, wizards=1),
                   FixedResources(fighters=1), cost=3),
    CustomBuilding("House of Bad Spirits, Cleric", FixedResources(rogues=1, clerics=1),
                   FixedResources(rogues=1), cost=3),
    CustomBuilding("House of Bad Spirits, Wizard", FixedResources(rogues=1, wizards=1),
                   FixedResources(rogues=1), cost=3),
    CustomBuilding("The Yawning Clerics", FixedResources(clerics=2), 
                   FixedResources(clerics=1), cost=4),
    CustomBuilding("The Yawning Wizards", FixedResources(wizards=2), 
                   FixedResources(wizards=1), cost=4),
    CustomBuilding("House of the Moon", FixedResources(clerics=1, quests=1), 
                   FixedResources(gold=2), cost=3),
    CustomBuilding("New Olamn", FixedResources(rogues=2, wizards=1), 
                   FixedResources(rogues=1, wizards=1), cost=8),
    CustomBuilding("Northgate Wizard", FixedResources(wizards=1, gold=2), 
                   FixedResources(VPs=2), cost=3),
    CustomBuilding("Northgate Cleric", FixedResources(clerics=1, gold=2), 
                   FixedResources(VPs=2), cost=3),
    CustomBuilding("The Skulkway", FixedResources(rogues=1, fighters=1, gold=2), 
                   FixedResources(rogues=1, fighters=1), cost=4),
    CustomBuilding("The Tower of Luck", FixedResources(clerics=1, rogues=2), 
                   FixedResources(clerics=1, rogues=1), cost=8),
]

# Ensure buildings have unique names
building_names = [building.name for building in DEFAULT_BUILDINGS + CUSTOM_BUILDINGS] 
assert len(building_names) == len(set(building_names)),"Buildings must have unique names!"

NUM_ADDITIONAL_BUILDINGS = 8