from dataclasses import dataclass
from game.resources import FixedResources

# Define all quest types
# (setting as constants for autocomplete + limiting magic constants)
ARCANA = "Arcana"
PIETY = "Piety"
SKULLDUGGERY = "Skullduggery"
WARFARE = "Warfare"
COMMERCE = "Commerce"

QUEST_TYPES = [ARCANA, PIETY, SKULLDUGGERY, WARFARE, COMMERCE]

@dataclass(frozen=True)
class Quest:
    ''' Class representing a quest '''
    name: str 
    type: str 
    requirements: FixedResources 
    rewards: FixedResources 

    def __repr__(self) -> str:
        return f"{self.name} ({self.type}):\n\t\tRequires {self.requirements}\n\t\tRewards {self.rewards}"


QUESTS = [
    Quest('Train Bladesingers', WARFARE, 
          requirements=FixedResources(fighters=3, wizards=1), 
          rewards=FixedResources(fighters=1, wizards=1, VPs=4)),
    Quest('Spy on the House of Light', COMMERCE,
        requirements=FixedResources(fighters=3, rogues=2),
        rewards=FixedResources(gold=6, VPs=6)),
    Quest('Safeguard Eltorchul Mage', COMMERCE,
        requirements=FixedResources(fighters=1, rogues=1, wizards=1, gold=4),
        rewards=FixedResources(wizards=2, VPs=4)),
    Quest('Expose Cult Corruption', SKULLDUGGERY,
        requirements=FixedResources(clerics=1, rogues=4),
        rewards=FixedResources(clerics=2, VPs=4)),
    Quest('Domesticate Owlbears', ARCANA,
        requirements=FixedResources(clerics=1, wizards=2),
        rewards=FixedResources(fighters=1, gold=2, VPs=8)),
    Quest('Procure Stolen Goods', SKULLDUGGERY,
        requirements=FixedResources(rogues=3, gold=6),
        rewards=FixedResources(intrigues=2, VPs=8)),
    Quest('Ambush Artor Modlin', WARFARE,
        requirements=FixedResources(clerics=1, fighters=3, rogues=1),
        rewards=FixedResources(gold=4, VPs=8)),
    Quest('Raid Orc Stronghold', WARFARE,
        requirements=FixedResources(fighters=4, rogues=2),
        rewards=FixedResources(gold=4, VPs=8)),
    Quest('Build a Reputation in Skullport', SKULLDUGGERY,
        requirements=FixedResources(fighters=1, rogues=3, gold=4),
        rewards=FixedResources(intrigues=1, VPs=10)),
    Quest('Convert a Noble to Lathander EDITED', PIETY,
        requirements=FixedResources(clerics=2, wizards=1),
        rewards=FixedResources(quests=1, VPs=10)), # Changed from 8 to 10 to put on equal footing with the next two
    Quest('Discover Hidden Temple of Lolth', PIETY,
        requirements=FixedResources(clerics=2, fighters=1, rogues=1),
        rewards=FixedResources(quests=1, VPs=10)),
    Quest('Form an Alliance with the Rashemi', PIETY,
        requirements=FixedResources(clerics=2, wizards=1),
        rewards=FixedResources(quests=1, VPs=10)),
    Quest('Thin the City Watch EDITED', COMMERCE,
        requirements=FixedResources(clerics=1, fighters=1, rogues=1, gold=4),
        rewards=FixedResources(rogues=2, VPs=8)), # Seems to be OP otherwise (see spreadsheet)
    Quest('Steal Spellbook from Silverhand', ARCANA,
        requirements=FixedResources(fighters=1, rogues=2, wizards=2),
        rewards=FixedResources(gold=4, intrigues=2, VPs=7)),
    Quest('Investigate Aberrant Infestation', ARCANA,
        requirements=FixedResources(clerics=1, fighters=1, wizards=2),
        rewards=FixedResources(intrigues=1, VPs=13))
    # TODO: FIll this with the other quests (except for plot quests)
    # TODO (later): add plot quests
]
