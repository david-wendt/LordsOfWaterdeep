from dataclasses import dataclass
from typing import Self

SCORE_PER_VP_2 = 0.5 # to avoid circular import from game_info

@dataclass 
class Resources: 
    ''' Class representing a resource bundle '''
    wizards: int = 0
    clerics: int = 0
    fighters: int = 0
    rogues: int = 0
    gold: int = 0
    VPs: int = 0

    def __repr__(self) -> str:
        res = ""
        # if self.VPs > 0: # Only display VPs for FixedResources
        #     res += f"VPs: {self.VPs}, "
        if self.wizards != 0:
            res += f"wizards: {self.wizards}, "
        if self.clerics != 0:
            res += f"clerics: {self.clerics}, "
        if self.fighters != 0:
            res += f"fighters: {self.fighters}, "
        if self.rogues != 0:
            res += f"rogues: {self.rogues}, "
        if self.gold != 0:
            res += f"gold: {self.gold}, "

        if res[-2:] == ", ":
            res = res[:-2]

        return res
    
    def __bool__(self) -> bool:
        return bool(
            self.wizards
            or self.clerics 
            or self.fighters
            or self.rogues
            or self.gold
            or self.VPs
        )
    
    def __mul__(self, other: int):
        assert isinstance(other, int)
        return Resources(
            wizards=self.wizards * other,
            clerics=self.clerics * other,
            fighters=self.fighters * other,
            rogues=self.rogues * other,
            gold=self.gold * other,
            VPs=self.VPs * other,
        )
    
    __rmul__ = __mul__

    def dot(self, other: Self):
        ''' Dot (inner) product between two resource bundles 
        according to score of each resource'''
        assert isinstance(other, (Resources, FixedResources))
        return (
            self.wizards * other.wizards
            + self.clerics * other.clerics
            + self.fighters * other.fighters / 2.
            + self.rogues * other.rogues / 2.
            + self.gold * other.gold / 4.
            + self.VPs * other.VPs * SCORE_PER_VP_2
        )

    def __add__(self, other: Self):
        assert isinstance(other, Resources)
        return Resources(
            wizards=self.wizards + other.wizards,
            clerics=self.clerics + other.clerics,
            fighters=self.fighters + other.fighters,
            rogues=self.rogues + other.rogues,
            gold=self.gold + other.gold,
            VPs=self.VPs + other.VPs,
        )

    def __sub__(self, other: Self):
        assert isinstance(other, Resources)
        return Resources(
            wizards=self.wizards - other.wizards,
            clerics=self.clerics - other.clerics,
            fighters=self.fighters - other.fighters,
            rogues=self.rogues - other.rogues,
            gold=self.gold - other.gold,
            VPs=self.VPs - other.VPs,
        )
    
    def scaled_mean(self):
        ''' Computes mean over adventurers + gold (EXCLUDES VPs),
        scaled by score of each resource type'''
        return (
            self.wizards
            + self.clerics 
            + self.fighters / 2.
            + self.rogues / 2.
            + self.gold / 4.
        ) / 5. 
    
    def toFixedResources(self, quests=0, intrigues=0):
        return FixedResources(
            wizards=self.wizards,
            clerics=self.clerics,
            fighters=self.fighters,
            rogues=self.rogues,
            gold=self.gold,
            VPs=self.VPs,
            quests=quests,
            intrigues=intrigues
        )

    
@dataclass(frozen=True) 
class FixedResources: 
    ''' Class representing an immutable resource bundle (with quests + intrigues) '''
    wizards: int = 0
    clerics: int = 0
    fighters: int = 0
    rogues: int = 0
    gold: int = 0
    VPs: int = 0
    quests: int = 0
    intrigues: int = 0

    def __repr__(self) -> str:
        res = ""
        if self.VPs > 0:
            res += f"VPs: {self.VPs}, "
        if self.wizards != 0:
            res += f"wizards: {self.wizards}, "
        if self.clerics != 0:
            res += f"clerics: {self.clerics}, "
        if self.fighters != 0:
            res += f"fighters: {self.fighters}, "
        if self.rogues != 0:
            res += f"rogues: {self.rogues}, "
        if self.gold != 0:
            res += f"gold: {self.gold}, "
        if self.quests != 0:
            res += f"quests: {self.quests}, "
        if self.intrigues != 0:
            res += f"intrigues: {self.intrigues}, "

        if res[-2:] == ", ":
            res = res[:-2]

        return res
    
    def __bool__(self) -> bool:
        return bool(
            self.wizards
            or self.clerics 
            or self.fighters
            or self.rogues
            or self.gold
            or self.quests
            or self.intrigues
            or self.VPs
        )
    
    def toResources(self) -> tuple[Resources,int,int]:
        ''' Returns tuple of (Resources, nQuests, nIntrigues)
        from FixedResources. '''
        return Resources(
            wizards=self.wizards,
            clerics=self.clerics,
            fighters=self.fighters,
            rogues=self.rogues,
            gold=self.gold,
            VPs=self.VPs
        ), self.quests, self.intrigues
    
    def split(self) -> list[Self]:
        singleResourceBundles = []

        if self.VPs > 0:
            singleResourceBundles.append(FixedResources(VPs=self.VPs))
        if self.wizards > 0:
            singleResourceBundles.append(FixedResources(wizards=self.wizards))
        if self.clerics > 0:
            singleResourceBundles.append(FixedResources(clerics=self.clerics))
        if self.fighters > 0:
            singleResourceBundles.append(FixedResources(fighters=self.fighters))
        if self.rogues > 0:
            singleResourceBundles.append(FixedResources(rogues=self.rogues))
        if self.gold > 0:
            singleResourceBundles.append(FixedResources(gold=self.gold))
        if self.quests > 0:
            raise ValueError('Custom buildings should not give quests to owners!')
        if self.intrigues > 0:
            singleResourceBundles.append(FixedResources(intrigues=self.intrigues))
        
        return singleResourceBundles
    
    def __mul__(self, other: int):
        assert isinstance(other, int)
        return FixedResources(
            wizards=self.wizards * other,
            clerics=self.clerics * other,
            fighters=self.fighters * other,
            rogues=self.rogues * other,
            gold=self.gold * other,
            VPs=self.VPs * other,
            quests=self.quests * other,
            intrigues=self.intrigues * other
        )
    
    __rmul__ = __mul__

    def dot(self, other: Resources | Self):
        ''' Dot (inner) product between a FixedResources bundle
         and either another Fixedresources bundle or a Resources bundle '''
        if isinstance(other, Resources):
            return (
                self.wizards * other.wizards
                + self.clerics * other.clerics
                + self.fighters * other.fighters / 2.
                + self.rogues * other.rogues / 2.
                + self.gold * other.gold / 4.
                + self.VPs * other.VPs * SCORE_PER_VP_2
            )
        elif isinstance(other, FixedResources):
            return (
                self.wizards * other.wizards
                + self.clerics * other.clerics
                + self.fighters * other.fighters / 2.
                + self.rogues * other.rogues / 2.
                + self.gold * other.gold / 4.
                + self.VPs * other.VPs * SCORE_PER_VP_2
                + self.quests * other.quests * 0.5 # Hardcoded score per quest
                + self.intrigues * other.intrigues * 0.5 # Hardcoded score per intrigue
            )
        else:
            raise TypeError(f'Invalid `dot` method between instance of FixedResources and {type(other)}')
    


STANDARD_RESOURCE_BUNDLES = [
    Resources(wizards=1),
    Resources(clerics=1),
    Resources(fighters=2),
    Resources(rogues=2),
    Resources(gold=4)
]
N_RESOURCE_TYPES = len(STANDARD_RESOURCE_BUNDLES) # cubes + gold, not VP/Q/I
