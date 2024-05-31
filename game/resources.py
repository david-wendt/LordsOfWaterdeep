from dataclasses import dataclass
from typing import Self

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
        if self.wizards > 0:
            res += f"wizards: {self.wizards}, "
        if self.clerics > 0:
            res += f"clerics: {self.clerics}, "
        if self.fighters > 0:
            res += f"fighters: {self.fighters}, "
        if self.rogues > 0:
            res += f"rogues: {self.rogues}, "
        if self.gold > 0:
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
            or self.quests
            or self.intrigues
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
        if self.wizards > 0:
            res += f"wizards: {self.wizards}, "
        if self.clerics > 0:
            res += f"clerics: {self.clerics}, "
        if self.fighters > 0:
            res += f"fighters: {self.fighters}, "
        if self.rogues > 0:
            res += f"rogues: {self.rogues}, "
        if self.gold > 0:
            res += f"gold: {self.gold}, "
        if self.quests > 0:
            res += f"quests: {self.quests}, "
        if self.intrigues > 0:
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
    
    def toResources(self) -> Resources:
        return Resources(
            wizards=self.wizards,
            clerics=self.clerics,
            fighters=self.fighters,
            rogues=self.rogues,
            gold=self.gold,
            VPs=self.VPs
        )
    
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


STANDARD_RESOURCE_BUNDLES = [
    Resources(wizards=1),
    Resources(clerics=1),
    Resources(fighters=2),
    Resources(rogues=2),
    Resources(gold=4)
]
N_RESOURCE_TYPES = len(STANDARD_RESOURCE_BUNDLES) # cubes + gold, not VP/Q/I
