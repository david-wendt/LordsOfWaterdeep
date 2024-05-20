from dataclasses import dataclass

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
