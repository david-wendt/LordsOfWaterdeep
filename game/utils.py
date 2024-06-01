from game.buildings import Building
from game.player import Player

def filterWaterdeep(buildings: list[Building]):
    ''' Make sure only one waterdeep harbor slot is in list of buildings. '''
    buildingDict = {building.name: building for building in buildings}
    if ("Waterdeep Harbor 3" in buildingDict.keys()
        and ("Waterdeep Harbor 2" in buildingDict.keys()
             or "Waterdeep Harbor 1" in buildingDict.keys())):
            buildings.remove(buildingDict["Waterdeep Harbor 3"])

    if ("Waterdeep Harbor 2" in buildingDict.keys()
        and "Waterdeep Harbor 1" in buildingDict.keys()):
        buildings.remove(buildingDict["Waterdeep Harbor 2"])

    waterdeepHarbors = [building.name for building in buildings
                     if "Waterdeep Harbor" in building.name]
    assert len(waterdeepHarbors) in [0,1]
    return buildings

def getWaterdeepHarbors(buildings: dict[Building,str]):
    return sorted([
        building for building in buildings
        if isinstance(building, Building) and building.reassign
    ], key=lambda building: building.name)

def reorderPlayers(players: list[Player], currentPlayer: Player):
     # Reorder players so currentPlayer is always first
    players = players.copy()
    currentPlayerIdx = players.index(currentPlayer)
    if currentPlayerIdx != 0:
        players = players[currentPlayerIdx:] + players[:currentPlayerIdx]
    assert players[0] == currentPlayer # Double-check the above
    return players

def getOpponents(players: list[Player], currentPlayer: Player):
     return reorderPlayers(players, currentPlayer)[1:]