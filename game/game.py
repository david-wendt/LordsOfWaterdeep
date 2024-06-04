from random import shuffle
from collections import defaultdict

from game.game_info import *
from game.player import Player
from game import board
from game import utils 
from agents.agent import Agent

# Class to control the flow of the game, focused on turn progression and move 
# options. Broadly, this class handles anything involving the game state and
# the players, while those other classes handle either only the game state
# itself or only the players themselves (to the extent possible).

# actionTypes = defaultdict(int)

class GameState():
    '''
    Class to control the flow of the game, 
    focused on turn progression and move 
    options. 
    '''
    def __init__(self, 
                 agents: list[Agent], 
                 numRounds: int = 8, 
                 playerNames=None):
        '''
        Initialize the game state and players.

        Args: 
            agents: the agent objects for the players in the game
            numRounds: the number of rounds in the game
            playerNames (optional): the names for each player
        '''

        self.numPlayers = len(agents)
        # Check that we have a valid number of players
        assert 2 <= self.numPlayers <= 5

        # Initialize the remaining number of rounds
        self.roundsLeft = numRounds

        # Initialize the BoardState
        self.boardState = board.BoardState()
        self.resourcesToGive = None # Used for intrigue cards that give resources to others

        # Set default player names
        if playerNames is None:
            playerNames = [
                "PlayerOne", "PlayerTwo", 
                "PlayerThree", "PlayerFour",
                "PlayerFive"
            ][:self.numPlayers]
        else:
            assert len(playerNames) == self.numPlayers

        if len(playerNames) != len(set(playerNames)):
            raise ValueError("Need to have unique player names")
        assert REASSIGNED not in playerNames
        
        # Shuffle the lord cards
        shuffled_lord_cards = LORD_CARDS.copy()
        shuffle(shuffled_lord_cards)

        # Initialize the players
        self.players = [Player(
            name=playerNames[i],
            agent=agent,
            numAgents=agentsPerPlayer(self.numPlayers),
            lordCard=shuffled_lord_cards[i]
        ) for i,agent in enumerate(agents)]

        self.playersInitOrder = self.players.copy()

        self.namesToPlayers = {player.name: player 
                               for player in self.players}

        # This is not only a list of players, but 
        # also represents the turn order. It will 
        # be reordered each turn.
        shuffle(self.players)

        # Deal quest/intrigue cards to players
        for i,player in enumerate(self.players):
            for _ in range(2):
                player.getQuest(self.boardState.drawQuest())
                player.getIntrigue(self.boardState.drawIntrigue())
            player.getResources(Resources(gold = 4 + i))

        # Finally, start a new round
        # (decrements roundsLeft, places VPs on
        # buildings at builder's hall, and some
        # other logic)
        self.newRound()

    def printPlayers(self):
        print('\nPLAYERS:')
        for i,player in enumerate(self.players):
            print(i,player)

    def newRound(self):
        '''Reset the board at the beginning of each round.'''
        self.roundsLeft -= 1
        for i in range(NUM_BUILDERS_HALL):
            self.boardState.buildersHallVPs[i] += 1 # Place a VP on the building

        # Reset all buildings
        self.boardState.clearBuildings()

        # Get new agent at fifth round
        if self.roundsLeft == 4:
            for player in self.players:
                player.getAgent()

        # Return all agents
        for player in self.players:
            player.returnAgents()

        # Reorder players if one took the castle
        playersHaveCastle = [player.hasCastle for player in self.players]
        assert sum(playersHaveCastle) in [0,1],playersHaveCastle
        if True in playersHaveCastle:
            player_idx = playersHaveCastle.index(True)
            castlePlayer = self.players[player_idx]
            assert castlePlayer.hasCastle,(player_idx, playersHaveCastle)
            castlePlayer.hasCastle = False 
            self.players = self.players[player_idx:] + self.players[:player_idx]
        
        # Make sure nobody has the castle after reordering
        for player in self.players:
            assert player.hasCastle == False 

    def removeFromOpponents(self, player: Player, resources: Resources):
        n_could_not_remove = 0
        for other in self.players:
            if other == player: continue # Do not remove from self
            if other.canRemoveResources(resources):
                other.removeResources(resources)
            else:
                n_could_not_remove += 1
        return n_could_not_remove
    
    def rewardQuests(self, currentPlayer: Player, nQuests: int):
        for _ in range(nQuests):
            # actionTypes["choose a quest from cliffwatch inn"] += 1
            quest_idx = currentPlayer.selectMove(self, self.boardState.availableQuests)
<<<<<<< HEAD
            assert isinstance(quest_idx, int), quest_idx
=======
            assert isinstance(quest_idx, int)
>>>>>>> f75df6a (bug fixes)
            quest = self.boardState.chooseQuest(quest_idx)
            currentPlayer.getQuest(quest)

    def rewardIntrigues(self, currentPlayer: Player, numIntrigues: int):
        for _ in range(numIntrigues):
            currentPlayer.getIntrigue(self.boardState.drawIntrigue())

    def completeQuest(self, player: Player, quest: Quest):
        # Make sure the agent has this quest
        if quest not in player.activeQuests:
            raise ValueError("This agent does not have this quest.")

        # Check if the quest can be completed                
        if not player.isValidQuestCompletion(quest):
            raise ValueError("Do not have enough resources to complete this quest.")
        
        requirements,reqQuests,reqIntrigues = quest.requirements.toResources()
        assert reqQuests == reqIntrigues == 0
        player.removeResources(requirements)

        rewards,nQuests,numIntrigues = quest.rewards.toResources()
        player.getResources(rewards)
        player.resources.VPs += PLOT_BONUS_VP * player.completedPlotQuests[quest.type]
        self.rewardQuests(player, nQuests)
        self.rewardIntrigues(player, numIntrigues)

        if quest.plot:
            player.completedPlotQuests[quest.type] += 1
        player.completedQuests.append(quest)
        player.activeQuests.remove(quest)

    def setResourcesToGive(self, resources: Resources):
        assert self.resourcesToGive is None 
        self.resourcesToGive = resources

    def getResourcesToGive(self):
        resourcesToGive = self.resourcesToGive
        self.resourcesToGive = None 
        if resourcesToGive is None:
            resourcesToGive = Resources()
        return resourcesToGive

    def playIntrigue(self, currentPlayer: Player, intrigue: str):
        assert intrigue in INTRIGUES,intrigue
        # TODO: Sometiems this assert fails with intrigue = None ??
        opponents = utils.getOpponents(self.players, currentPlayer)
        if intrigue == 'Call in a Favor':
            resource_options = STANDARD_RESOURCE_BUNDLES
            # actionTypes["Call in a favor: select a resource bundle from standard resource bundles"] += 1
            resource_idx = currentPlayer.selectMove(self, resource_options)
            currentPlayer.getResources(resource_options[resource_idx])
        elif intrigue in ['Lack of Faith', 'Ambush', 'Assassination', 'Arcane Mishap']:
            if intrigue == 'Lack of Faith':
                lostResources = Resources(clerics=1)
                gainedResources = Resources(VPs=2)
            elif intrigue == 'Ambush':
                lostResources = Resources(fighters=1)
                gainedResources = Resources(fighters=1)
            elif intrigue == 'Assassination':
                lostResources = Resources(rogues=1)
                gainedResources = Resources(gold=2)
            elif intrigue == 'Arcane Mishap':
                lostResources = Resources(wizards=1)
                gainedResources = 'Intrigue'
            else:
                raise ValueError(f"Unknown `remove + get` intrigue card: {intrigue}")
            
            n_rewards = self.removeFromOpponents(currentPlayer, lostResources)
            if gainedResources == 'Intrigue':
                self.rewardIntrigues(currentPlayer, n_rewards)
            else:
                currentPlayer.getResources(n_rewards * gainedResources)
        elif intrigue in ['Spread the Wealth', 'Graduation Day', 'Conscription', 'Good Faith', 'Crime Wave']:
            if intrigue == 'Spread the Wealth':
                currentPlayer.getResources(Resources(gold=4))
                oppResources = Resources(gold=2)
            elif intrigue == 'Graduation Day':
                currentPlayer.getResources(Resources(wizards=2))
                oppResources = Resources(wizards=1)
            elif intrigue == 'Conscription':
                currentPlayer.getResources(Resources(fighters=2))
                oppResources = Resources(fighters=1)
            elif intrigue == 'Good Faith':
                currentPlayer.getResources(Resources(clerics=2))
                oppResources = Resources(clerics=1)
            elif intrigue == 'Crime Wave':
                currentPlayer.getResources(Resources(rogues=2))
                oppResources = Resources(rogues=1)
            else:
                raise ValueError(f"Unknown `get + give` intrigue card: {intrigue}")
            # actionTypes["Select an opponent to give a resource to."] += 1
            self.setResourcesToGive(oppResources)
            opponent_idx = currentPlayer.selectMove(self, opponents)
            opponents[opponent_idx].getResources(oppResources)
            self.resourcesToGive = None 
        else:
            raise ValueError(f"Unknown intrigue card: {intrigue}")

    def takeTurn(self, currentPlayer: Player):
        '''Take a single turn in the turn order.'''

        # Return if you don't have any agents to play
        if currentPlayer.agents == 0:
            return 
        
        # Possible places to play an agent are unoccupied buildings
        possibleMoves = [building for building,occupier in self.boardState.buildings.items() 
                         if occupier is None] 
        
        # Make sure only one waterdeep harbor slot is present
        possibleMoves = utils.filterWaterdeep(possibleMoves)

        assert len(possibleMoves) > 0,"Issue if there are not enough buildings to play"
        
        # If player has no intrigues, remove buildings 
        #   where you need to play an intrigue
        if currentPlayer.numIntrigues() == 0:
            for building in possibleMoves:
                if isinstance(building, Building) and building.playIntrigue:
                    possibleMoves.remove(building)

        # If player has insufficient gold, remove Builder's Hall
        if BUILDERS_HALL in possibleMoves:
            min_cost = min([building.cost for building in self.boardState.availableBuildings])
            if currentPlayer.resources.gold < min_cost:
                possibleMoves.remove(BUILDERS_HALL)
        

        # Choose a building to play an agent at
        # actionTypes["Choose a building to play an agent at (MAIN ACTION)"] += 1
        if len(possibleMoves) == 0:

            # Optionally complete a quest
            completableQuests = currentPlayer.completableQuests()
            if completableQuests:
                # actionTypes["Choose which completable quest to complete"] += 1
                move_idx = currentPlayer.selectMove(self, [DO_NOT_COMPLETE_QUEST] + completableQuests) 
                if move_idx > 0:
                    self.completeQuest(currentPlayer, completableQuests[move_idx - 1])
            
            return currentPlayer
         
        move_idx = currentPlayer.selectMove(self, possibleMoves) 
        building = possibleMoves[move_idx]
        self.boardState.buildings[building] = currentPlayer.name
        currentPlayer.agents -= 1

        buildingRewards,buildingQuests,buildingIntrigues = building.rewards.toResources()
        currentPlayer.getResources(buildingRewards)
        
        if isinstance(building, Building) and building.getCastle:
            currentPlayer.hasCastle = True 

        if isinstance(building, Building) and building.resetQuests:
            self.boardState.resetQuests()

        self.rewardIntrigues(currentPlayer, buildingIntrigues)

        if isinstance(building, CustomBuilding) and building.owner != currentPlayer.name:
            owner = self.namesToPlayers[building.owner]
            ownerRewardBundles = building.ownerRewards.split()
            if len(ownerRewardBundles) > 1:
                assert len(ownerRewardBundles) == 2,f"Owner rewards should only have either 1 or 2 options, but has {len(ownerRewardBundles)}"
                # actionTypes["Choose which owner reward to receive from a building"] += 1
                reward_idx = owner.selectMove(self, ownerRewardBundles)
            else:
                reward_idx = 0
            
            ownerRewards,ownerQuests,ownerIntrigues = ownerRewardBundles[reward_idx].toResources()
            owner.getResources(ownerRewards)
            assert ownerQuests == 0,"Owner should not receive quests from buildings"
            self.rewardIntrigues(owner, ownerIntrigues)

        # Secondary choices (quest, intrigue card)
        self.rewardQuests(currentPlayer, buildingQuests)
        
        if isinstance(building, Building) and building.playIntrigue:
            if currentPlayer.numIntrigues() == 0:
                raise ValueError(f'Player {currentPlayer.name} has no intrigue cards to play!')
            
            # actionTypes["Choose an intrigue card to play"] += 1
            playerUniqueIntrigues = currentPlayer.uniqueIntrigues()
            uniqueIntrigueIdx = currentPlayer.selectMove(self, playerUniqueIntrigues)
            intrigue = playerUniqueIntrigues[uniqueIntrigueIdx]
            currentPlayer.removeIntrigue(intrigue)
            self.playIntrigue(currentPlayer, intrigue)

        if isinstance(building, Building) and building.buyBuilding:
            # actionTypes["Choose which building from Builder's Hall to purchase"] += 1
            affordableBuildings = [building for building in self.boardState.availableBuildings
                                   if building.cost <= currentPlayer.resources.gold]
            assert len(affordableBuildings) > 0
            building_idx = currentPlayer.selectMove(self, affordableBuildings)
            building = affordableBuildings[building_idx]
            VPs,cost = self.boardState.purchaseBuilding(building, currentPlayer.name)
            currentPlayer.getResources(Resources(VPs=VPs))
            currentPlayer.removeResources(Resources(gold=cost))

        # Optionally complete a quest
        completableQuests = currentPlayer.completableQuests()
        if completableQuests:
            # actionTypes["Choose which completable quest to complete"] += 1
            move_idx = currentPlayer.selectMove(self, [DO_NOT_COMPLETE_QUEST] + completableQuests) 
            if move_idx > 0:
                self.completeQuest(currentPlayer, completableQuests[move_idx - 1])

    def runGame(self):
        '''Umbrella function to run the game.'''
        while self.roundsLeft > 0:
            # Keep looping until a player runs out of agents
            playerOutOfMoves = None
            while sum([player.agents for player in self.players]) > 0:
                e = self.takeTurn(self.players[0])
                # Reorder turn order to show that the player has moved.
                self.players = self.players[1:] + [self.players[0]]

                if e is not None:
                    # A player, `e`, could not make a move
                    if playerOutOfMoves == e:
                        # This same player could not make a move twice in a row
                        break # so we exit the loop of turns in the round
                    playerOutOfMoves = e

            # Reassign agents from waterdeep harbor
            waterdeepHarbors = utils.getWaterdeepHarbors(self.boardState.buildings)
            for waterdeepHarbor in waterdeepHarbors:
                occupier = self.boardState.buildings[waterdeepHarbor]
                if occupier is not None:
                    self.boardState.buildings[waterdeepHarbor] = REASSIGNED
                    player = self.namesToPlayers[occupier]
                    player.agents += 1
                    self.takeTurn(player)

            self.newRound()
        
        # end game code
        scores = []
        VPs = []
        for player in self.playersInitOrder:
            score = player.endGame()
            scores.append(score)
            VPs.append(player.resources.VPs)
        return scores, VPs

    def __repr__(self) -> str:
        return f"ROUNDS LEFT: {self.roundsLeft}\n\nBOARD STATE:\n{self.boardState}\nPLAYERS:" + "".join([f"{player}" for player in self.players])

def main(agents):
    gs = GameState(agents, numRounds=8)
    return gs.runGame()#,actionTypes