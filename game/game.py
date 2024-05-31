from random import shuffle

from game.game_info import *
from game.player import Player
from game import board
from game import utils 
from agents.agent import Agent

# Class to control the flow of the game, focused on turn progression and move 
# options. Broadly, this class handles anything involving the game state and
# the players, while those other classes handle either only the game state
# itself or only the players themselves (to the extent possible).

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

        # TODO (later version): put new resources on buildings that need them

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

    def playIntrigue(self, currentPlayer: Player, intrigue: str):
        assert intrigue in INTRIGUES
        opponents = [player for player in self.players if player != currentPlayer]
        if intrigue == 'Call in a Favor':
            resource_options = STANDARD_RESOURCE_BUNDLES
            resource_idx = currentPlayer.selectMove(self, resource_options)
            currentPlayer.getResources(resource_options[resource_idx])
        elif intrigue in ['Lack of Faith', 'Ambush', 'Assassination', 'Arcane Mishap']:
            # TODO: select resources based on intrigue card name
            lostResources = Resources(...)
            gainedResources = Resources(...)
            n_rewards = self.removeFromOpponents(currentPlayer, lostResources)
            currentPlayer.getResources(n_rewards * gainedResources)

            raise NotImplementedError('Every opponent removes x, you gain y for each that could not')
        elif intrigue == 'Free Drinks':
            # Idea: first look at set of adventurers that at least 1 opponent has,
            # then choose resource from this list,
            # then choose opponent to steal it from
            valid_opponents = ...
            opponent_idx = currentPlayer.selectMove(self, valid_opponents, "Take from opponent")
            raise NotImplementedError("choose 1 opp, steal 1 adventurer")
        elif intrigue in ['Spread the Wealth', 'Graduation Day', 'Conscription', 'Good Faith', 'Crime Wave']:
            # TODO: Switch case based on intrigue card
                # Give resource to player within switching
                # Define resource to be given to opponent
            opponent_idx = currentPlayer.selectMove(self, opponents, "Give to opponent")
            raise NotImplementedError("Get a resource, choose 1 opp to get a resource")
        elif intrigue == 'Call for Adventurers':
            # TODO: Choose which adventurer you want, and get two
            for opponent in opponents:
                raise NotImplementedError # Let each opp select a adventurer, and then give them 1
            raise NotImplementedError
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
        # ^^ This will happen without builder's hall in a 5-player game,
        # or in any other game after round 4
        
        # If player has no intrigues, remove buildings 
        #   where you need to play an intrigue
        if len(currentPlayer.intrigues) == 0:
            for building in possibleMoves:
                if building.playIntrigue:
                    possibleMoves.remove(building)

        # Choose a building to play an agent at
        move_idx = currentPlayer.selectMove(self, possibleMoves) 
        building = possibleMoves[move_idx]
        self.boardState.buildings[building] = currentPlayer.name
        currentPlayer.agents -= 1

        currentPlayer.getResources(building.rewards.toResources())
        
        if isinstance(building, Building) and building.getCastle:
            currentPlayer.hasCastle = True 

        if isinstance(building, Building) and building.resetQuests:
            self.boardState.resetQuests()

        if building.rewards.intrigues > 0:
            for _ in range(building.rewards.intrigues):
                currentPlayer.getIntrigue(self.boardState.drawIntrigue())

        if isinstance(building, CustomBuilding) and building.owner != currentPlayer.name:
            owner = self.namesToPlayers[building.owner]
            ownerRewards = building.ownerRewards.split()
            if len(ownerRewards) > 1:
                reward_idx = owner.selectMove(self, ownerRewards)
            else:
                reward_idx = 0
            owner.getResources(ownerRewards[reward_idx])

        # Secondary choices (quest, intrigue card)
        if building.rewards.quests > 0:
            for _ in range(building.rewards.quests):
                quest_idx = currentPlayer.selectMove(self, self.boardState.availableQuests)
                quest = self.boardState.chooseQuest(quest_idx)
                currentPlayer.getQuest(quest)
        
        if isinstance(building, Building) and building.playIntrigue:
            if len(currentPlayer.intrigues) == 0:
                raise ValueError(f'Player {currentPlayer.name} has no intrigue cards to play!')
            
            intrigue_idx = currentPlayer.selectMove(self, currentPlayer.intrigues)
            intrigue = currentPlayer.intrigues.pop(intrigue_idx)
            self.playIntrigue(currentPlayer, intrigue)

        if isinstance(building, Building) and building.buyBuilding:
            building_idx = currentPlayer.selectMove(self, self.boardState.availableBuildings)
            self.boardState.purchaseBuilding(building_idx, currentPlayer.name)

        # Optionally complete a quest
        completableQuests = currentPlayer.completableQuests()
        if completableQuests:
            move_idx = currentPlayer.selectMove(self, [DO_NOT_COMPLETE_QUEST] + completableQuests) 
            if move_idx > 0:
                currentPlayer.completeQuest(completableQuests[move_idx - 1])

    def runGame(self, verbose=False):
        '''Umbrella function to run the game.'''
        while self.roundsLeft > 0:
            # Keep looping until a player runs out of agents
            while sum([player.agents for player in self.players]) > 0:
                self.takeTurn(self.players[0])
                # Reorder turn order to show that the player has moved.
                self.players = self.players[1:] + [self.players[0]]

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

            if verbose:
                print(self)
        
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
    gs = GameState(agents, numRounds=4)
    gs.runGame(False)