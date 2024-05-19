from random import shuffle
from game.game_info import *
from game.player import Player
from game.board import BoardState
from agents import init_agent

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
    def __init__(self, numPlayers: int = 3, numRounds: int = 8, 
                 playerNames=None, playerAgents=None):
        '''
        Initialize the game state and players.

        Args: 
            numPlayers: the number of players in the game
            numRounds: the number of rounds in the game
            playerNames (optional): the names for each player
        '''
        # Initialize the remaining number of rounds
        self.roundsLeft = numRounds

        # Initialize the BoardState
        self.boardState = BoardState()

        # Check that we have a valid number of players
        assert 2 <= numPlayers <= 5
        self.numPlayers = numPlayers

        # Set default player names
        if playerNames is None:
            self.playerNames = [
                "PlayerOne", "PlayerTwo", 
                "PlayerThree", "PlayerFour",
                "PlayerFive"
            ][:numPlayers]
        else:
            assert len(playerNames) == self.numPlayers
            self.playerNames = playerNames

        if len(self.playerNames) != len(set(self.playerNames)):
            raise ValueError("Need to have unique player names")

        if playerAgents is None:
            self.playerAgents = ["Manual"] * self.numPlayers
        else:
            assert len(playerAgents) == self.numPlayers
            self.playerAgents = playerAgents
        
        # Shuffle the lord cards
        shuffled_lord_cards = LORD_CARDS.copy()
        shuffle(shuffled_lord_cards)

        # Initialize the players
        self.players = [Player(
            name=self.playerNames[i],
            agent=init_agent.init_agent(self.playerAgents[i]),
            numAgents=agentsPerPlayer(numPlayers),
            lordCard=shuffled_lord_cards[i]
        ) for i in range(numPlayers)]

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

        # Finally, start a new round (at this stage, just 
        # decrements roundsLeft and places VPs on
        # buildings at builder's hall)
        self.newRound()

    def newRound(self):
        '''Reset the board at the beginning of each round.'''
        self.roundsLeft -= 1
        # TODO (later version): put VPs on buildings at bulider's hall

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

    def takeTurn(self):
        '''Take a single turn in the turn order.'''
        # TODO: Implement 
        currentPlayer = self.players[0]
        possibleMoves = [building for building in self.boardState.buildings 
                         if building.occupier is None] 
        move_idx = currentPlayer.selectMove(self, possibleMoves) # Implement this
        building = possibleMoves[move_idx]

        # Secondary choices (quest, intrigue card)
        if building.rewards.quests > 0:
            for _ in range(building.rewards.quests):
                raise NotImplementedError("Choose a quest")
            building.rewards.quests = 0

        if building.rewards.intrigues > 0:
            for _ in range(building.rewards.intrigues):
                raise NotImplementedError("Choose an intrigue")
            building.rewards.intrigues = 0
        
        if building.specialRewards == "Play intrigue": # maybe make this a list of str intead of str
            raise NotImplementedError("Choose an intrigue to play")
        
        completableQuests = currentPlayer.completableQuests()
        if completableQuests:
            move_idx = currentPlayer.selectMove(self, completableQuests) # Implement this

        # Reorder turn order to show that the player has moved.
        self.players = self.players[1:] + [currentPlayer]

    def runGame(self):
        '''Umbrella function to run the game.'''
        while self.roundsLeft > 0:
            # Keep looping until a player runs out of agents
            while self.players[0].agents >= 0:
                self.takeTurn()

            # TODO: reorder the players if one of them picked up the castle.
            self.newRound()

    def displayGame(self) -> None:
        '''Display the state of the game and players.'''
        print(self.boardState)
        print("\nPLAYERS:")
        for player in self.players:
            print(player)     

def main():
    # Test gameState
    gs = GameState()
    print('initialized gs:')
    print(gs)
    gs.displayGame()
    gs.takeTurn()
    gs.displayGame()