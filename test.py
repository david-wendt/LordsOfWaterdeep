from game.player import Player
from agents.baseline.random_agent import RandomAgent
from agents.baseline.manual_agent import ManualAgent

if __name__ == "__main__":
    players = [
        Player('Joey', RandomAgent(), 3, ('hello', 'there')),
        Player('Bay', ManualAgent(), 3, ('general', 'kenobi')),
    ]

    joey = players[0]
    
    for player in players.copy():
        print(player.name == joey.name)
        print(player == joey)