from game import game
from agents.baseline.manual_agent import ManualAgent
from agents.baseline.random_agent import RandomAgent
from game import quests

def main():
    agents = [RandomAgent(), RandomAgent()]
    # agents = [ManualAgent(), RandomAgent()]
    wins = []
    ties = []
    for i in range(100):
        results = game.main(agents)
        if i % 10 == 0:
            print(results)

        vps = results[1]
        wins.append(vps[0] > vps[1])
        ties.append(vps[0] == vps[1])
    print(sum(wins) / len(wins))
    print(sum(ties) / len(ties))

if __name__ == '__main__':
    quests.main()