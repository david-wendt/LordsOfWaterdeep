from game import game
from agents.baseline.manual_agent import ManualAgent
from agents.baseline.random_agent import RandomAgent
from game import quests
import printing_utils 

def main():
    # agents = [ManualAgent(), RandomAgent()]
    agents = [RandomAgent(), RandomAgent()]
    wins = []
    ties = []
    for i in range(10000):
        # results,actionTypes = game.main(agents)
        results = game.main(agents)
        if i % 1000 == 0:
            print(i,results)
            # printing_utils.pprint(actionTypes)

        vps = results[1]
        wins.append(vps[0] > vps[1])
        ties.append(vps[0] == vps[1])
    print(sum(wins) / len(wins))
    print(sum(ties) / len(ties))

if __name__ == '__main__':
    # quests.main()
    main()
    # game.main()