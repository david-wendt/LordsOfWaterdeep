import numpy as np

from game.game import GameState
from agents.baseline.manual_agent import ManualAgent
from agents.baseline.random_agent import RandomAgent
from agents.baseline.strategic_agent import BasicStrategicAgent, AntiStrategicAgent
from agents.rl.dqn import DQNAgent
from game import quests
import printing_utils 
import eval

def main():
    # agents = [ManualAgent(), RandomAgent()]
    # agents = [BasicStrategicAgent(), RandomAgent()]
    # agents = [AntiStrategicAgent(), RandomAgent()]
    # agents = [RandomAgent(), RandomAgent()]
    agents = [DQNAgent(), RandomAgent()]
    eval.setEval(agents)
    wins = []
    ties = []
    edges = []
    for i in range(1000):
        # print(i)
        game = GameState(agents, numRounds=8)
        scores, vps = game.runGame()
        if i % 100 == 0:
            print(i,scores, vps)
            # printing_utils.pprint(actionTypes)

        wins.append(vps[0] > vps[1])
        ties.append(vps[0] == vps[1])
        edges.append(vps[0] - vps[1])
    print(sum(wins) / len(wins))
    print(np.mean(edges), np.std(edges), np.mean(edges) / np.std(edges))
    # print(sum(ties) / len(ties))

if __name__ == '__main__':
    # quests.main()
    main()
    # game.main()