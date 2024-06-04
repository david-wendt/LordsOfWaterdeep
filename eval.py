import numpy as np 
from tqdm import tqdm

from game.game import GameState
from agents.agent import Agent

def setEval(agents: list[Agent]):
    for agent in agents:
        agent.eval()


STATS = [
    'wins', # True if won, False if lost
    'scores', # Final score at the end of the game
    'VPs', # Final number of VPs at the end of the game
    'score edge', # Difference between best player and you 
    'VP edge',    # (or you and second best, if you won)
]

def appendStats(stats, scores, VPs, n_agents):
    # Collect stats
    for iagent in range(n_agents):
        stats['scores'][iagent].append(scores[iagent])
        stats['VPs'][iagent].append(VPs[iagent])
        
        if iagent == np.argmax(VPs):
            stats['wins'][iagent].append(True)
            stats['scores'][iagent].append(scores[iagent])
            stats['VPs'][iagent].append(VPs[iagent])

            otherVPs = VPs.copy()
            otherVPs.pop(iagent)
            stats['VP edge'][iagent].append(VPs[iagent] - np.max(otherVPs))
        else:
            stats['wins'][iagent].append(False) 
            stats['scores'][iagent].append(scores[iagent])
            stats['VPs'][iagent].append(VPs[iagent])
            stats['VP edge'][iagent].append(VPs[iagent] - np.max(VPs))


        if iagent == np.argmax(scores):
            otherScores = scores.copy()
            otherScores.pop(iagent)
            stats['score edge'][iagent].append(scores[iagent] - np.max(otherScores))
        else:
            stats['score edge'][iagent].append(scores[iagent] - np.max(scores))


def eval(agents: list[Agent], n_games: int = 100, verbose=False):
    setEval(agents)

    n_agents = len(agents)
    stats = {
        statname: {
            iagent: []
            for iagent in range(n_agents)
        }
        for statname in STATS
    }

    for igame in tqdm(range(n_games), desc='Evaluation', disable=not verbose):
        game = GameState(agents, numRounds=8)
        scores,VPs = game.runGame() 
        # print(scores)
        appendStats(stats, scores, VPs, n_agents)
    
    mean_stats = {
    statname: [
            np.mean(stats[statname][iagent]) 
            for iagent in range(len(agents))
        ]
        for statname in STATS
    }
    return mean_stats