import numpy as np 
from tqdm import tqdm

from game.game import GameState
from agents.agent import Agent

def setEval(agents: list[Agent]):
    for agent in agents:
        agent.eval()


STATS = [
    'wins', # True if won, False if lost
    'VPs', # Final number of VPs at the end of the game
    'winner VPs' # VPs if you were the winner
]

def appendStats(stats, VPs, n_agents):
    # Collect stats
    for iagent in range(n_agents):
        stats['VPs'][iagent].append(VPs[iagent])
        
        if iagent == np.argmax(VPs):
            stats['wins'][iagent].append(True)
            stats['winner VPs'][iagent].append(VPs[iagent])
        else:
            stats['wins'][iagent].append(False) 


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
        appendStats(stats, VPs, n_agents)
    
    mean_stats = [
        {
            'win rate': np.mean(stats['wins'][iagent]),
            'mean VPs': np.mean(stats['VPs'][iagent]),
            'std VPs': np.std(stats['VPs'][iagent]),
            'mean winner VPs': np.mean(stats['winner VPs'][iagent]),
            'std winner VPs': np.std(stats['winner VPs'][iagent])
        }
        for iagent in range(len(agents))
    ]

    return mean_stats