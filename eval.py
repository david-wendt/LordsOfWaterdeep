import numpy as np 
import matplotlib.pyplot as plt 
import os 
from tqdm import tqdm
from collections import defaultdict

from game.game import GameState
from agents.agent import Agent
from agents.baseline.random_agent import RandomAgent

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
    
    return [
        dict({
            'eval games': n_games,
            'agent type': agent.agent_type(),
            'win rate': np.mean(stats['wins'][iagent]),
            'mean VPs': np.mean(stats['VPs'][iagent]),
            'std VPs': np.std(stats['VPs'][iagent]),
            'mean winner VPs': np.mean(stats['winner VPs'][iagent]),
            'std winner VPs': np.std(stats['winner VPs'][iagent]),
        }, **agent.getStats(),
        )
        for iagent,agent in enumerate(agents)
    ]

if __name__ == "__main__":

    agents = [RandomAgent(), RandomAgent(), RandomAgent(), RandomAgent()]
    setEval(agents)
    ngames = 1000
    all_vps = []
    winner_vps = []
    winner_vp_edges = []
    vp_spreads = []
    for i in range(ngames):
        # print(i)
        game = GameState(agents, numRounds=8)
        scores, vps = game.runGame()
        assert len(vps) == len(agents)

        winner_n_vps = max(vps)
        other_vps = vps.copy()
        other_vps.remove(winner_n_vps)
        winner_vp_edges.append(winner_n_vps - max(other_vps))
        vp_spreads.append(winner_n_vps - min(vps))

        for agent_vps in vps:
            if agent_vps == winner_n_vps:
                # Agent is winner
                winner_vps.append(agent_vps)
            all_vps.append(agent_vps)
            
        if i % 100 == 0:
            print(i,scores, vps)
            # printing_utils.pprint(actionTypes)

    for agent in agents:
        print(agent.getStats(ngames))

    save_dir = f'results/random/{len(agents)}p'
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.title('VPs')
    plt.hist(all_vps, label='all players', density=True, bins=25)
    plt.hist(winner_vps, label='winners players', histtype='step', density=True, bins=25)
    plt.savefig(os.path.join(save_dir, 'VPs.png'))
    plt.close()

    plt.figure()
    plt.title('Winner VP edges')
    plt.hist(winner_vp_edges, density=True, bins=25)
    plt.savefig(os.path.join(save_dir, 'winner_VP_edges.png'))
    plt.close()

    plt.figure()
    plt.title('VP spreads')
    plt.hist(vp_spreads, density=True, bins=25)
    plt.savefig(os.path.join(save_dir, 'VP_spreads.png'))
    plt.close()