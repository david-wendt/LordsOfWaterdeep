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

def getAgentStats(agents: list[Agent], n_games: int):
    stats = defaultdict(list)
    for agent in agents:
        agent_stats = agent.getStats()
        for key,stat in agent_stats.items():
            stats[key].append(stat / n_games)
    return stats

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