import numpy as np
from tqdm import tqdm

from agents.rl.dqn import DeepQNet, DQNAgent
from agents.baseline.random_agent import RandomAgent
from game.game import GameState
from features import featurize

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

def train(agents, n_games, verbose=False):
    n_agents = len(agents)
    stats = {
        statname: {
            iagent: []
            for iagent in range(n_agents)
        }
        for statname in STATS
    }

    for igame in tqdm(range(n_games), disable=not verbose):
        game = GameState(agents, numRounds=4)
        scores,VPs = game.runGame() 
        # print(scores)
        appendStats(stats, scores, VPs, n_agents)
    
    mean_stats = {
    statname: [
            str(round(np.mean(stats[statname][iagent]),2)) 
            for iagent in range(len(agents))
        ]
        for statname in STATS
    }
    print(mean_stats)
    return mean_stats

    
def main():
    nPlayers = 2
    stateDim = featurize.stateDim(nPlayers)
    deepQAgent = DQNAgent(
        state_dim=stateDim, 
        action_dim=featurize.N_ACTIONS
    )

    randomAgent = RandomAgent()

    agentTypes = ['Deep Q Agent', 'Random Agent']
    agents = [deepQAgent, randomAgent]
    assert len(agents) == len(agentTypes) == nPlayers

    n_games = 1000
    mean_stats = train(agents=agents, n_games=n_games, verbose=True)

    results_fname = f'dqn_vs_random_{n_games}games.txt'
    with open(results_fname, 'w') as f:
        f.write("\n".join([
            "Agent types:\t\t" + ",\t".join(agentTypes),
            "Win rate:\t\t\t" + ",\t\t".join(mean_stats['wins']),
            "Mean scores:\t\t" + ",\t\t".join(mean_stats['scores']),
            "Mean VPs:\t\t\t" + ",\t\t".join(mean_stats['VPs']),
            "Mean score edge:\t" + ",\t\t".join(mean_stats['score edge']),
            "Mean VP edge:\t\t" + ",\t\t".join(mean_stats['VP edge']),
        ]))

if __name__ == "__main__":
    main()