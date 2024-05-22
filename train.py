import numpy as np

from agents.rl.dqn import DeepQNet, DQNAgent
from agents.baseline.random_agent import RandomAgent
from game.game import GameState

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

            otherScores = scores.copy()
            otherScores.pop(iagent)
            otherVPs = VPs.copy()
            otherVPs.pop(iagent)
            stats['score edge'][iagent].append(scores[iagent] - np.max(otherScores))
            stats['VP edge'][iagent].append(VPs[iagent] - np.max(otherVPs))
        else:
            stats['wins'][iagent].append(False) 
            stats['scores'][iagent].append(scores[iagent])
            stats['VPs'][iagent].append(VPs[iagent])
            stats['score edge'][iagent].append(scores[iagent] - np.max(scores))
            stats['VP edge'][iagent].append(VPs[iagent] - np.max(VPs))

def train(agents, n_games):
    n_agents = len(agents)
    stats = {
        statname: {
            iagent: []
            for iagent in range(n_agents)
        }
        for statname in STATS
    }

    for igame in range(n_games):
        game = GameState(agents, numRounds=4)
        scores,VPs = game.runGame() 
        print(scores)
        # TODO: Implement some logic in runGame
        # to save stats to agents
        appendStats(stats, scores, VPs, n_agents)
    
    mean_stats = {
    statname: [
            str(np.mean(stats[statname][iagent])) 
            for iagent in range(len(agents))
        ]
        for statname in STATS
    }
    print(mean_stats)
    return mean_stats

    
def main():
    deepQAgent = DQNAgent(
        DeepQNet(observation_dim=100, n_actions=10), # fix these as constants computed in featurize
        eps_start=.8, # Maybe set some defaults for these?
        eps_decay=1000,
        n_actions=10
    )

    randomAgent = RandomAgent()

    agentTypes = ['Deep Q Agent', 'Random Agent']

    agents = [deepQAgent, randomAgent]
    n_games = 100
    mean_stats = train(agents=agents, n_games=n_games)

    results_fname = f'dqn_vs_random_{n_games}games.txt'
    with open(results_fname, 'w') as f:
        f.writelines([
            "Agent types: " + ", ".join(agentTypes),
            "Win rate: " + ", ".join(mean_stats['wins']),
            "Mean scores: " + ", ".join(mean_stats['scores']),
            "Mean VPs: " + ", ".join(mean_stats['VPs']),
            "Mean score edge: " + ", ".join(mean_stats['score edge']),
            "Mean VP edge: " + ", ".join(mean_stats['VP edge']),
        ])

if __name__ == "__main__":
    main()