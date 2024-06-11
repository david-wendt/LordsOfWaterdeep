import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train import train
from eval import eval
import numpy as np
from tqdm import tqdm
import argparse
from agents.agent import Agent
from agents.rl.dqn_agent import DeepQNet, DQNAgent
from agents.rl.policy_agent import PolicyAgent
from agents.baseline.random_agent import RandomAgent
from game.game import GameState
from features import featurize
import eval 

def main(args):
    deepQAgent = DQNAgent(
        state_dim=1224, 
        action_dim=featurize.ACTION_DIM
    )
    policyAgent = PolicyAgent(
        state_dim=1224, 
        action_dim=featurize.ACTION_DIM
    )
    randomAgent = RandomAgent()
    randomAgent2 = RandomAgent()
    # agentTypes = ['Deep Q Agent', 'Random Agent']
    agents = [deepQAgent, policyAgent, randomAgent, randomAgent2]

    eval_results = []
    eval_results_2 = []
    for iteration in range(2):
        mean_stats = eval.eval(agents=agents, n_games=args.eval_ngames, verbose=True)
        eval_results.append(mean_stats)
        train(agents=agents, n_games=5)
    mean_stats = eval.eval(agents=agents, n_games=args.eval_ngames, verbose=True)
    eval_results.append(mean_stats)

    data = {
        'games_played': [],
        'agent': [],
        'wins': [],
        'VPs': [],
        'scores': [],
        'score_edge': [],
        'VP_edge': []
    }

    for iteration, result in enumerate(eval_results):
        for agent, stats in enumerate(zip(*result.values())):
                if agent < 2:
                    data['games_played'].append(iteration * 100)
                    if agent == 0:
                        data['agent'].append("DQN")
                    if agent == 1:
                        data['agent'].append("REINFORCE")
                    data['wins'].append(float(stats[0]))
                    data['VPs'].append(float(stats[2]))
                    data['scores'].append(float(stats[1]))
                    data['score_edge'].append(float(stats[3]))
                    data['VP_edge'].append(float(stats[4]))
    df = pd.DataFrame(data)
    df.to_csv('evaluation_results_4.csv', index=False)

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='games_played', y='wins', hue='agent', marker='o')
    plt.title('Win Rate over Training Games')
    plt.xlabel('Training Games')
    plt.ylabel('Win Rate')
    plt.legend(title='Agent')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='games_played', y='VPs', hue='agent', marker='o')
    plt.title('VPs over Training Games')
    plt.xlabel('Training Games')
    plt.ylabel('VPs')
    plt.legend(title='Agent')
    plt.show()
    nPlayers = 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_ngames", type=int, default=int(1e2))
    args = parser.parse_args()
    
    main(args)