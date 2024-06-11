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
from agents.rl.dqn_agent import DQNAgent
from agents.rl.dqnet import DeepQNet
from agents.rl.policy_agent import PolicyAgent
from agents.baseline.random_agent import RandomAgent
from game.game import GameState
from features import featurize
import eval 

def main(args):
    policyAgent = PolicyAgent(
        state_dim=featurize.STATE_DIM, 
        action_dim=featurize.ACTION_DIM
    )
    randomAgent = RandomAgent()
    agents2 = [policyAgent, randomAgent]

    # eval_results = []
    eval_results_2 = []
    for iteration in range(40):
    #    mean_stats = eval.eval(agents=agents1, n_games=args.eval_ngames, verbose=True)
    #    eval_results.append(mean_stats)
        mean_stats_2 = eval.eval(agents=agents2, n_games=args.eval_ngames, verbose=True)
        eval_results_2.append(mean_stats_2)
    #    train(agents=agents1, n_games=100)
        train(agents=agents2, n_games=100)
    #mean_stats = eval.eval(agents=agents1, n_games=args.eval_ngames, verbose=True)
    #eval_results.append(mean_stats)
    mean_stats_2 = eval.eval(agents=agents2, n_games=args.eval_ngames, verbose=True)
    eval_results_2.append(mean_stats_2)

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
            if agent % 2 == 0:
                data['games_played'].append(iteration * 100)
                data['agent'].append("DQN")
                data['wins'].append(float(stats[0]))
                data['VPs'].append(float(stats[2]))
                data['scores'].append(float(stats[1]))
                data['score_edge'].append(float(stats[3]))
                data['VP_edge'].append(float(stats[4]))
    for iteration, result in enumerate(eval_results_2):
        for agent, stats in enumerate(zip(*result.values())):
            if agent % 2 == 0:
                data['games_played'].append(iteration * 100)
                data['agent'].append("REINFORCE")
                data['wins'].append(float(stats[0]))
                data['VPs'].append(float(stats[2]))
                data['scores'].append(float(stats[1]))
                data['score_edge'].append(float(stats[3]))
                data['VP_edge'].append(float(stats[4]))
    df = pd.DataFrame(data)
    df.to_csv('evaluation_results_larger_replay18.csv', index=False)

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