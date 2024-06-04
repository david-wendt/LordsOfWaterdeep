import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
import argparse

from agents.agent import Agent
from agents.rl.dqn import DeepQNet, DQNAgent
from agents.baseline.random_agent import RandomAgent
from game.game import GameState
from features import featurize
import eval 

def setTrain(agents: list[Agent]):
    for agent in agents:
        agent.train()

def train(agents, n_games):
    setTrain(agents)
    for _ in tqdm(range(n_games), desc='Training'):
        game = GameState(agents, numRounds=8)
        game.runGame() 

def train_and_eval(agents, train_ngames, eval_every=200, eval_ngames=100):
    setTrain(agents)
    igames = []
    vp_edges = []
    for igame in tqdm(range(train_ngames), desc='Training'):
        game = GameState(agents, numRounds=8)
        game.runGame() 
        if igame % eval_every == 0:
            mean_stats = eval.eval(agents=agents, n_games=eval_ngames, verbose=True)
            igames.append(igame)
            vp_edges.append(mean_stats['VP edge'])
    return igames, np.array(vp_edges)
    
def main(args):
    nPlayers = 2
    deepQAgent = DQNAgent()

    randomAgent = RandomAgent()

    agentTypes = ['Deep Q Agent', 'Random Agent']
    agents = [deepQAgent, randomAgent]
    assert len(agents) == len(agentTypes) == nPlayers

    igames, vp_edges = train_and_eval(agents=agents, train_ngames=args.train_ngames)
    print(vp_edges.shape)
    plt.figure()
    plt.plot(igames, vp_edges[:,0])
    plt.plot(igames, vp_edges[:,1])
    plt.show()

    mean_stats = eval.eval(agents=agents, n_games=args.eval_ngames, verbose=True)

    results_fname = f'results/dqn_vs_random_{args.train_ngames}games.txt'
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ngames", type=int, default=int(1e3))
    parser.add_argument("--eval_ngames", type=int, default=int(1e2))
    args = parser.parse_args()
    
    main(args)