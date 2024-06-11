import numpy as np
import torch
import random
import matplotlib.pyplot as plt 
from tqdm import tqdm
import argparse

from agents.agent import Agent
from agents.rl.dqn_agent import DeepQNet, DQNAgent
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
    game_stats = dict()
    for igame in tqdm(range(train_ngames), desc='Training'):
        game = GameState(agents, numRounds=8)
        game.runGame() 
        if igame % eval_every == 0 or igame:
            game_stats[igame] = eval.eval(agents=agents, n_games=eval_ngames, verbose=True)
    return game_stats
    
def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):
    seed_all(args.seed)
    nPlayers = 2

    q_net = DeepQNet(
        input_dim=featurize.STATE_DIM,
        output_dim=featurize.ACTION_DIM,
        hidden_layer_sizes=[512,256,256,128],
        layernorm='layernorm',
        activation='LeakyReLU'
    )

    agents = [
        # DQNAgent(q_net),
        # DQNAgent(q_net),
        RandomAgent(),
        RandomAgent()
    ]
    assert len(agents) == nPlayers

    igames, vp_edges = train_and_eval(agents=agents, train_ngames=args.train_ngames, 
                                      eval_every=args.eval_every, eval_ngames=args.eval_ngames)
    print(vp_edges.shape)
    plt.figure()
    plt.plot(igames, vp_edges[:,0])
    plt.plot(igames, vp_edges[:,1])
    plt.show()

    final_game_stats = eval.eval(agents=agents, n_games=args.final_eval_ngames, verbose=True)
    print(final_game_stats)
    for agent in agents:
        print(agent.agent_type())
        print(agent.)
    # for key,ls in mean_stats.items():
    #     mean_stats[key] = [str(round(elt,2)) for elt in ls]

    # results_fname = f'results/dqn_vs_random_{args.train_ngames}games.txt'
    # with open(results_fname, 'w') as f:
    #     f.write("\n".join([
    #         "Agent types:\t\t" + ",\t".join(agentTypes),
    #         "Win rate:\t\t\t" + ",\t\t".join(mean_stats['wins']),
    #         "Mean scores:\t\t" + ",\t\t".join(mean_stats['scores']),
    #         "Mean VPs:\t\t\t" + ",\t\t".join(mean_stats['VPs']),
    #         "Mean score edge:\t" + ",\t\t".join(mean_stats['score edge']),
    #         "Mean VP edge:\t\t" + ",\t\t".join(mean_stats['VP edge']),
    #     ]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ngames", type=int, default=2500)
    parser.add_argument("--eval_ngames", type=int, default=250)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--final_eval_ngames", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1224)
    args = parser.parse_args()
    
    main(args)