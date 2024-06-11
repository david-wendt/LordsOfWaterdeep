import numpy as np
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt 
from tqdm import tqdm
import argparse

from agents.agent import Agent
from agents.rl.dqn_agent import DQNAgent
from agents.rl.dqnet import DeepQNet
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
    stats = dict()
    for igame in tqdm(range(train_ngames), desc='Training'):
        game = GameState(agents, numRounds=8)
        game.runGame() 
        if igame % eval_every == 0:
            stats[igame] = pd.DataFrame(eval.eval(agents=agents, n_games=eval_ngames, verbose=False))

    return stats
    
def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):
    seeds = [int(seed) for seed in args.seeds.split(',')]
    dfs = {}
    for seed in seeds:
        print('Using seed', seed)
        seed_all(seed)
        nPlayers = 2

        q_net = DeepQNet(
            input_dim=featurize.STATE_DIM[nPlayers],
            output_dim=featurize.ACTION_DIM[nPlayers],
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
        agent_types = [agent.agent_type() for agent in agents]

        stats = train_and_eval(agents=agents, train_ngames=args.train_ngames, 
                                        eval_every=args.eval_every, eval_ngames=args.eval_ngames)
        
        final_stats = eval.eval(agents=agents, n_games=args.final_eval_ngames, verbose=True)
        stats.update({args.train_ngames: pd.DataFrame(final_stats)})

        df = pd.concat(stats)
        df.index = df.index.set_names(['train games', 'agent index'])
        df.reset_index(inplace=True)
        dfs[seed] = df
        df.to_csv(f'results/training/{"-".join(agent_types)}_{args.expname}_seed{seed}.csv')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ngames", type=int, default=2500)
    parser.add_argument("--eval_ngames", type=int, default=250)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--final_eval_ngames", type=int, default=1500)
    parser.add_argument("--seeds", type=str, default="1224")
    parser.add_argument("--expname", type=str, default='default')
    args = parser.parse_args()
    
    main(args)