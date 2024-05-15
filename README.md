# LordsOfWaterdeep
Custom implementation of Lords of Waterdeep, and RL approaches to playing the game

# Organization

`main.py`: main function to run the game and train agents

`baseline_agents`: dir containing implementations of non-RL baseline agents

`rl_agents`: dir containing RL implementations of agents

`features`: dir containing code for featurization (or any other intermediaries
between RL code in `agents` and implementation code in `game`)

`game`: dir containing LoW implementation 