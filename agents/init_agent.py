from agents.baseline import manual_agent
from agents.rl import dqn,policy_gradient

def init_agent(agent_type: str = "Manual", **kwargs):
    match agent_type:
        case "Manual":
            return manual_agent.ManualAgent(**kwargs)
        case "DQN":
            return dqn.DQNAgent(**kwargs)
        case "PPO":
            return policy_gradient.PPOAgent(**kwargs)
        case _:
            raise ValueError(f"Invalid agent type: {agent_type}")