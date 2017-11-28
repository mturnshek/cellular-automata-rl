import keras
import numpy as np

from two_player.playthrough import single_playthrough
from two_player.agent import Agent
from two_player.environment import Environment

def evaluate(agent):
    total_runs = 20
    current_run = 0
    wins = 0
    losses = 0

    env = Environment()
    test_agent = Agent(env, model=agent.model)

    while (current_run < total_runs):
        if np.random.random() > 0.5:
            red_agent = test_agent
            blue_agent = Agent(env)
        else:
            red_agent = Agent(env)
            blue_agent = test_agent

        if test_agent == single_playthrough(red_agent, blue_agent, env):
            wins += 1
        else:
            losses += 1
        env.reset()
        current_run += 1

    win_rate = wins/(wins + losses)
    return win_rate
