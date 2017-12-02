import numpy as np

from create_pattern.environment import Environment
from create_pattern.agent import Agent

pattern = np.zeros((10, 10, 3), dtype='bool')
pattern[1, 1, 1] = 1
pattern[1, 2, 1] = 1
pattern[2, 1, 1] = 1
pattern[2, 2, 1] = 1

print(pattern)

env = Environment(display=True)
agent = Agent(env, load=False)
agent.train(pattern)
