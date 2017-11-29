import numpy as np

from create_pattern.environment import Environment
from create_pattern.agent import Agent

pattern = np.zeros((6, 6, 3), dtype='bool')
pattern[0, :, 2] = 1
pattern[1, :, 2] = 1
pattern[2, :, 2] = 1
print(pattern)

env = Environment()
agent = Agent(env)
agent.train(pattern)
