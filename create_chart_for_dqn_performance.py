import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


episode_rewards_small = np.load('logs/grow_dqn_small_performance.npy')
episode_rewards_low_high_low = np.load('logs/grow_dqn_low_high_low_performance.npy')
episode_rewards_deep = np.load('logs/grow_dqn_deep_performance.npy')

df_small = pd.DataFrame(episode_rewards_small)
df_low_high_low = pd.DataFrame(episode_rewards_low_high_low)
df_deep = pd.DataFrame(episode_rewards_deep)


x = np.array(list(range(len(episode_rewards_small))))
x = 200 * x # 200 epochs of training per episode reward log

plt.plot(x, episode_rewards_small, color='blue', linewidth=2)
plt.plot(x, episode_rewards_low_high_low, color='red', linewidth=2)
plt.plot(x, episode_rewards_deep, color='green', linewidth=2)
plt.ylabel('episode reward')
plt.xlabel('epochs')
plt.show()
