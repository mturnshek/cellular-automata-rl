from keras.models import clone_model
import numpy as np

from two_player.environment import Environment
from two_player.agent import Agent
from two_player.evaluate import evaluate
from two_player.playthrough import single_playthrough


def spawn_new_model(agent):
    # copy the model from the given agent
    model = clone_model(agent.model)

    # add gaussian noise to parameters at random
    noise_sd = 0.01 # kind of like learning rate

    weights = np.array(model.get_weights())
    noise = np.random.normal(scale=noise_sd, size=weights.shape)
    model.set_weights(np.add(weights, noise))

    return model


def evolution(save_path='weights/lineage1.h5'):
    env = Environment()
    red_agent = Agent(env)
    blue_agent = Agent(env)

    playthroughs = 10000
    evaluate_period = 100

    for i in range(playthroughs):
        winner = single_playthrough(red_agent, blue_agent, env)
        if np.random.random() < .5:
            red_agent = winner
            blue_agent = Agent(env, model=spawn_new_model(winner))
        else:
            blue_agent = winner
            red_agent = Agent(env, model=spawn_new_model(winner))
        env.reset()
        print('playthrough', i)

        if (i % evaluate_period == 0):
            print('evaluating top agent for 20 playthroughs... ...')
            print("win rate vs. random agent", evaluate(winner))

    winner.model.save_weights(save_path)
