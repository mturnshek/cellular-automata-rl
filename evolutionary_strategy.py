from keras.models import clone_model
import numpy as np

from environment import Environment
from agent import Agent


def spawn_new_model(agent):
    # copy the model from the given agent
    model = clone_model(agent.model)

    # add gaussian noise to parameters at random
    noise_sd = 0.01 # kind of like learning rate

    weights = np.array(model.get_weights())
    noise = np.random.normal(scale=noise_sd, size=weights.shape)
    model.set_weights(np.add(weights, noise))

    return model


def single_playthrough(red_agent, blue_agent, env):
    max_moves = 1000
    move_count = 0
    while not env.is_done():
        if env.turn == 'red':
            cell = red_agent.act(env.state())
        else:
            cell = blue_agent.act(env.state())
        env.step(cell)
        move_count += 1
        if move_count > max_moves:
            break

    if env.reward('red') == 1:
        print('Red wins')
        return red_agent
    else:
        print('Blue wins')
        return blue_agent


def evolution(save_path='weights/lineage1.h5'):
    env = Environment()
    red_agent = Agent(env)
    blue_agent = Agent(env)

    playthroughs = 10000

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

    winner.model.save_weights(save_path)
