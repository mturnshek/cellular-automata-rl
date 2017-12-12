import numpy as np

from keras.models import Model
from keras.layers import Dense, Reshape, Flatten, Input, Conv2D, concatenate, GaussianNoise, BatchNormalization
from keras.optimizers import RMSprop

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.processors import MultiInputProcessor

from grow.environment import Environment


def create_input(shape):
    """ Doing this to create the model inputs, due to the extra dimension
        keras-rl interprets at the beginning of observations """
    unprocessed_input = Input((1,) + shape)
    return Reshape(shape)(unprocessed_input)


class DQN_Agent():
    def __init__(self, display=False):
        self.env = Environment(display=display)
        self.nb_actions = self.env.rows*self.env.cols + 1

        self.model_save_path = 'weights/grow_dqn_small_weights.h5'
        self.log_save_path = 'logs/grow_dqn_small_performance.npy'
        self.episode_reward_log = []

        self.create_model()
        self.initialize_dqn()

    def create_model(self):
        board_shape = self.env.x1_shape
        move_tile_counter_shape = self.env.x2_shape
        round_counter_shape = self.env.x3_shape

        # Keras-rl interprets an extra dimension at axis=0
        # tacked on to our observations, which we take into account here
        board_input = Input((1,) + board_shape)
        move_tile_counter_input = Input((1,) + move_tile_counter_shape)
        round_counter_input = Input((1,) + round_counter_shape)

        board = Reshape(board_shape)(board_input)
        move_tile_counter = Reshape(move_tile_counter_shape)(move_tile_counter_input)
        round_counter = Reshape(round_counter_shape)(round_counter_input)

        ### Neural network
        cnn = Conv2D(64, (5, 5), padding='same', activation='relu')(board)
        flat_cnn = Flatten()(cnn)

        denses = concatenate([flat_cnn, move_tile_counter, round_counter])
        denses = Dense(128, activation='relu')(denses)

        predictions = Dense(self.nb_actions, kernel_initializer='zeros', activation='linear')(denses)
        ###

        model = Model(
            inputs=[board_input, move_tile_counter_input, round_counter_input],
            outputs=predictions
        )

        self.model = model

    def initialize_dqn(self):
        memory = SequentialMemory(limit=1000000, window_length=1)
        self.policy = BoltzmannQPolicy()
        processor = MultiInputProcessor(nb_inputs=3)
        self.dqn = DQNAgent(model=self.model, nb_actions=self.nb_actions,
                            memory=memory, nb_steps_warmup=10, gamma=0.95,
                            processor=processor, enable_dueling_network=True,
                            target_model_update=500, policy=self.policy)
        self.dqn.compile(RMSprop())

    def train(self):
        steps = 200
        cycles = 10000
        for i in range(cycles):
            history = self.dqn.fit(self.env, nb_steps=steps, log_interval=50)
            # save current episode reward
            episode_rewards = history.history['episode_reward']
            self.episode_reward_log.append((i*steps, np.mean(episode_rewards)))

            if i % 50 == 0:
                print("Saving model and performance log ...")
                self.save(self.model_save_path)
                np.save(self.log_save_path, self.episode_reward_log)

    def load(self, path):
        self.dqn.load_weights(path)

    def save(self, path):
        self.dqn.save_weights(path, overwrite=True)

    def test(self):
        self.dqn.test(env, nb_episodes=50000)
