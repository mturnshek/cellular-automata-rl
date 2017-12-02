import numpy as np

from keras.models import Model
from keras.layers import Dense, Reshape, Activation, Flatten, Input, Conv2D, concatenate, Dropout
from keras.optimizers import RMSprop

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
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
        self.env = Environment(display=False)
        self.nb_actions = self.env.rows*self.env.cols + 1

        self.create_model()
        self.initialize_dqn()

    def create_model(self):
        board_shape = self.env.x1_shape
        move_tile_counter_shape = self.env.x2_shape
        round_counter_shape = self.env.x3_shape

        # Keras-rl interprets an extra axis=0 dimension
        # tacked on to observations
        board_input = Input((1,) + board_shape)
        move_tile_counter_input = Input((1,) + move_tile_counter_shape)
        round_counter_input = Input((1,) + round_counter_shape)

        board = Reshape(board_shape)(board_input)
        move_tile_counter = Reshape(move_tile_counter_shape)(move_tile_counter_input)
        round_counter = Reshape(round_counter_shape)(round_counter_input)

        ### Neural network
        cnn = Conv2D(96, (5, 5), padding='same', activation='relu')(board)
        cnn = Conv2D(96, (5, 5), padding='same', activation='relu')(cnn)
        cnn = Conv2D(96, (5, 5), padding='same', activation='relu')(cnn)
        flat_cnn = Flatten()(cnn)

        denses = concatenate([flat_cnn, move_tile_counter, round_counter])
        denses = Dense(512, activation='relu')(denses)
        denses = Dropout(0.3)(denses)
        denses = Dense(512, activation='relu')(denses)
        denses = Dropout(0.3)(denses)
        predictions = Dense(self.nb_actions, activation='softmax')(denses)
        ###

        model = Model(
            inputs=[board_input, move_tile_counter_input, round_counter_input],
            outputs=predictions
        )

        self.model = model

    def initialize_dqn(self):
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy()
        processor = MultiInputProcessor(nb_inputs=3)
        self.dqn = DQNAgent(model=self.model, nb_actions=self.nb_actions,
                            memory=memory, nb_steps_warmup=10,
                            processor=processor,
                            target_model_update=1, policy=policy)
        self.dqn.compile(RMSprop())

    def train(self):
        self.dqn.fit(self.env, nb_steps=900000, log_interval=50)
        self.save()

    def load(self):
        self.dqn.load_weights('weights/grow_dqn_weights.h5')

    def save(self):
        self.dqn.save_weights('weights/grow_dqn_weights.h5', overwrite=True)

    def test(self):
        self.dqn.test(env, nb_episodes=50000)
