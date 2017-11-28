import keras

from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, Reshape, Activation, concatenate

import numpy as np

from two_player.game import Game

"""
Evolutionary Strategy
"""

class Agent:
    def __init__(self, env, load_model=False, model=None, rows=8, cols=8):
        self.rows = rows
        self.cols = cols
        self.env = env
        if model != None:
            self.model = model
        else:
            self.create_model()
        if load_model:
            self.load_model()

    def create_model(self):
        board_shape = (self.rows, self.cols, 5)
        actions = self.rows*self.cols

        board_input = Input(shape=board_shape)
        move_tile_counter_input = Input(shape=(3,))

        cnn = Conv2D(32, (3, 3), padding='same', activation='relu')(board_input)
        cnn = Conv2D(32, (3, 3), padding='same', activation='relu')(cnn)
        cnn = Conv2D(32, (3, 3), padding='same', activation='relu')(cnn)
        cnn = Conv2D(32, (3, 3), padding='same', activation='relu')(cnn)
        flat_cnn = Flatten()(cnn)

        denses = concatenate([flat_cnn, move_tile_counter_input])
        denses = Dense(64, activation='relu')(denses)
        denses = Dense(64, activation='relu')(denses)
        predictions = Dense(actions, activation='softmax')(denses)

        model = Model(inputs=[board_input, move_tile_counter_input], outputs=predictions)
        self.model = model

    def load_model(self, path='weights/lineage1.h5'):
        self.model.load_weights(path)

    def act(self, state):
        board_state, move_tile_counter = state
        action_probs = self.model.predict([np.array([board_state]), np.array([move_tile_counter])])
        legal_action_probs = np.multiply(self.env.legal_move_mask(), action_probs)
        chosen_action = np.argmax(legal_action_probs)
        row, col = (chosen_action//self.cols, chosen_action%self.cols)
        return (row, col)
