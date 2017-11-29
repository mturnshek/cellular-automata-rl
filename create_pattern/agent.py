import numpy as np
import keras
from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, concatenate, Dropout
from keras.optimizers import RMSprop

import keyboard

class Agent:
    def __init__(self, env):
        self.env = env

        self.max_replays = 40000
        self.replay_index = 0
        self.use_all_replays = False

        self.epsilon = 0.95
        self.epsilon_max = 0.95 # percentage chance of random action
        self.epsilon_min = 0.2
        self.move_epsilon_down = True
        self.change_epsilon = True

        # experience replays for each of the inputs and outputs
        self.replays_x1 = np.zeros((self.max_replays,) + env.model_board_shape)
        self.replays_x2 = np.zeros((self.max_replays,) + env.x2_shape)
        self.replays_x3 = np.zeros((self.max_replays,) + env.x3_shape)
        self.replays_y = np.zeros((self.max_replays,) + env.y_shape)

        self.save_path = 'weights/pattern_maker1.h5'
        self.create_model()

    def alter_epsilon(self):
        epsilon_movement_rate = 0.02
        if self.move_epsilon_down:
            self.epsilon -= epsilon_movement_rate
            if self.epsilon <= self.epsilon_min:
                self.move_epsilon_down = False
        else:
            self.epsilon += epsilon_movement_rate
            if self.epsilon >= self.epsilon_max:
                self.move_epsilon_down = True

    def toggle_set_epsilon_to_zero(self):
        if self.change_epsilon:
            self.epsilon = 0.0
            self.change_epsilon = False
        else:
            self.epsilon = 1.0
            self.change_epsilon = True

    def add_memories(self, memories):
        board_states, move_tile_counters, round_counters, actions = memories
        n_replays = len(board_states) # number of replays being added in this call
        if self.replay_index + n_replays >= self.max_replays:
            self.replay_index = 0
            self.use_all_replays = True

        i = self.replay_index

        self.replays_x1[i:i+n_replays] = board_states
        self.replays_x2[i:i+n_replays] = move_tile_counters
        self.replays_x3[i:i+n_replays] = round_counters
        self.replays_y[i:i+n_replays] = actions

        self.replay_index += n_replays

    def create_model(self):
        board_shape = self.env.model_board_shape # concatted state + desired state
        move_tile_counter_shape = self.env.x2_shape
        round_counter_shape = self.env.x3_shape
        actions = self.env.rows*self.env.cols

        board_input = Input(shape=board_shape)
        move_tile_counter_input = Input(shape=move_tile_counter_shape)
        round_counter_input = Input(shape=round_counter_shape)

        cnn = Conv2D(64, (3, 3), padding='same', activation='relu')(board_input)
        cnn = Conv2D(64, (3, 3), padding='same', activation='relu')(cnn)
        cnn = Conv2D(64, (3, 3), padding='same', activation='relu')(cnn)
        cnn = Conv2D(64, (3, 3), padding='same', activation='relu')(cnn)
        cnn = Dropout(0.25)(cnn)
        flat_cnn = Flatten()(cnn)

        denses = concatenate([flat_cnn, move_tile_counter_input, round_counter_input])
        denses = Dense(128, activation='relu')(denses)
        denses = Dropout(0.25)(denses)
        denses = Dense(128, activation='relu')(denses)
        denses = Dense(128, activation='relu')(denses)
        denses = Dropout(0.25)(denses)
        predictions = Dense(actions, activation='softmax')(denses)

        model = Model(
            inputs=[board_input, move_tile_counter_input, round_counter_input],
            outputs=predictions
        )

        model.compile(
        optimizer=RMSprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

        self.model = model

    def save_model(self):
        self.model.save_weights(self.save_path)

    def train(self, desired_board_state):

        runs_per_cycle = 32

        while True:
            for i in range(runs_per_cycle):
                self.run(desired_board_state)
                if keyboard.is_pressed('t'):
                    self.toggle_set_epsilon_to_zero()
                    print('Epsilon oscillation:', self.change_epsilon)

            if self.change_epsilon:
                self.alter_epsilon()
            print(f'epsilon: {self.epsilon}')

            self.train_on_replays()

    def run(self, desired_board_state):
        env = self.env

        while not env.is_done():
            board, move_tile_counter, round_counter = env.state()
            concat_board = np.concatenate((board, desired_board_state), axis=1)

            cell = self.act((concat_board, move_tile_counter, round_counter))
            env.step(cell)

        self.add_memories(env.get_episode_memories())
        self.env.reset()

    def train_on_replays(self):
        # Use only the replays with data
        if self.use_all_replays:
            n_replays = self.max_replays
        else:
            n_replays = self.replay_index

        x1 = self.replays_x1[:n_replays]
        x2 = self.replays_x2[:n_replays]
        x3 = self.replays_x3[:n_replays]
        y = self.replays_y[:n_replays]

        # Train model on all replays once
        self.model.fit(
            x=[x1, x2, x3],
            y=y,
            shuffle=True,
            batch_size=64,
            epochs=1
        )
        self.save_model()

    def act(self, state):
        """
        State must be the complete input, with
        CURRENT BOARD STATE : DESIRED BOARD STATE
        concatenation
        """
        legal_move_mask = self.env.legal_move_mask()
        if np.random.random() > self.epsilon:
            board_state, move_tile_counter, round_counter = state
            action_probs = self.model.predict([np.array([board_state]), np.array([move_tile_counter]), np.array([round_counter])])
            legal_action_probs = np.multiply(legal_move_mask, action_probs)
            chosen_action = np.argmax(legal_action_probs)
        else: # random legal action
            legal_actions = np.nonzero(legal_move_mask)[0]
            chosen_action = np.random.choice(legal_actions)

        row, col = (chosen_action // self.env.cols, chosen_action % self.env.cols)
        return (row, col)
