import numpy as np
import keras

class Agent:
    def __init__(self, env):
        # Problem-specific shapes #
        self.x1_shape = np.zeros((env.rows, env.cols, 5)) # board state
        self.x2_shape = np.zeros((3,)) # move tile counter
        self.x3_shape = np.zeros((10,)) # round counter
        self.x4_shape = self.x1_shape # desired board state
        self.y_shape = np.zeros(env.rows*env.cols) # actions
        ###

        self.max_replays = 10000
        self.replays_stored = 0

        # experience replays for each of the inputs and outputs
        self.replays_x1 = np.zeros((self.max_replays,) + self.x1_shape)
        self.replays_x2 = np.zeros((self.max_replays,) + self.x2_shape)
        self.replays_x3 = np.zeros((self.max_replays,) + self.x3_shape)
        self.replays_x4 = np.zeros((self.max_replays,) + self.x4_shape)
        self.replays_y = np.zeros((self.max_replays,) + self.y_shape)

    def add_memories(board_states, move_tile_counters, round_counters, final_state, actions):
        n_replays = len(board_states) # number of replays being added in this call
        stored = self.replays_stored
        i = stored % self.max_replays # overwrite from beginning after max stored

        self.replays_x1[i:i+n_replays] = board_states
        self.replays_x2[i:i+n_replays] = move_tile_counters
        self.replays_x3[i:i+n_replays] = round_counters
        self.replays_x4[i:i+n_replays] = final_state
        self.replays_y[i:i+n_replays] = actions

        self.replays_stored += n_replays

    def create_model():
        pass

    def train():
        pass

    def predict():
        pass
