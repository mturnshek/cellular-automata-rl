import numpy as np

from life import Life, is_live

class Environment:
    def __init__(self, rounds_per_episode=10, display=False):
        self.life = Life(display=display)
        self.rows = self.life.rows
        self.cols = self.life.cols
        self.color = 'blue'

        self.rounds_per_episode = rounds_per_episode
        self.moves_per_round = 3 # TODO clean variable name like this in two_player

        # Inputs and outputs for our models
        self.x1_shape = (self.rows, self.cols, 3) # board state
        self.x2_shape = (self.moves_per_round,) # move tile counter
        self.x3_shape = (self.rounds_per_episode + 1,) # round counter
        self.y_shape = (self.rows*self.cols + 1,) # actions, one additional for "passing"

        self.nb_actions = self.rows*self.cols + 1

        self.reset()

    def reset(self):
        # these variables are reset every episode
        self.episode_round_n = 0
        self.episode_action_n = 0
        self.move_tile_counter = 0
        self.first_move = True

        self.life.clean()
        return self.state()

    def encode_action(self, cell):
        action = np.zeros(self.y_shape, dtype='bool')
        if cell == None:
            action[self.cols*self.rows] = 1
        else:
            i, j = cell
            action[i*self.cols + j] = 1
        return action

    def decode_action(self, action):
        legal_move_mask = self.legal_move_mask()

        if legal_move_mask[action] == 0: # move is illegal
            return None # illegal move means pass

        if action == self.rows*self.cols:
            return None
        else:
            row, col = (action // self.cols, action % self.cols)
            return (row, col)

    def encode_board(self):
        """ Returns encoded board state with shape (rows, cols, 3,) """
        # 3 possible cell states
        blank = np.array([1, 0, 0], dtype='bool')
        alive = np.array([0, 1, 0], dtype='bool')
        dead = np.array([0, 0, 1], dtype='bool')

        encoded_board = np.zeros(self.x1_shape, dtype='bool')
        for i in range(self.life.rows):
            for j in range(self.life.cols):
                v = self.life.get_cell_value((i, j))
                if v == self.life.BLANK:
                    encoded_board[i][j] = blank
                elif v == self.life.LIVE_BLUE:
                    encoded_board[i][j] = alive
                elif v == self.life.DEAD_BLUE:
                    encoded_board[i][j] = dead
        return encoded_board

    def encode_move_tile_counter(self):
        """ Returns encoded form of how many moves have occured this round """
        move_tile_counter_i_mat = np.identity(self.moves_per_round, dtype='bool')
        encoded_move_tile_counter = move_tile_counter_i_mat[self.move_tile_counter]
        return encoded_move_tile_counter

    def encode_round_counter(self):
        """ Returns encoded form of how many rounds have occured this episode """
        round_identity_matrix = np.identity(self.rounds_per_episode + 1, dtype='bool')
        encoded_round_counter = round_identity_matrix[self.episode_round_n]
        return encoded_round_counter

    def state(self):
        """ Returns [board_state, move_tile_counter, round_counter]"""
        return self.encode_board(), self.encode_move_tile_counter(), self.encode_round_counter()

    def step(self, action):
        cell = self.decode_action(action)
        if cell != None: # cell being None would mean the agent "passed"
            self.life.accept_tiles([cell], self.color)
            self.first_move = False

        self.move_tile_counter += 1
        self.episode_action_n += 1

        if self.move_tile_counter == self.moves_per_round:
            self.life.advance_state()
            self.move_tile_counter = 0
            self.episode_round_n += 1

        state = self.state()
        done = self.is_done()
        if done:
            reward = self.life.count_colored()
        else:
            reward = 0

        return (state, reward, done, {})

    def is_done(self):
        if self.episode_round_n == self.rounds_per_episode:
            return True
        if self.life.count_live_total() == 0 and self.first_move == False:
            return True
        return False

    def is_legal_move(self, cell):
        """
        On the first tile placement on the first round of an episode,
        the tile can be legally placed anywhere.

        After that, the a legal tile placement must neighbor a live tile.
        """
        if cell == None:
            return True
        if is_live(self.life.get_cell_value(cell)):
            return False
        if self.first_move:
            return True
        elif self.life.count_live_neighbors(cell) == 0:
            return False
        return True

    def legal_move_mask(self):
        legal_move_mask = np.zeros((self.rows, self.cols), dtype='bool')
        for i in range(self.rows):
            for j in range(self.cols):
                legal_move_mask[i][j] = self.is_legal_move((i, j))
        flattened = np.ravel(legal_move_mask)
        return np.append(flattened, np.array([1]))
