import numpy as np

from life import Life, is_live

class Environment:
    def __init__(self):
        self.life = Life()
        self.rows = self.life.rows
        self.cols = self.life.cols
        self.color = 'blue'

        self.rounds_per_episode = 5
        self.moves_per_round = 3 # TODO clean variable name like this in two_player

        # Inputs and outputs for our models
        self.x1_shape = (self.rows, self.cols, 3) # board state
        self.x2_shape = (self.moves_per_round,) # move tile counter
        self.x3_shape = (self.rounds_per_episode,) # round counter
        self.model_board_shape = (self.rows, self.cols*2, 3) # concatenated boards shape
        self.y_shape = (self.rows*self.cols,) # actions

        self.reset()

    def reset(self):
        # these variables are reset every episode
        self.episode_round_n = 0
        self.episode_action_n = 0
        self.move_tile_counter = 0
        self.first_move = True

        max_per_ep = self.rounds_per_episode*self.moves_per_round
        self.board_states = np.zeros((max_per_ep,) + self.x1_shape, dtype='bool')
        self.move_tile_counters = np.zeros((max_per_ep,) + self.x2_shape, dtype='bool')
        self.round_counters = np.zeros((max_per_ep,) + self.x3_shape, dtype='bool')
        self.actions = np.zeros((max_per_ep,) + self.y_shape, dtype='bool')
        self.final_board = np.zeros(self.x1_shape, dtype='bool')

        self.life.clean()

    def encode_action(self, cell):
        action = np.zeros(self.rows*self.cols, dtype='bool')
        i, j = cell
        action[i*self.cols + j] = 1
        return action

    def board_state(self):
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

    def state(self):
        """ Returns (board_state, move_tile_counter, round_counter)"""
        move_tile_counter_i_mat = np.identity(self.moves_per_round, dtype='bool')
        encoded_move_tile_counter = move_tile_counter_i_mat[self.move_tile_counter]

        # encoded round_number, which tells you how many rounds have occured this episode
        round_identity_matrix = np.identity(self.rounds_per_episode, dtype='bool')
        encoded_round_counter = round_identity_matrix[self.episode_round_n]

        return (self.board_state(), encoded_move_tile_counter, encoded_round_counter)

    def step(self, cell):
        if self.is_legal_move(cell):
            # for replay experience
            self.memorize_state()
            self.memorize_action(cell)

            self.life.accept_tiles([cell], self.color)
            self.move_tile_counter += 1
            self.episode_action_n += 1

            self.first_move = False

            if self.move_tile_counter == self.moves_per_round:
                self.life.advance_state()
                self.move_tile_counter = 0
                self.episode_round_n += 1
            return

    def is_done(self):
        if self.episode_round_n == self.rounds_per_episode:
            self.memorize_final_board()
            return True
        if self.life.count_live_total() == 0 and self.first_move == False:
            self.memorize_final_board()
            return True
        return False

    def is_legal_move(self, cell):
        """
        On the first tile placement on the first round of an episode,
        the tile can be legally placed anywhere.

        After that, the a legal tile placement must neighbor a live tile.
        """
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
        return np.ravel(legal_move_mask)

    def memorize_final_board(self):
        encoded_final_board = self.board_state()
        self.final_board = encoded_final_board

    def memorize_action(self, cell):
        i = self.episode_action_n
        self.actions[i] = self.encode_action(cell)

    def memorize_state(self):
        i = self.episode_action_n
        encoded_board, encoded_move_tile_counter, encoded_round_counter = self.state()
        self.board_states[i] = encoded_board
        self.move_tile_counters[i] = encoded_move_tile_counter
        self.round_counters[i] = encoded_round_counter

    def get_episode_memories(self):
        # put the actual and final board states as a single 'image'
        final_board_expanded = np.array([self.final_board]*len(self.board_states), dtype='bool')
        board_states = np.concatenate((self.board_states, final_board_expanded), axis=2)

        # remove initial padding
        n = self.episode_action_n
        board_states = board_states[:n]
        move_tile_counters = self.move_tile_counters[:n]
        round_counters = self.round_counters[:n]
        actions = self.actions[:n]

        return (board_states, move_tile_counters, round_counters, actions)
