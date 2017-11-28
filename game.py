from life import Life, is_live
import numpy as np

"""
Wrapper for Life which turns it into a 2-player game
"""

BLANK = 0
LIVE_RED = 1
DEAD_RED = 2
LIVE_BLUE = 3
DEAD_BLUE = 4

class Game:
    def __init__(self):
        self.life = Life()
        self.reset()

    def reset(self):
        self.life.clean()
        self.turn = 'red'
        self.move_tile_counter = 0
        self.tiles_per_move = 3
        self.first_red_move = True
        self.first_blue_move = True

    def is_done(self):
        """
        There are two conditions for a completed game.
        1. All tiles are colored
        2. A color has no live tiles
        """

        blank_count = 0
        live_red_count = 0
        live_blue_count = 0

        for i in range(self.life.rows):
            for j in range(self.life.cols):
                cell = self.life.get_cell_value((i, j))
                if cell == BLANK:
                    blank_count += 1
                elif cell == LIVE_RED:
                    live_red_count += 1
                elif cell == LIVE_BLUE:
                    live_blue_count += 1

        if blank_count == 0 or live_red_count == 0 or live_blue_count == 0:
            if not self.first_red_move and not self.first_blue_move:
                return True
        return False

    def score(self):
        red = 0
        blue = 0
        for i in range(self.life.rows):
            for j in range(self.life.cols):
                cell = self.life.get_cell_value((i, j))
                if cell == LIVE_RED or cell == DEAD_RED:
                    red += 1
                elif cell == LIVE_BLUE or cell == DEAD_BLUE:
                    blue += 1
        return red, blue

    def winner(self):
        red, blue = self.score()
        if red > blue:
            return 'red'
        else:
            return 'blue'

    def accept_move(self, cell, player):
        if player == self.turn and self.is_legal_move(cell, player):
            self.life.accept_tiles([cell], self.turn)

            self.move_tile_counter += 1
            if self.move_tile_counter == self.tiles_per_move:
                if self.turn == 'red':
                    self.turn = 'blue'
                else:
                    self.turn = 'red'
                self.life.advance_state()
                self.move_tile_counter = 0

            if player == 'red':
                self.first_red_move = False
            else:
                self.first_blue_move = False

            return True
        return False

    def is_legal_move(self, cell, player):
        """
        All moves must be made on dead or blank tiles.

        Your first tile can be placed anywhere as long as it doesn't border
        an opponent's live tile.

        After that, a legal move is one in which your chosen tile is a
        neighbor of a live tile you control.
        """

        if is_live(self.life.get_cell_value(cell)):
            return False

        red_neighbors, blue_neighbors = self.life.neighbors(cell)
        if player == 'red':
            ally_neighbors = red_neighbors
            opponent_neighbors = blue_neighbors
        else:
            ally_neighbors = blue_neighbors
            opponent_neighbors = red_neighbors

        if (self.first_red_move and player == 'red') or (self.first_blue_move and player == 'blue'):
            if opponent_neighbors > 0:
                return False
            return True
        else:
            if ally_neighbors > 0:
                return True
            return False

    def encoded(self):
        player = self.turn

        # 5 possible cell states
        blank = np.array([1, 0, 0, 0, 0], dtype='bool')
        ally_alive = np.array([0, 1, 0, 0, 0], dtype='bool')
        ally_dead = np.array([0, 0, 1, 0, 0], dtype='bool')
        enemy_alive = np.array([0, 0, 0, 1, 0], dtype='bool')
        enemy_dead = np.array([0, 0, 0, 0, 1], dtype='bool')

        encoded_board = np.zeros((self.life.rows, self.life.cols, 5), dtype='bool')
        for i in range(self.life.rows):
            for j in range(self.life.cols):
                if self.life.get_cell_value((i, j)) == BLANK:
                    encoded_board[i][j] = blank
                elif self.life.get_cell_value((i, j)) == LIVE_RED:
                    if player == 'red':
                        encoded_board[i][j] = ally_alive
                    else:
                        encoded_board[i][j] = enemy_alive
                elif self.life.get_cell_value((i, j)) == LIVE_BLUE:
                    if player == 'blue':
                        encoded_board[i][j] = ally_alive
                    else:
                        encoded_board[i][j] = enemy_alive
                elif self.life.get_cell_value((i, j)) == DEAD_RED:
                    if player == 'red':
                        encoded_board[i][j] = ally_dead
                    else:
                        encoded_board[i][j] = enemy_dead
                elif self.life.get_cell_value((i, j)) == DEAD_BLUE:
                    if player == 'blue':
                        encoded_board[i][j] = ally_dead
                    else:
                        encoded_board[i][j] = enemy_dead

        move_tile_count_1 = np.array([1, 0, 0], dtype='bool')
        move_tile_count_2 = np.array([0, 1, 0], dtype='bool')
        move_tile_count_3 = np.array([0, 0, 1], dtype='bool')

        if self.move_tile_counter == 0:
            encoded_move_tile_count = move_tile_count_1
        elif self.move_tile_counter == 1:
            encoded_move_tile_count = move_tile_count_2
        else:
            encoded_move_tile_count = move_tile_count_3

        return (encoded_board, encoded_move_tile_count)
