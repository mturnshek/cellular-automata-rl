from life import Life, is_live

"""
Wrapper for Life which turns it into a 2-player game
"""

class Game:
    def __init__(self):
            self.life = Life()
            self.turn = 'red'
            self.move_tile_counter = 0
            self.tiles_per_move = 3
            self.first_red_move = True
            self.first_blue_move = True

    def accept_move(self, cell, player):
        if player == self.turn and self.is_legal_move(cell, player):
            self.life.accept_tiles([cell], self.turn)

            self.move_tile_counter += 1
            if self.move_tile_counter == self.tiles_per_move:
                if self.turn == 'red':
                    self.turn = 'blue'
                else:
                    self.turn = 'red'

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

        if self.first_move:
            if opponent_neighbors > 0:
                return False
            return True
        else:
            if ally_neighbors > 0:
                return True
            return False
