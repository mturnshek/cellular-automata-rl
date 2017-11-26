import numpy as np
"""
Life, modified for 2-player, competitive, territorial play

For live cells:
    Each cell with one or no neighbors dies, as if by solitude.
    Each cell with four or more neighbors dies, as if by overpopulation.
    Each cell with two or three neighbors survives.
For dead cells:
    Each cell with three neighbors becomes live with the color of the majority.

"""

#######

BLANK = 0
LIVE_RED = 1
DEAD_RED = 2
LIVE_BLUE = 3
DEAD_BLUE = 4

def is_live(value):
    if value == LIVE_RED or value == LIVE_BLUE:
        return True
    return False

def color(value):
    if value == LIVE_RED or value == DEAD_RED:
        return 'red'
    else:
        return 'blue'

# returns count, majority
def neighbors_info(neighbor_values):
    red = 0
    blue = 0

    for v in neighbor_values:
        if is_live(v):
            if color(v) == 'red':
                red += 1
            else:
                blue += 1

    if red > blue:
        majority = 'red'
    else:
        majority = 'blue'

    return red, blue


def kill(value):
    if value == LIVE_RED:
        return DEAD_RED
    else:
        return DEAD_BLUE

#######

class Life:
    def __init__(self, rows=20, cols=20):
        self.rows = rows
        self.cols = cols
        self.clean()

    def __str__(self):
        s = ""
        for i in range(self.rows):
            for j in range(self.cols):
                s += str(self.get_cell_value((i, j)))
            s += '\n'
        return s

    def get_cell_value(self, cell):
        i, j = cell
        return self.state[i+1][j+1]

    def set_cell_value(self, cell, value):
        i, j = cell
        self.state[i+1][j+1] = value

    def neighbors(self, cell):
        i, j = cell
        vals = np.copy(np.ravel(self.state[i:i+3, j:j+3]))
        vals[4] = BLANK # remove the value of the cell itself
        return neighbors_info(vals) # returns count, majority

    # sets coordinates in the list 'tiles' to be live with given color
    def accept_tiles(self, tiles, color):
        if color == 'red':
            value = LIVE_RED
        else:
            value = LIVE_BLUE
        for tile in tiles:
            if not is_live(self.get_cell_value(tile)):
                self.set_cell_value(tile, value)

    def advance_cell(self, cell):
        value = self.get_cell_value(cell)
        next_value = value
        red, blue = self.neighbors(cell)
        count = red + blue

        if is_live(value):
            if count <= 1: # underpopulation
                next_value = kill(value)
            if count >= 4: # overpopulation
                next_value = kill(value)

        else: # dead
            if count == 3:
                if red > blue:
                    next_value = LIVE_RED
                else:
                    next_value = LIVE_BLUE

        return next_value

    def advance_state(self):
        self.next_state = np.zeros(self.state.shape, dtype='uint8')
        for i in range(self.rows):
            for j in range(self.cols):
                self.next_state[i+1][j+1] = self.advance_cell((i, j))
                # adding 1 due to padding on all sides
        self.state = np.copy(self.next_state)

    def clean(self):
        # padded with a static 0 value on all edges
        self.state = np.zeros((self.rows + 2, self.cols + 2), dtype='uint8')
