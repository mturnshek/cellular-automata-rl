import numpy as np
import pygame
import sys
import time

"""
Life, modified for 2-player, competitive, territorial play

For live cells:
	Each cell with one or no neighbors dies, as if by solitude.
	Each cell with four or more neighbors dies, as if by overpopulation.
	Each cell with two or three neighbors survives.
For dead cells:
	Each cell with three neighbors becomes live with the color of the majority.

"""

#############
# constants #
#############

BLANK = 0
LIVE_RED = 1
DEAD_RED = 2
LIVE_BLUE = 3
DEAD_BLUE = 4

BLANK_COLOR = np.array([200, 200, 200])
LIVE_RED_COLOR = np.array([150, 0, 0])
DEAD_RED_COLOR = np.array([250, 100, 100])
LIVE_BLUE_COLOR = np.array([0, 0, 150])
DEAD_BLUE_COLOR = np.array([100, 100, 250])

###########
# helpers #
###########

def is_live(value):
	if value == LIVE_RED or value == LIVE_BLUE:
		return True
	return False

def color(value):
	if value == LIVE_RED or value == DEAD_RED:
		return 'red'
	else:
		return 'blue'


def neighbors_info(neighbor_values):
	red = 0
	blue = 0

	for v in neighbor_values:
		if is_live(v):
			if color(v) == 'red':
				red += 1
			else:
				blue += 1

	return red, blue


def kill(value):
	if value == LIVE_RED:
		return DEAD_RED
	else:
		return DEAD_BLUE


#########################
# Main class definition #
#########################

class Life:
	def __init__(self, rows=6, cols=6, display=True, step=False):
		self.rows = rows
		self.cols = cols
		self.clean()

		self.BLANK = 0
		self.LIVE_RED = 1
		self.DEAD_RED = 2
		self.LIVE_BLUE = 3
		self.DEAD_BLUE = 4

		self.step = step
		self.display = display
		if display:
			self.px = 40
			height = self.px*rows
			width = self.px*cols
			pygame.init()
			size = width, height
			self.screen = pygame.display.set_mode(size)
			pygame.event.set_grab(False)

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
		return neighbors_info(vals) # returns red_count, blue_count

	def count_live_neighbors(self, cell):
		red_neighbors, blue_neighbors = self.neighbors(cell)
		total_neighbors = red_neighbors + blue_neighbors
		return total_neighbors

	def count_live_red_and_blue(self):
		n_red, n_blue = 0, 0
		for i in range(self.rows):
			for j in range(self.cols):
				v = self.get_cell_value((i, j))
				if v == LIVE_RED:
					n_red += 1
				elif v == LIVE_BLUE:
					n_blue += 1
		return n_red, n_blue

	def count_live_total(self):
		n_red, n_blue = self.count_live_red_and_blue()
		return n_red + n_blue

	# sets coordinates in the list 'tiles' to be live with given color
	def accept_tiles(self, tiles, color):
		if color == 'red':
			value = LIVE_RED
		else:
			value = LIVE_BLUE
		for tile in tiles:
			if not is_live(self.get_cell_value(tile)):
				self.set_cell_value(tile, value)
		if self.display:
			self.update_display()

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
		if self.display:
			self.update_display()

	def clean(self):
		# padded with a static 0 value on all edges
		self.state = np.zeros((self.rows + 2, self.cols + 2), dtype='uint8')

	def update_display(self):
		if self.display:
			frame = self.create_frame_for_state()
			self.screen.blit(pygame.surfarray.make_surface(frame), (0, 0))
			pygame.display.flip()
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					sys.exit()

			if self.step:
				_ = input(' ')

	def create_frame_for_state(self):
		px = self.px
		frame = np.zeros((self.rows*px, self.cols*px, 3))
		for i in range(self.rows):
			for j in range(self.cols):
				val = self.get_cell_value((i, j))
				if val == BLANK:
					frame[i*px:(i+1)*px, j*px:(j+1)*px] = BLANK_COLOR
				if val == LIVE_RED:
					frame[i*px:(i+1)*px, j*px:(j+1)*px] = LIVE_RED_COLOR
				if val == DEAD_RED:
					frame[i*px:(i+1)*px, j*px:(j+1)*px] = DEAD_RED_COLOR
				if val == LIVE_BLUE:
					frame[i*px:(i+1)*px, j*px:(j+1)*px] = LIVE_BLUE_COLOR
				if val == DEAD_BLUE:
					frame[i*px:(i+1)*px, j*px:(j+1)*px] = DEAD_BLUE_COLOR
		return frame
