from life import Life
import pygame
import numpy as np
import time
import sys
import scipy.ndimage

px = 20
rows = 8
cols = 8
height = px*rows
width = px*cols
life = Life(rows=rows, cols=cols)
life.accept_tiles(
	[(4, 4), (4, 6), (3, 3),
	 (3, 4), (5, 5), (6, 6),
	 (7, 7), (6, 7), (5, 7)],
	'blue')

pygame.init()
size = width, height
screen = pygame.display.set_mode(size)
pygame.event.set_grab(False)

BLANK = 0
LIVE_RED = 1
DEAD_RED = 2
LIVE_BLUE = 3
DEAD_BLUE = 4

blank = np.array([200, 200, 200])
blue = np.array([0, 0, 150])
red = np.array([150, 0, 0])
blueish = np.array([100, 100, 250])
reddish = np.array([250, 100, 100])

def create_frame_for_state():
	display = np.zeros((life.rows*px, life.cols*px, 3))
	for i in range(life.rows):
		for j in range(life.cols):
			val = life.get_cell_value((i, j))
			if val == BLANK:
				display[i*px:(i+1)*px, j*px:(j+1)*px] = blank
			if val == LIVE_RED:
				display[i*px:(i+1)*px, j*px:(j+1)*px] = red
			if val == LIVE_BLUE:
				display[i*px:(i+1)*px, j*px:(j+1)*px] = blue
			if val == DEAD_RED:
				display[i*px:(i+1)*px, j*px:(j+1)*px] = reddish
			if val == DEAD_BLUE:
				display[i*px:(i+1)*px, j*px:(j+1)*px] = blueish
	return display

step_time = 0.5 #sec


while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()

	frame = create_frame_for_state()
	screen.blit(pygame.surfarray.make_surface(frame), (0, 0))
	pygame.display.flip()

	life.advance_state()
	time.sleep(step_time)
