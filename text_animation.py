import os
import sys
import time

DELAY = 0.2
TILE_X = 10
TILE_Y = 10

def loading(x, y, t):
	chars = '◜◟◝◞'
	pos = t + (x - y*TILE_X)
	# return pos % len(chars)
	return chars[pos % len(chars)]

def defiling(x, y, t):
	pass

t = 0
while True:
	mapped = []
	for x in range(TILE_X):
		mapped.append([])
		for y in range(TILE_Y):
			char = loading(x, y, t)
			mapped[x].append(str(char))

	print('\n'.join(map(''.join, mapped)))
	time.sleep(DELAY)
	os.system('cls' if os.name == 'nt' else 'clear')
	t += 1