import os
import sys
import time

DELAY = 0.2
TILE_X = 10
TILE_Y = 20

def loading(x, y, t):
	chars = '◜◟◝◞'
	pos = t + (x - y*TILE_X)
	# return pos % len(chars)
	return chars[pos % len(chars)]

def pacman(x, y, t):
	if x != 0:
		return ''
	size = TILE_Y
	open = t%2 == 0
	if t%(size*2) < size:
		pos = t%size
		char = 'ᗧ' if open else '𝙾'
		if pos < size:
			line = ' ' * (pos-1) + char + '•' * (size - pos - 1) + '🍒'
		else:
			line = ' ' * (pos-1) + char + '•' * (size - pos)

	else:
		pos = size - t%size
		char = 'ᗤ' if open else '𝙾'
		if pos > 2:
			line = 'ᗣ' + ' ' * (pos-2) + char + ' ' * (size - pos)
		else:
			line = ' ' * (pos-1) + char + ' ' * (size - pos)

	return line[y]
	

t = 0
while True:
	mapped = []
	for x in range(TILE_X):
		mapped.append([])
		for y in range(TILE_Y):
			char = pacman(x, y, t)
			# char = loading(x, y, t)
			mapped[x].append(str(char))

	print('\n'.join(map(''.join, mapped)))
	time.sleep(DELAY)
	os.system('cls' if os.name == 'nt' else 'clear')
	t += 1