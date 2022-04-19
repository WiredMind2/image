import os
import zipfile
import json
import hashlib
import time
import ctypes

import pygame


class Viewer():
	def __init__(self, files=None, root=None, save=None):
		self.savePath = save
		self.root = root

		self.current = None
		self.imgpos, self.pos = 0, 0

		self.save_data = {}

		last = None
		if self.savePath is not None and os.path.exists(self.savePath):
			last, imgpos, pos = self.parseSave()
			if last is not None:
				self.root = os.path.dirname(last)
				self.get_files = lambda: [
					os.path.normpath(os.path.join(self.root, f)) 
					for f in os.listdir(self.root)
					if f.endswith('.cbz')
					or (os.path.isdir(os.path.join(self.root, f)) and len(os.listdir(os.path.join(self.root, f))))
				]
				self.current = last
				self.imgpos, self.pos = imgpos, pos

		if last is None:
			if files is not None:
				self.get_files = lambda: files

			elif root is not None:
				self.get_files = lambda: [os.path.normpath(os.path.join(self.root, f)) for f in os.listdir(self.root) if f.endswith('.cbz')]

		self.opened_files = self.get_files()

		if len(self.opened_files) == 0:
			print('No files found!')
			return

		if self.current is None:
			self.current = self.opened_files[0]

		self.width = None

		pygame.init()
		pygame.font.init()
		pygame.display.set_caption('Manga Reader')

		self.font = pygame.font.SysFont('Comic Sans MS', 20)

		scroll_off = 100
		scroll_off_continuous = scroll_off // 5

		# TODO - Top bar menu
		self.screen_pos = (0, 0) # (0, 50)

		self.size = self.width, self.height = (1000, 900)

		self.update_size()
		
		self.imgs = {}
		self.parse_file(self.current)

		if len(self.imgs[self.current]) == 0:
			print('No images found!')
			return
		img = self.imgs[self.current][0]
		self.size = self.width, self.height = img.width, self.height
		
		self.update_size()

		pressed = []

		self.update()

		running = True
		while running:
			time.sleep(1/60)
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False
				elif event.type == pygame.MOUSEBUTTONDOWN:
					self.opened_files = self.get_files()
					if event.button == pygame.BUTTON_LEFT:
						self.next_file()
					elif event.button == pygame.BUTTON_RIGHT:
						self.last_file()
					elif event.button == pygame.BUTTON_WHEELUP:
						self.pos = self.pos - scroll_off
						self.update()
					elif event.button == pygame.BUTTON_WHEELDOWN:
						self.pos = self.pos + scroll_off
						self.update()
				elif event.type == pygame.VIDEORESIZE:
					self.size = self.width, self.height = event.w, event.h
					self.update_size()
					self.parse_file(self.current)
					self.update()
				elif event.type == pygame.DROPFILE:
					self.open_file(event.file)
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_RIGHT or event.key == pygame.K_SPACE:
						self.next_file()
					elif event.key == pygame.K_LEFT:
						self.last_file()
					else:
						if event.mod == pygame.KMOD_LCTRL:
							if event.key == pygame.K_v:
								#get clipboard
								types = pygame.scrap.get_types()

								if 'FileName' in types:
									filename = pygame.scrap.get('FileName')[:-1].decode()

									try:
										filename = self.GetLongPathName(filename) # TODO - Check when it's not needed
									except:
										pass

									self.open_file(filename)
						if event.key not in pressed:
							pressed.append(event.key)
				elif event.type == pygame.KEYUP:
					if event.key in pressed:
						pressed.remove(event.key)
			# print(pressed)
			for key in pressed:
				if key == pygame.K_UP:
					self.pos = self.pos - scroll_off_continuous
					self.update()
				elif key == pygame.K_DOWN:
					self.pos = self.pos + scroll_off_continuous
					self.update()

		self.save()

	def next_file(self):
		i = list(self.opened_files).index(self.current)
		if i+1 < len(self.opened_files)-1:
			self.save()
			self.current = list(self.opened_files)[i+1]
			self.parse_file(self.current)
			self.imgpos, self.pos = 0, 0
			self.update()

	def last_file(self):
		i = list(self.opened_files).index(self.current)
		if i-1 >= 0:
			self.save()
			self.current = list(self.opened_files)[i-1]
			self.parse_file(self.current)
			self.imgpos, self.pos = 0, 0
			self.update()

	def open_file(self, file):
		self.save()
		self.current = os.path.normpath(file)
		
		if os.path.isdir(self.current):
			self.root = self.current
			
			dir_hash = self.hash(os.path.normpath(self.current))
			file_hash = self.save_data.get(dir_hash, None)
			print(dir_hash, file_hash)
			if file_hash is not None:
				self.current, self.imgpos, self.pos = self.save_data[file_hash]
			else:
				self.current = self.get_files()[0]
				self.imgpos, self.pos = 0, 0
		else:
			self.root = os.path.dirname(self.current)
			self.imgpos, self.pos = 0, 0

		self.get_files = lambda: [os.path.normpath(os.path.join(self.root, f)) for f in os.listdir(self.root) if f.endswith('.cbz')]
		self.opened_files = self.get_files()
		self.parse_file(self.current)
		self.update()

	def update(self):
		for img, pos in self.get_img_inview():
			size = (
				self.width, 
				int(img.height*self.width/img.width)
			)
			if int(img.height*self.width/img.width) != img.height:
				img.update(pygame.transform.smoothscale(img.img, size))

			self.screen.blit(img.img, (0, pos))

		text = self.font.render(f'{os.path.basename(self.current)}', False, (0, 0, 0))
		self.screen.blit(text, (10, 10))

		self.display.blit(self.screen, self.screen_pos)
		pygame.display.update(self.screen.get_rect(topleft=self.screen_pos))
		
	def update_size(self):
		self.display = pygame.display.set_mode((self.size[0], self.size[1]+self.screen_pos[1]), pygame.RESIZABLE)
		self.screen = pygame.Surface(self.size)

		if not pygame.scrap.get_init():
			pygame.scrap.init()

	def get_img_inview(self):
		i = 0
		imgs = self.imgs[self.current]

		if self.imgpos < 0:
			self.imgpos = 0
		elif self.imgpos > len(imgs)-1:
			self.imgpos = len(imgs)-1

		img = imgs[self.imgpos]


		while self.pos > img.height and self.imgpos < len(imgs)-1:
			self.pos -= img.height
			self.imgpos += 1
			img = imgs[self.imgpos]

		while self.pos < 0 and self.imgpos > 0:
			self.imgpos -= 1
			img = imgs[self.imgpos]
			self.pos += img.height

		if self.pos < 0 and self.imgpos == 0:
			self.pos = 0

		tmp_imgpos = self.imgpos
		tmp_pos = -self.pos
		out = []

		while tmp_pos < self.height and tmp_imgpos <= len(imgs)-1:
			img = imgs[tmp_imgpos]
			out.append((img, tmp_pos))
			tmp_pos += img.height
			tmp_imgpos += 1

		if tmp_pos < self.height and tmp_imgpos == len(imgs):
			self.pos -= self.height - tmp_pos
			return self.get_img_inview()

		return out

	def parse_file(self, file):
		if file is None or not os.path.exists(file):
			return

		self.imgs[file] = []

		if os.path.isfile(file):
			with zipfile.ZipFile(file, 'r') as f:
				for name in f.namelist():
					img_file = f.open(name, 'r')

					# data = self.parseImg(img_file)
					data = Manga_Image(lambda img_file=img_file: self.parseImg(img_file))

					# if len(self.imgs[file]) > 0:
					# 	data['start'] = self.imgs[file][-1]['start'] + self.imgs[file][-1]['height']

					self.imgs[file].append(data)
		elif os.path.isdir(file):
			for name in sorted(os.listdir(file), key=lambda e:''.join(map(lambda s: s.zfill(3) if s.isdigit() else '', e.split('-')))):
				img_file = open(os.path.join(file, name))

				data = Manga_Image(lambda img_file=img_file: self.parseImg(img_file))

				# if data is not None:
				# 	if len(self.imgs[file]) > 0:
				# 		data['start'] = self.imgs[file][-1]['start'] + self.imgs[file][-1]['height']

				self.imgs[file].append(data)

	def parseImg(self, img_file):
		if type(img_file) == pygame.Surface:
			img = img_file
		else:
			try:
				img = pygame.image.load(img_file).convert()
			except Exception as e:
				print(f'Error on image: {img_file} - {e}')
				return None

		if self.width is None:
			self.width = img.get_width()

		if img.get_width() != self.width:
			size = (
				self.width, 
				int(img.get_height()*self.width/img.get_width())
			)
			img = pygame.transform.smoothscale(img, size)

		data = {
			'img':img,
			'height':img.get_height(),
			'width':img.get_width(),
			'start':0
		}
		return img

	def parseSave(self):
		if not os.path.exists(self.savePath):
			return None, None
		with open(self.savePath, 'r') as f:
			self.save_data = json.load(f)
			last_hash = self.save_data['last']
			file, imgpos, pos = self.save_data[last_hash]
		imgpos, pos = int(imgpos), int(pos)
		return file, imgpos, pos

	def save(self):
		if self.savePath is None:
			return
		if os.path.exists(self.savePath):
			with open(self.savePath, 'r') as f:
				self.save_data = json.load(f)
		else:
			self.save_data = {
			}

		last_hash = self.hash(self.current)
		self.save_data['last'] = last_hash
		self.save_data[last_hash] = self.current, self.imgpos, self.pos

		dir_hash = self.hash(os.path.normpath(os.path.dirname(self.current)))
		self.save_data[dir_hash] = last_hash

		with open(self.savePath, 'w') as f:
			json.dump(self.save_data, f, indent=4)

	def hash(self, data):
		return hashlib.sha1(str(data).encode("utf-8")).hexdigest()

	def GetLongPathName(self, path):				
		GetLongPathName = ctypes.windll.kernel32.GetLongPathNameW
		buffer = ctypes.create_unicode_buffer(GetLongPathName(path, 0, 0))

		GetLongPathName(path, buffer, len(buffer))
		return buffer.value


class Manga_Image:
	def __init__(self, img):
		self.update(img)

	@property
	def height(self):
		if self._height is None:
			if self.img is None:
				self.img = self.caller()
			self._width, self._height = self.img.get_size()
		return self._height

	@property
	def width(self):
		if self._width is None:
			if self.img is None:
				self.img = self.caller()
			self._width, self._height = self.img.get_size()
		return self._width

	def update(self, img):
		if callable(img):
			self.caller = img
			self.img = None
			self._width, self._height = None, None
		else:
			self.img = img
			self._width, self._height = self.img.get_size()

root = "C:/Users/moi/Documents/Series/Webtoons/Versatile Mage"
files = [os.path.normpath(os.path.join(root, f)) for f in os.listdir(root)]
Viewer(save="save_data.json", files=files)