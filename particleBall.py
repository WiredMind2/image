import pygame
import win32api
import win32con
import win32gui
from random import random,randint
from time import sleep
from math import sqrt
import os,cv2

class Slime():
    def __init__(self,size,x=None,y=None,vel=None,col=None):
        self.alive = True
        
        if x:
            self.x = x
        else:
            if size:
                self.x = size/2
            else:
                self.x = 0
        if y:
            self.y = y
        else:
            if size:
                self.y = size/2
            else:
                self.y = 0
        if vel:
            self.vel = vel
        else:
            self.vel = []
            for i in range(2):
                velx,vely = 0,0
                while 0.5 > (abs(velx)+abs(vely)) or (abs(velx)+abs(vely)) > 1.5:
                    velx = random()*2-1
                    vely = random()*2-1
                self.vel = [velx,vely]

        #self.randCol = lambda : [randint(0, 255), randint(0, 255), randint(0, 255)]
        self.randCol = lambda : [int(self.x/size*255),int(self.y/size*255),int((self.x+self.y)/(size*2)*255),255]
        if col:
            self.col = col
        else:
            self.col = [255,255,255,255]

        self.life = 255

        self.size = size

    def move(self):
        """if self.x >= self.size or self.x <= 0:
            self.vel[0] *= -1 + random()/5 -0.1
            self.col = self.randCol()
        if self.y >= self.size or self.y <= 0:
            self.vel[1] *= -1 + random()/5 -0.1
            self.col = self.randCol()"""

        self.x += self.vel[0]
        self.y += self.vel[1]

        self.life -= 2
        if self.life <= 0:
            self.alive = False
        else:
            self.col[3] = self.life

    def pos(self):
        return (int(self.x),int(self.y))

class SlimesAnim():
    def __init__(self,slimeAmount,spawnAmount,size,record=False,folder=None):
        pygame.init()
        self.display = pygame.display.set_mode((size,size), 0, 32)

        self.screen = self.display.copy().convert_alpha()
        self.screen.fill((0,0,0,0))
        pygame.display.update()

        hwnd = pygame.display.get_wm_info()["window"]
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                               win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)
        # Set window transparency color
        #win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0,0,0), 0, win32con.LWA_COLORKEY)

        
        self.size = size
        self.spawnAmount = spawnAmount

        self.record = record
        self.folder = folder
        self.lastindex = 0

        if self.record:
            video_name = 'video.avi'

            height, width = size,size

            self.video = cv2.VideoWriter(video_name, 0, 60, (width,height))

        self.slimes = []
        for i in range(slimeAmount):
            s = Slime(size=size)
            self.slimes.append(s)

        self.running = True
        while self.running:
            #pygame.time.wait(int(1/60*1000))
            self.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
        pygame.quit()

        if record:
            cv2.destroyAllWindows()
            self.video.release()

    def update(self):
        for i in range(self.spawnAmount):
            self.slimes.append(Slime(size=self.size))
        for s in self.slimes:
            s.move()
        self.render()

    def render(self):
        #Reddit - fading old pixels from the screen
        darken_percent = .03
        #dark = pygame.Surface(self.screen.get_size()).convert_alpha()
        #dark.fill((0,0,0, darken_percent*255))
        dark = self.screen.convert_alpha()
        dark.set_alpha(darken_percent*255)
        self.screen = dark.copy()
        #self.screen.blit(dark, (0, 0))
       

        for s in self.slimes:
            #self.screen.set_at(s.pos(),(255,255,255))
            if s.alive:
                pygame.draw.circle(self.screen,s.col, s.pos(), 1)
            else:
                self.slimes.remove(s)

        if self.record:
            pygame.image.save(self.screen.convert_alpha(),os.path.join(self.folder, "image{}.png".format(self.lastindex)))
            self.video.write(cv2.imread(os.path.join(self.folder, "image{}.png".format(self.lastindex))))
            
            self.lastindex += 1
        self.display.fill((0,0,0,0))
        self.display.blit(self.screen,(0,0))
        pygame.display.flip()

s = SlimesAnim(0,10,1000,True,"C:/Users/willi/Pictures/background/")
