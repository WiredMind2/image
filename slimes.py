import pygame
from random import random,randint
from time import sleep
from math import sqrt
import os,cv2

class Slime():
    def __init__(self,size,x=None,y=None,vel=None,col=None):
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
        self.randCol = lambda : [int(self.x/size*255),int(self.y/size*255),int((self.x+self.y)/(size*2)*255)]
        if col:
            self.col = col
        else:
            self.col = [255,255,255]

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

    def pos(self):
        return (int(self.x),int(self.y))

class SlimesAnim():
    def __init__(self,slimeAmount,spawnAmount,size,record=False,folder=None):
        pygame.init()
        self.screen = pygame.display.set_mode((size,size))
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
        dark = pygame.Surface(self.screen.get_size()).convert_alpha()
        dark.fill((0,0,0, darken_percent*255))
        self.screen.blit(dark, (0, 0))

        #Computing blur
        #pixels = pygame.surfarray.array2d(self.screen)
        """color_surf = pygame.Surface((3,3))
        for x in range(1,self.screen.get_height()-1,1):
            for y in range(1,self.screen.get_width()-1,1):
                avg = pygame.transform.average_color(self.screen,pygame.Rect(x-1,y-1,3,3))
                color_surf.fill(avg)
                self.screen.blit(color_surf,(x-1,y-1))"""
        """blurScreen = pygame.transform.smoothscale(self.screen.copy(),(int(self.size/blurFactor),int(self.size/blurFactor)))
        blurScreen = pygame.transform.smoothscale(blurScreen,(self.size,self.size))
        self.screen.blit(blurScreen,(0,0))"""
        

        for s in self.slimes:
            #self.screen.set_at(s.pos(),(255,255,255))
            pygame.draw.circle(self.screen,s.col, s.pos(), 1)

        if self.record:
            pygame.image.save(self.screen,os.path.join(self.folder, "image{}.png".format(self.lastindex)))
            self.video.write(cv2.imread(os.path.join(self.folder, "image{}.png".format(self.lastindex))))
            
            self.lastindex += 1
        pygame.display.flip()

s = SlimesAnim(0,10,1000,True,"C:/Users/willi/Pictures/background/")
