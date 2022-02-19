import pygame
from math import sin,cos,atan
from random import randint
from time import sleep

def distanceSqr(a,b):
    print(a)
    print(b)
    return (b[0] - a[0])**2 + (b[1] - a[1])**2
    

def closest(coords,pos):
    closest = None
    for c in coords:
        distance = distanceSqr(c,pos)
        if closest == None or distance < closest:
            closest = c
    return closest

class Ant():
    def __init__(self,screen):
        self.screen = screen
        self.x, self.y = self.screen.get_width()/2,self.screen.get_height()/2
        self.rot = 0
        self.speed = 2

        self.followDistance = 5

        self.state = "wander"
        self.home_markers = []
        self.food_markers = []

    def move(self):    
        if self.state == "home":
            best = closest(self.home_markers,(self.x,self.y))
        else:
            best = closest(self.food_markers,(self.x,self.y))

        if best != None and distanceSqr(best,(self.x,self.y)) <= self.followDistance**2:
            self.rot = atan((best[1] - self.y)/(best[0] - self.x)) 
            self.food_markers.append((x,y))
        else:
            self.state = "wander"

        x_pos, y_pos = cos(self.rot)*self.speed, sin(self.rot)*self.speed
        self.x += x_pos
        self.y += y_pos
        self.rot += randint(-100,100)/200

        if self.state == "wander":
            self.home_markers.append((self.x,self.y))
        elif self.state == "home":
            self.food_markers.append((self.x,self.y))

    def draw(self):
        pygame.draw.circle(self.screen,(255,0,0),(int(self.x),int(self.y)),self.speed)

pygame.init()
screen = pygame.display.set_mode((1000,1000))

ants = []
for i in range(100):
    ants.append(Ant(screen))

running = True
while running:
    screen.fill((0,0,0))
    for ant in ants:
        ant.move()
        ant.draw()
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONUP:
            pass
    sleep(1/60)
pygame.display.quit()
