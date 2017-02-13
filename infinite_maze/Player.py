import pygame
from pygame.locals import *

class Player:
    def __init__(self, xPosition, yPosition):
        self.cursor = pygame.image.load('img/player.png')
        self.position = (xPosition, yPosition)
        self.speed = 10

    def setX(self, xPosition):
        self.position = (xPosition, self.position[1])

    def setY(self, yPosition):
        self.position = (self.position[0], yPosition)
    
    def moveX(self, units):
        self.position = (self.position[0] + (units * self.speed), self.position[1]) 

    def moveY(self, units):
        self.position = (self.position[0], self.position[1] + (units * self.speed)) 

    def getX(self):
        return self.position[0]

    def getY(self):
        return self.position[1]   

    def getPosition(self):
        return self.position

    def getSpeed(self):
        return self.speed

    def getCursor(self):
        return self.cursor
