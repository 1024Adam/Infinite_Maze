import pygame
from pygame.locals import *

class Player:
    def __init__(self, xPosition, yPosition, headless=False):
        self.headless = headless
        if not headless:
            try:
                self.cursor = pygame.image.load('img/player.png')
            except (pygame.error, FileNotFoundError):
                # If player image can't be loaded, create a simple colored surface
                self.cursor = pygame.Surface((10, 10))
                self.cursor.fill(pygame.Color(255, 0, 0))  # Red square
        else:
            self.cursor = None
            
        self.position = (xPosition, yPosition)
        self.width = 10
        self.height = 10
        self.speed = 1

    def setX(self, xPosition):
        self.position = (xPosition, self.position[1])

    def setY(self, yPosition):
        self.position = (self.position[0], yPosition)
    
    def moveX(self, units):
        self.position = (self.position[0] + (units * self.speed), self.position[1]) 

    def moveY(self, units):
        self.position = (self.position[0], self.position[1] + (units * self.speed)) 

    def getX(self):
        return (self.position[0])

    def getY(self):
        return (self.position[1])

    def getPosition(self):
        return (self.position)

    def getSpeed(self):
        return (self.speed)

    def setCursor(self, image):
        if not self.headless:
            try:
                self.cursor = pygame.image.load(image)
            except (pygame.error, FileNotFoundError):
                # If image can't be loaded, keep current cursor
                pass

    def getCursor(self):
        return (self.cursor)
        
    def getWidth(self):
        return self.width
    
    def getHeight(self):
        return self.height

    def reset(self, xPosition, yPosition):
        self.setX(xPosition)
        self.setY(yPosition)
        self.speed = 1
