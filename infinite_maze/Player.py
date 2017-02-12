import pygame
from pygame.locals import *

class Player:
    def __init__(self, cursorImage, xPosition, yPosition, moveSpeed):
        self.cursor = pygame.image.load(cursorImage)
        self.position = (xPosition, yPosition)
        self.speed = moveSpeed

    def xSet(self, xPosition):
        self.position = (xPosition, self.position[1])

    def ySet(self, yPosition):
        self.position = (self.position[0], yPosition)
    
    def xMove(self, units):
        self.position = (self.position[0] + (units * self.speed), self.position[1]) 

    def yMove(self, units):
        self.position = (self.position[0], self.position[1] + (units * self.speed)) 
    
