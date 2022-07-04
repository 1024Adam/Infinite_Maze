import pygame
from pygame.locals import *

class Clock:
    def __init__(self):
        self.time = pygame.time.Clock()
        self.startTime = self.time.get_time()
        
        self.milliseconds = 0
        self.seconds = 0
        self.minutes = 0

    def update(self):
        self.time.tick()
        self.milliseconds += self.time.get_time()
        self.seconds = self.milliseconds / 1000
        if (self.seconds >= 60):
            self.minutes += 1
            self.seconds -= 60
            self.milliseconds -= 60000

    def getTimeString(self):
        return ('{:02f}'.format(self.minutes) + ':' + '{:02f}'.format(self.seconds))

    def reset(self):
        self.time = pygame.time.Clock()
        self.startTime = self.time.get_time()
        
        self.milliseconds = 0
        self.seconds = 0
        self.minutes = 0

    def getMills(self):
        return (self.milliseconds)

    def getSecs(self):
        return (self.seconds)

    def getMins(self):
        return (self.minutes)

    def getFullTime(self):
        return (self.milliseconds + (self.seconds * 1000) + (self.minutes * 60000))
