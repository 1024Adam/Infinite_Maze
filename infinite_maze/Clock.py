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
        return ('{:02d}'.format(self.minutes) + ':' + '{:02d}'.format(self.seconds))
