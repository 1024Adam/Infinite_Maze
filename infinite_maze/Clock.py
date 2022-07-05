import pygame
from pygame.locals import *

class Clock:
    def __init__(self):
        self.time = pygame.time.Clock()
        self.startTime = self.time.get_time()
        
        self.milliseconds = 0

    def update(self):
        self.time.tick()
        self.millis += self.time.get_time()

    def getTimeString(self):
        minutes = int(self.millis / 60000)
        seconds = int((self.millis / 1000) % 60)
        return (f'{minutes:02}' + ':' + f'{seconds:02}')

    def reset(self):
        self.time = pygame.time.Clock()
        self.startTime = self.time.get_time()
        
        self.millis = 0

    def getMillis(self):
        return (self.millis)
    
    def getSeconds(self):
        return (self.millis / 60)
