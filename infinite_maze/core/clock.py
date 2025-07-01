import pygame


class Clock:
    def __init__(self):
        self.time = pygame.time.Clock()
        self.startTime = self.time.get_time()

        self.prevMillis = 0
        self.millis = 0
        self.millisPaused = 0
        self.ticks = 0

    def update(self):
        self.prevMillis = self.millis
        self.time.tick()
        self.millis += self.time.get_time()
        self.tick()

    def getTimeString(self):
        minutes = int(self.millis / 60000)
        seconds = int((self.millis / 1000) % 60)
        return f"{minutes:02}" + ":" + f"{seconds:02}"

    def getFps(self):
        return self.time.get_fps()

    def reset(self):
        self.time = pygame.time.Clock()
        self.startTime = self.time.get_time()

        self.prevMillis = 0
        self.millis = 0
        self.ticks = 0

    def getPrevMillis(self):
        return self.prevMillis

    def getMillis(self):
        return self.millis

    def getSeconds(self):
        return int((self.millis / 1000) % 60)

    def getPrevSeconds(self):
        return int((self.prevMillis / 1000) % 60)

    def rollbackMillis(self, rollback):
        self.millis -= rollback

    def getMillisPaused(self):
        return self.millisPaused

    def setMillisPaused(self, millis):
        self.millisPaused = millis

    def tick(self):
        self.ticks += 1

    def getTicks(self):
        return self.ticks
