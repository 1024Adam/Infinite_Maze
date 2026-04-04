import pygame


class Clock:
    def __init__(self) -> None:
        self.time = pygame.time.Clock()
        self.startTime = self.time.get_time()

        self.prevMillis = 0
        self.millis = 0
        self.millisPaused = 0
        self.ticks = 0

    def update(self) -> None:
        self.prevMillis = self.millis
        self.time.tick()
        self.millis += self.time.get_time()
        self.tick()

    def getTimeString(self) -> str:
        minutes = int(self.millis / 60000)
        seconds = int((self.millis / 1000) % 60)
        return f"{minutes:02}" + ":" + f"{seconds:02}"

    def getFps(self) -> float:
        return self.time.get_fps()

    def reset(self) -> None:
        self.time = pygame.time.Clock()
        self.startTime = self.time.get_time()

        self.prevMillis = 0
        self.millis = 0
        self.millisPaused = 0
        self.ticks = 0

    def getPrevMillis(self) -> int:
        return self.prevMillis

    def getMillis(self) -> int:
        return self.millis

    def getSeconds(self) -> int:
        return int((self.millis / 1000) % 60)

    def getPrevSeconds(self) -> int:
        return int((self.prevMillis / 1000) % 60)

    def rollbackMillis(self, rollback: int) -> None:
        self.millis -= rollback

    def getMillisPaused(self) -> int:
        return self.millisPaused

    def setMillisPaused(self, millis: int) -> None:
        self.millisPaused = millis

    def tick(self) -> None:
        self.ticks += 1

    def getTicks(self) -> int:
        return self.ticks
