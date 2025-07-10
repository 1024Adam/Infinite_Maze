import pygame


class Clock:
    def __init__(self):
        self.time = pygame.time.Clock()
        self.start_time = self.time.get_time()

        self.prev_millis = 0
        self.millis = 0
        self.millis_paused = 0
        self.ticks = 0

    def update(self):
        self.prev_millis = self.millis
        self.time.tick()
        self.millis += self.time.get_time()
        self.tick()

    def get_time_string(self):
        minutes = int(self.millis / 60000)
        seconds = int((self.millis / 1000) % 60)
        return f"{minutes:02}" + ":" + f"{seconds:02}"

    def get_fps(self):
        return self.time.get_fps()

    def reset(self):
        self.time = pygame.time.Clock()
        self.start_time = self.time.get_time()

        self.prev_millis = 0
        self.millis = 0
        self.ticks = 0

    def get_prev_millis(self):
        return self.prev_millis

    def get_millis(self):
        return self.millis

    def get_seconds(self):
        return int((self.millis / 1000) % 60)

    def get_prev_seconds(self):
        return int((self.prev_millis / 1000) % 60)

    def rollback_millis(self, rollback):
        self.millis -= rollback

    def get_millis_paused(self):
        return self.millis_paused

    def set_millis_paused(self, millis):
        self.millis_paused = millis

    def tick(self):
        self.ticks += 1

    def get_ticks(self):
        return self.ticks
