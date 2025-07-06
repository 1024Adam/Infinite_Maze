import pygame
from .clock import Clock
from ..utils.config import config


class Game:
    BG_COLOR = pygame.Color(255, 255, 255)
    FG_COLOR = pygame.Color(0, 0, 0)

    def __init__(self, headless=False):
        if not headless:
            pygame.init()

            # Font Config
            self.font = pygame.font.SysFont("", config.PLAYER_HEIGHT * 2)

            self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

            self.icon = pygame.Surface((config.ICON_SIZE, config.ICON_SIZE))
            try:
                iconImage = pygame.image.load(config.get_image_path("icon"))
                self.icon.blit(iconImage, (0, 0))
            except (pygame.error, FileNotFoundError):
                # If icon can't be loaded, use a default surface
                self.icon.fill(pygame.Color(255, 255, 255))

            pygame.display.set_caption("Infinite Maze")
            pygame.display.set_icon(self.icon)
        else:
            # Headless mode - minimal pygame initialization
            pygame.init()
            self.font = None
            self.screen = None
            self.icon = None

        self.pace = 0
        self.score = 0

        self.paused = False
        self.over = False
        self.shutdown = False
        self.headless = headless

        self.clock = Clock()

    def updateScreen(self, player, lines):
        if self.headless:
            # In headless mode, just update the clock and pace
            self.clock.update()
            if self.paused:
                self.clock.rollbackMillis(
                    self.clock.getMillis() - self.clock.getPrevMillis()
                )

            # Update Pace
            if (
                not self.paused
                and self.clock.getMillis() > 10000
                and self.clock.getSeconds() % config.PACE_UPDATE_INTERVAL == 0
                and self.clock.getPrevSeconds() != self.clock.getSeconds()
            ):
                self.pace += 1
            return

        # Paint Screen
        self.screen.fill(self.BG_COLOR)
        self.screen.blit(player.getCursor(), player.getPosition())
        for line in lines:
            pygame.draw.line(
                self.getScreen(), self.FG_COLOR, line.getStart(), line.getEnd(), 1
            )

        # Update Clock/Ticks
        self.clock.update()
        if self.paused:
            self.clock.rollbackMillis(
                self.clock.getMillis() - self.clock.getPrevMillis()
            )

        # Update Pace
        if (
            not self.paused
            and self.clock.getMillis() > 10000
            and self.clock.getSeconds() % config.PACE_UPDATE_INTERVAL == 0
            and self.clock.getPrevSeconds() != self.clock.getSeconds()
        ):
            self.pace += 1

        # Print Border
        pygame.draw.line(
            self.getScreen(),
            self.FG_COLOR,
            (config.X_MIN, config.Y_MIN),
            (config.SCREEN_WIDTH, config.Y_MIN),
            2,
        )
        pygame.draw.line(
            self.getScreen(),
            self.FG_COLOR,
            (config.X_MIN, config.Y_MAX + config.BORDER_OFFSET),
            (config.SCREEN_WIDTH, config.Y_MAX + config.BORDER_OFFSET),
            2,
        )
        pygame.draw.line(
            self.getScreen(), self.FG_COLOR, (config.PLAYER_START_X, config.Y_MIN), (config.PLAYER_START_X, config.Y_MAX + config.BORDER_OFFSET), 2
        )

        # Print Display Text
        timeText = self.font.render(
            "Time: " + self.clock.getTimeString(), 1, self.FG_COLOR
        )
        self.screen.blit(timeText, (config.TEXT_MARGIN, config.TEXT_MARGIN))
        score_text = f"Score: {self.score}"
        scoreText = self.font.render(score_text, 1, self.FG_COLOR)
        self.screen.blit(scoreText, (config.TEXT_MARGIN, config.TEXT_MARGIN + 15))

        if self.paused:
            scoreText = self.font.render(
                "Paused (press space to continue)", 1, self.FG_COLOR
            )
            self.screen.blit(scoreText, (100, config.TEXT_MARGIN))

        pygame.display.flip()

    def printEndDisplay(self):
        # Paint Screen
        self.screen.fill(self.BG_COLOR)
        endText = self.font.render("Continue? (y/n)", 1, self.FG_COLOR)
        score_text = f"Score: {self.score}"
        scoreText = self.font.render(score_text, 1, self.FG_COLOR)

        # Print Display Text
        self.screen.blit(endText, (config.TEXT_MARGIN, config.TEXT_MARGIN))
        self.screen.blit(scoreText, (config.TEXT_MARGIN, config.TEXT_MARGIN + 15))

        pygame.display.flip()

    def end(self):
        self.over = True

    def cleanup(self):
        pygame.quit()

    def isActive(self):
        return not self.over

    def quit(self):
        self.shutdown = True

    def isPlaying(self):
        return not self.shutdown

    def reset(self):
        self.pace = 0
        self.score = 0
        self.over = False

        self.clock.reset()

    def getClock(self):
        return self.clock

    def getScreen(self):
        return self.screen

    def getScore(self):
        return self.score

    def updateScore(self, amount):
        self.score += amount

    def incrementScore(self):
        self.score += config.SCORE_INCREMENT

    def decrementScore(self):
        self.score -= config.SCORE_INCREMENT if self.score > 0 else 0

    def setScore(self, newScore):
        self.score = newScore

    def isPaused(self):
        return self.paused

    def changePaused(self, player):
        self.paused = not self.paused
        if self.paused:
            self.FG_COLOR = pygame.Color(128, 128, 128)
            player.setCursor(config.get_image_path("player_paused"))
        else:
            self.FG_COLOR = pygame.Color(0, 0, 0)
            player.setCursor(config.get_image_path("player"))

    def getPace(self):
        return self.pace

    def setPace(self, newPace):
        self.pace = newPace
