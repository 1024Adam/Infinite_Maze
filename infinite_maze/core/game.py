import pygame
from typing import List, Optional

from .clock import Clock
from ..utils.config import config
from ..entities.player import Player
from ..entities.maze import Line


class Game:
    # Display config
    WIDTH = config.SCREEN_WIDTH
    HEIGHT = config.SCREEN_HEIGHT
    X_MIN = config.PLAYER_START_X
    Y_MIN = 40

    X_MAX = WIDTH / 2
    Y_MAX = HEIGHT - config.ICON_SIZE

    SCORE_INCREMENT = 1

    BG_COLOR = pygame.Color(255, 255, 255)
    FG_COLOR = pygame.Color(0, 0, 0)

    def __init__(self, headless: bool = False) -> None:
        self.font: Optional[pygame.font.Font]
        self.screen: Optional[pygame.Surface]
        self.icon: Optional[pygame.Surface]
        if not headless:
            pygame.init()

            # Font Config
            self.font = pygame.font.SysFont("", config.PLAYER_HEIGHT * 2)

            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

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

    def updateScreen(self, player: Player, lines: List[Line]) -> None:
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
                and self.clock.getSeconds() % config.PACE_UPDATE_INTERVAL == 0
                and self.clock.getPrevSeconds() != self.clock.getSeconds()
            ):
                self.pace += 1
            return

        if self.screen is None or self.font is None:
            return

        screen = self.screen

        # Paint Screen
        screen.fill(self.BG_COLOR)
        cursor = player.getCursor()
        if cursor is not None:
            screen.blit(cursor, player.getPosition())
        for line in lines:
            pygame.draw.line(screen, self.FG_COLOR, line.getStart(), line.getEnd(), 1)

        # Update Clock/Ticks
        self.clock.update()
        if self.paused:
            self.clock.rollbackMillis(
                self.clock.getMillis() - self.clock.getPrevMillis()
            )

        # Update Pace
        if (
            not self.paused
            and self.clock.getSeconds() % config.PACE_UPDATE_INTERVAL == 0
            and self.clock.getPrevSeconds() != self.clock.getSeconds()
        ):
            self.pace += 1

        # Print Border
        pygame.draw.line(
            screen,
            self.FG_COLOR,
            (self.X_MIN, self.Y_MIN),
            (self.WIDTH, self.Y_MIN),
            2,
        )
        pygame.draw.line(
            screen,
            self.FG_COLOR,
            (self.X_MIN, self.Y_MAX + config.BORDER_OFFSET),
            (self.WIDTH, self.Y_MAX + config.BORDER_OFFSET),
            2,
        )
        pygame.draw.line(
            screen,
            self.FG_COLOR,
            (config.PLAYER_START_X, self.Y_MIN),
            (config.PLAYER_START_X, self.Y_MAX + config.BORDER_OFFSET),
            2,
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

    def printEndDisplay(self) -> None:
        if self.screen is None or self.font is None:
            return

        # Paint Screen
        self.screen.fill(self.BG_COLOR)
        endText = self.font.render("Continue? (y/n)", 1, self.FG_COLOR)
        score_text = f"Score: {self.score}"
        scoreText = self.font.render(score_text, 1, self.FG_COLOR)

        # Print Display Text
        self.screen.blit(endText, (config.TEXT_MARGIN, config.TEXT_MARGIN))
        self.screen.blit(scoreText, (config.TEXT_MARGIN, config.TEXT_MARGIN + 15))

        pygame.display.flip()

    def end(self) -> None:
        self.over = True

    def cleanup(self) -> None:
        pygame.quit()

    def isActive(self) -> bool:
        return not self.over

    def quit(self) -> None:
        self.shutdown = True

    def isPlaying(self) -> bool:
        return not self.shutdown

    def reset(self) -> None:
        self.pace = 0
        self.score = 0
        self.over = False

        self.clock.reset()

    def getClock(self) -> Clock:
        return self.clock

    def getScreen(self) -> Optional[pygame.Surface]:
        return self.screen

    def getScore(self) -> int:
        return self.score

    def updateScore(self, amount: int) -> None:
        self.score += amount

    def incrementScore(self) -> None:
        self.score += self.SCORE_INCREMENT

    def decrementScore(self) -> None:
        self.score -= self.SCORE_INCREMENT if self.score > 0 else 0

    def setScore(self, newScore: int) -> None:
        self.score = newScore

    def isPaused(self) -> bool:
        return self.paused

    def changePaused(self, player: Player) -> None:
        self.paused = not self.paused
        if self.paused:
            self.FG_COLOR = pygame.Color(128, 128, 128)
            player.setCursor(config.get_image_path("player_paused"))
        else:
            self.FG_COLOR = pygame.Color(0, 0, 0)
            player.setCursor(config.get_image_path("player"))

    def getPace(self) -> int:
        return self.pace

    def setPace(self, newPace: int) -> None:
        self.pace = newPace
