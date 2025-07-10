import pygame
from .clock import Clock
from ..utils.config import config


class Game:
    # Display config
    # WIDTH = config.SCREEN_WIDTH
    # HEIGHT = config.SCREEN_HEIGHT
    # X_MIN = config.PLAYER_START_X
    # Y_MIN = 40

    # X_MAX = WIDTH / 2
    # Y_MAX = HEIGHT - config.ICON_SIZE

    # SCORE_INCREMENT = 1

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
                icon_image = pygame.image.load(config.get_image_path("icon"))
                self.icon.blit(icon_image, (0, 0))
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

    def update_screen(self, player, lines):
        if self.headless:
            # In headless mode, just update the clock and pace
            self.clock.update()
            if self.paused:
                self.clock.rollback_millis(
                    self.clock.get_millis() - self.clock.get_prev_millis()
                )

            # Update Pace
            if (
                not self.paused
                and self.clock.get_millis() > 10000
                and self.clock.get_seconds() % config.PACE_UPDATE_INTERVAL == 0
                and self.clock.get_prev_seconds() != self.clock.get_seconds()
            ):
                self.pace += 1
            return

        # Paint Screen
        self.screen.fill(self.BG_COLOR)
        self.screen.blit(player.get_cursor(), player.get_position())
        for line in lines:
            pygame.draw.line(
                self.get_screen(), self.FG_COLOR, line.get_start(), line.get_end(), 1
            )

        # Update Clock/Ticks
        self.clock.update()
        if self.paused:
            self.clock.rollback_millis(
                self.clock.get_millis() - self.clock.get_prev_millis()
            )

        # Update Pace
        if (
            not self.paused
            and self.clock.get_millis() > 10000
            and self.clock.get_seconds() % config.PACE_UPDATE_INTERVAL == 0
            and self.clock.get_prev_seconds() != self.clock.get_seconds()
        ):
            self.pace += 1

        # Print Border
        pygame.draw.line(
            self.get_screen(),
            self.FG_COLOR,
            (config.X_MIN, config.Y_MIN),
            (config.SCREEN_WIDTH, config.Y_MIN),
            2,
        )
        pygame.draw.line(
            self.get_screen(),
            self.FG_COLOR,
            (config.X_MIN, config.Y_MAX + config.BORDER_OFFSET),
            (config.SCREEN_WIDTH, config.Y_MAX + config.BORDER_OFFSET),
            2,
        )
        pygame.draw.line(
            self.get_screen(), self.FG_COLOR, (config.PLAYER_START_X, config.Y_MIN), (config.PLAYER_START_X, config.Y_MAX + config.BORDER_OFFSET), 2
        )

        # Print Display Text
        time_text = self.font.render(
            "Time: " + self.clock.get_time_string(), 1, self.FG_COLOR
        )
        self.screen.blit(time_text, (config.TEXT_MARGIN, config.TEXT_MARGIN))
        score_text = f"Score: {self.score}"
        score_text = self.font.render(score_text, 1, self.FG_COLOR)
        self.screen.blit(score_text, (config.TEXT_MARGIN, config.TEXT_MARGIN + 15))

        if self.paused:
            score_text = self.font.render(
                "Paused (press space to continue)", 1, self.FG_COLOR
            )
            self.screen.blit(score_text, (100, config.TEXT_MARGIN))

        pygame.display.flip()

    def print_end_display(self):
        # Paint Screen
        self.screen.fill(self.BG_COLOR)
        end_text = self.font.render("Continue? (y/n)", 1, self.FG_COLOR)
        score_text = f"Score: {self.score}"
        score_text = self.font.render(score_text, 1, self.FG_COLOR)

        # Print Display Text
        self.screen.blit(end_text, (config.TEXT_MARGIN, config.TEXT_MARGIN))
        self.screen.blit(score_text, (config.TEXT_MARGIN, config.TEXT_MARGIN + 15))

        pygame.display.flip()

    def end(self):
        self.over = True

    def cleanup(self):
        pygame.quit()

    def is_active(self):
        return not self.over

    def quit(self):
        self.shutdown = True

    def is_playing(self):
        return not self.shutdown

    def reset(self):
        self.pace = 0
        self.score = 0
        self.over = False

        self.clock.reset()

    def get_clock(self):
        return self.clock

    def get_screen(self):
        return self.screen

    def get_score(self):
        return self.score

    def update_score(self, amount):
        self.score += amount

    def increment_score(self):
        self.score += config.SCORE_INCREMENT

    def decrement_score(self):
        self.score -= config.SCORE_INCREMENT if self.score > 0 else 0

    def set_score(self, new_score):
        self.score = new_score

    def is_paused(self):
        return self.paused

    def change_paused(self, player):
        self.paused = not self.paused
        if self.paused:
            self.FG_COLOR = pygame.Color(128, 128, 128)
            player.set_cursor(config.get_image_path("player_paused"))
        else:
            self.FG_COLOR = pygame.Color(0, 0, 0)
            player.set_cursor(config.get_image_path("player"))

    def get_pace(self):
        return self.pace

    def set_pace(self, new_pace):
        self.pace = new_pace
