import pygame
from ..utils.config import config


class Player:
    def __init__(self, x_position, y_position, headless=False):
        self.headless = headless
        if not headless:
            try:
                image_path = config.get_image_path("player")
                self.cursor = pygame.image.load(image_path)
            except (pygame.error, FileNotFoundError):
                # If player image can't be loaded, create a simple shape
                self.cursor = pygame.Surface((config.PLAYER_WIDTH, config.PLAYER_HEIGHT))
                self.cursor.fill(pygame.Color(255, 0, 0))  # Red square
        else:
            self.cursor = None

        self.position = (x_position, y_position)
        self.width = config.PLAYER_WIDTH
        self.height = config.PLAYER_HEIGHT
        self.speed = config.PLAYER_SPEED

    def set_x(self, x_position):
        self.position = (x_position, self.position[1])

    def set_y(self, y_position):
        self.position = (self.position[0], y_position)

    def move_x(self, units):
        new_x = self.position[0] + (units * self.speed)
        self.position = (new_x, self.position[1])

    def move_y(self, units):
        new_y = self.position[1] + (units * self.speed)
        self.position = (self.position[0], new_y)

    def get_x(self):
        return self.position[0]

    def get_y(self):
        return self.position[1]

    def get_position(self):
        return self.position

    def get_speed(self):
        return self.speed

    def set_cursor(self, image):
        if not self.headless:
            try:
                self.cursor = pygame.image.load(image)
            except (pygame.error, FileNotFoundError):
                # If image can't be loaded, keep current cursor
                pass

    def get_cursor(self):
        return self.cursor

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def reset(self, x_position, y_position):
        self.set_x(x_position)
        self.set_y(y_position)
        self.speed = config.PLAYER_SPEED
        
    def is_movement_blocked(self, direction, lines):
        """
        Check if movement in a specified direction is blocked by any maze lines.
        
        Args:
            direction: An integer representing the direction (from config.MOVEMENT_CONSTANTS)
            lines: List of Line objects that make up the maze
            
        Returns:
            bool: True if movement is blocked, False otherwise
        """
        RIGHT = config.get_movement_constant("RIGHT")
        LEFT = config.get_movement_constant("LEFT")
        UP = config.get_movement_constant("UP")
        DOWN = config.get_movement_constant("DOWN")
        
        blocked = False
        
        if direction == RIGHT:
            for line in lines:
                if line.get_is_horizontal():
                    blocked = blocked or (
                        self.get_y() <= line.get_y_start()
                        and self.get_y() + self.get_height() >= line.get_y_start()
                        and self.get_x() + self.get_width() + self.get_speed() == line.get_x_start()
                    )
                else:  # vertical line
                    blocked = blocked or (
                        self.get_x() + self.get_width() <= line.get_x_start()
                        and self.get_x() + self.get_width() + self.get_speed() >= line.get_x_start()
                        and (
                            (self.get_y() >= line.get_y_start() and self.get_y() <= line.get_y_end())
                            or (
                                self.get_y() + self.get_height() >= line.get_y_start()
                                and self.get_y() + self.get_height() <= line.get_y_end()
                            )
                        )
                    )
        elif direction == LEFT:
            for line in lines:
                if line.get_is_horizontal():
                    blocked = blocked or (
                        self.get_y() <= line.get_y_start()
                        and self.get_y() + self.get_height() >= line.get_y_start()
                        and self.get_x() - self.get_speed() == line.get_x_end()
                    )
                else:  # vertical line
                    blocked = blocked or (
                        self.get_x() >= line.get_x_end()
                        and self.get_x() - self.get_speed() <= line.get_x_end()
                        and (
                            (self.get_y() >= line.get_y_start() and self.get_y() <= line.get_y_end())
                            or (
                                self.get_y() + self.get_height() >= line.get_y_start()
                                and self.get_y() + self.get_height() <= line.get_y_end()
                            )
                        )
                    )
        elif direction == DOWN:
            for line in lines:
                if line.get_is_horizontal():
                    blocked = blocked or (
                        self.get_y() + self.get_height() <= line.get_y_start()
                        and self.get_y() + self.get_height() + self.get_speed() >= line.get_y_start()
                        and (
                            (self.get_x() >= line.get_x_start() and self.get_x() <= line.get_x_end())
                            or (
                                self.get_x() + self.get_width() >= line.get_x_start()
                                and self.get_x() + self.get_width() <= line.get_x_end()
                            )
                        )
                    )
                else:  # vertical line
                    blocked = blocked or (
                        self.get_x() <= line.get_x_start()
                        and self.get_x() + self.get_width() >= line.get_x_start()
                        and self.get_y() + self.get_height() + self.get_speed() == line.get_y_start()
                    )
        elif direction == UP:
            for line in lines:
                if line.get_is_horizontal():
                    blocked = blocked or (
                        self.get_y() >= line.get_y_start()
                        and self.get_y() - self.get_speed() <= line.get_y_start()
                        and (
                            (self.get_x() >= line.get_x_start() and self.get_x() <= line.get_x_end())
                            or (
                                self.get_x() + self.get_width() >= line.get_x_start()
                                and self.get_x() + self.get_width() <= line.get_x_end()
                            )
                        )
                    )
                else:  # vertical line
                    blocked = blocked or (
                        self.get_x() <= line.get_x_start()
                        and self.get_x() + self.get_width() >= line.get_x_start()
                        and self.get_y() - self.get_speed() == line.get_y_end()
                    )
        
        return blocked
