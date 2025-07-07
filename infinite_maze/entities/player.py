import pygame
from ..utils.config import config


class Player:
    def __init__(self, xPosition, yPosition, headless=False):
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

        self.position = (xPosition, yPosition)
        self.width = config.PLAYER_WIDTH
        self.height = config.PLAYER_HEIGHT
        self.speed = config.PLAYER_SPEED

    def setX(self, xPosition):
        self.position = (xPosition, self.position[1])

    def setY(self, yPosition):
        self.position = (self.position[0], yPosition)

    def moveX(self, units):
        new_x = self.position[0] + (units * self.speed)
        self.position = (new_x, self.position[1])

    def moveY(self, units):
        new_y = self.position[1] + (units * self.speed)
        self.position = (self.position[0], new_y)

    def getX(self):
        return self.position[0]

    def getY(self):
        return self.position[1]

    def getPosition(self):
        return self.position

    def getSpeed(self):
        return self.speed

    def setCursor(self, image):
        if not self.headless:
            try:
                self.cursor = pygame.image.load(image)
            except (pygame.error, FileNotFoundError):
                # If image can't be loaded, keep current cursor
                pass

    def getCursor(self):
        return self.cursor

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def reset(self, xPosition, yPosition):
        self.setX(xPosition)
        self.setY(yPosition)
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
                if line.getIsHorizontal():
                    blocked = blocked or (
                        self.getY() <= line.getYStart()
                        and self.getY() + self.getHeight() >= line.getYStart()
                        and self.getX() + self.getWidth() + self.getSpeed() == line.getXStart()
                    )
                else:  # vertical line
                    blocked = blocked or (
                        self.getX() + self.getWidth() <= line.getXStart()
                        and self.getX() + self.getWidth() + self.getSpeed() >= line.getXStart()
                        and (
                            (self.getY() >= line.getYStart() and self.getY() <= line.getYEnd())
                            or (
                                self.getY() + self.getHeight() >= line.getYStart()
                                and self.getY() + self.getHeight() <= line.getYEnd()
                            )
                        )
                    )
        elif direction == LEFT:
            for line in lines:
                if line.getIsHorizontal():
                    blocked = blocked or (
                        self.getY() <= line.getYStart()
                        and self.getY() + self.getHeight() >= line.getYStart()
                        and self.getX() - self.getSpeed() == line.getXEnd()
                    )
                else:  # vertical line
                    blocked = blocked or (
                        self.getX() >= line.getXEnd()
                        and self.getX() - self.getSpeed() <= line.getXEnd()
                        and (
                            (self.getY() >= line.getYStart() and self.getY() <= line.getYEnd())
                            or (
                                self.getY() + self.getHeight() >= line.getYStart()
                                and self.getY() + self.getHeight() <= line.getYEnd()
                            )
                        )
                    )
        elif direction == DOWN:
            for line in lines:
                if line.getIsHorizontal():
                    blocked = blocked or (
                        self.getY() + self.getHeight() <= line.getYStart()
                        and self.getY() + self.getHeight() + self.getSpeed() >= line.getYStart()
                        and (
                            (self.getX() >= line.getXStart() and self.getX() <= line.getXEnd())
                            or (
                                self.getX() + self.getWidth() >= line.getXStart()
                                and self.getX() + self.getWidth() <= line.getXEnd()
                            )
                        )
                    )
                else:  # vertical line
                    blocked = blocked or (
                        self.getX() <= line.getXStart()
                        and self.getX() + self.getWidth() >= line.getXStart()
                        and self.getY() + self.getHeight() + self.getSpeed() == line.getYStart()
                    )
        elif direction == UP:
            for line in lines:
                if line.getIsHorizontal():
                    blocked = blocked or (
                        self.getY() >= line.getYStart()
                        and self.getY() - self.getSpeed() <= line.getYStart()
                        and (
                            (self.getX() >= line.getXStart() and self.getX() <= line.getXEnd())
                            or (
                                self.getX() + self.getWidth() >= line.getXStart()
                                and self.getX() + self.getWidth() <= line.getXEnd()
                            )
                        )
                    )
                else:  # vertical line
                    blocked = blocked or (
                        self.getX() <= line.getXStart()
                        and self.getX() + self.getWidth() >= line.getXStart()
                        and self.getY() - self.getSpeed() == line.getYEnd()
                    )
        
        return blocked
