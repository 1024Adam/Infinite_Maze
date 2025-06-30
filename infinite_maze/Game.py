import pygame
from pygame.locals import *
from .Clock import Clock
from .Line import Line

class Game:
    # Display config
    WIDTH = 640
    HEIGHT = 480
    X_MIN = 80
    Y_MIN = 40

    X_MAX = (WIDTH / 2)
    Y_MAX = (HEIGHT - 32)

    SCORE_INCREMENT = 1

    BG_COLOR = pygame.Color(255, 255, 255)
    FG_COLOR = pygame.Color(0, 0, 0) 

    def __init__(self, headless=False):
        if not headless:
            pygame.init()

            # Font Config
            self.font = pygame.font.SysFont('', 20)

            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            
            self.icon = pygame.Surface((32, 32))
            try:
                iconImage = pygame.image.load('img/icon.png')
                self.icon.blit(iconImage, (0, 0))
            except (pygame.error, FileNotFoundError):
                # If icon can't be loaded, use a default surface
                self.icon.fill(pygame.Color(255, 255, 255))
        
            pygame.display.set_caption('Infinite Maze')
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
                self.clock.rollbackMillis(self.clock.getMillis() - self.clock.getPrevMillis())
            
            # Update Pace
            if not self.paused and self.clock.getMillis() > 10000 and self.clock.getSeconds() % 30 == 0 and self.clock.getPrevSeconds() != self.clock.getSeconds():
                self.pace += 1
            return
            
        # Paint Screen
        self.screen.fill(self.BG_COLOR)
        self.screen.blit(player.getCursor(), player.getPosition())
        for line in lines:
            pygame.draw.line(self.getScreen(), self.FG_COLOR, line.getStart(), line.getEnd(), 1)

        # Update Clock/Ticks
        self.clock.update()
        if self.paused:
            self.clock.rollbackMillis(self.clock.getMillis() - self.clock.getPrevMillis())
        
        # Update Pace
        if not self.paused and self.clock.getMillis() > 10000 and self.clock.getSeconds() % 30 == 0 and self.clock.getPrevSeconds() != self.clock.getSeconds():
            self.pace += 1

        # Print Border
        pygame.draw.line(self.getScreen(), self.FG_COLOR, (self.X_MIN, self.Y_MIN), (self.WIDTH, self.Y_MIN), 2)
        pygame.draw.line(self.getScreen(), self.FG_COLOR, (self.X_MIN, self.Y_MAX + 10), (self.WIDTH, self.Y_MAX + 10), 2)
        pygame.draw.line(self.getScreen(), self.FG_COLOR, (80, self.Y_MIN), (80, self.Y_MAX + 10), 2)
        
        # Print Display Text
        timeText = self.font.render('Time: ' + self.clock.getTimeString(), 1, self.FG_COLOR)
        self.screen.blit(timeText, (10, 10))
        scoreText = self.font.render('Score: ' + str(self.score), 1, self.FG_COLOR)
        self.screen.blit(scoreText, (10, 25))
      
        if (self.paused):
            scoreText = self.font.render('Paused (press space to continue)', 1, self.FG_COLOR)
            self.screen.blit(scoreText, (100, 10))

        pygame.display.flip()
   
    def printEndDisplay(self):
        # Paint Screen
        self.screen.fill(self.BG_COLOR)
        endText = self.font.render('Continue? (y/n)', 1, self.FG_COLOR)
        scoreText = self.font.render('Score: ' + str(self.score), 1, self.FG_COLOR)

        # Print Display Text
        self.screen.blit(endText, (10, 10))
        self.screen.blit(scoreText, (10, 25))
        
        pygame.display.flip()

    def end(self):
        self.over = True

    def cleanup(self):
        pygame.quit() 

    def isActive(self):
        return (not self.over)
    
    def quit(self):
        self.shutdown = True

    def isPlaying(self):
        return (not self.shutdown)

    def reset(self):
        self.pace = 0
        self.score = 0
        self.over = False

        self.clock.reset()

    def getClock(self):
        return (self.clock)

    def getScreen(self):
        return (self.screen)
  
    def getScore(self):
        return (self.score)

    def updateScore(self, amount):
        self.score += amount
        
    def incrementScore(self):
        self.score += self.SCORE_INCREMENT
        
    def decrementScore(self):
        self.score -= self.SCORE_INCREMENT if self.score > 0 else 0

    def setScore(self, newScore):
        self.score = newScore

    def isPaused(self):
        return (self.paused)

    def changePaused(self, player):
        self.paused = not self.paused
        if (self.paused):
            self.FG_COLOR = pygame.Color(128, 128, 128)
            player.setCursor('img/player_paused.png')
        else:
            self.FG_COLOR = pygame.Color(0, 0, 0) 
            player.setCursor('img/player.png')

    def getPace(self):
        return (self.pace)

    def setPace(self, newPace):
        self.pace = newPace
