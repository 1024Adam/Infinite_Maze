import pygame
from pygame.locals import *
from Clock import Clock
from Line import Line

class Game:
    def __init__(self):
        pygame.init()
        
        self.width = 640
        self.height = 480
        self.xMin = 80
        self.xMax = (self.width / 2)
        self.yMin = 40
        self.yMax = (self.height - 32)

        self.screen = pygame.display.set_mode((self.width, self.height))
        
        self.icon = pygame.Surface((32, 32))
        iconImage = pygame.image.load('img/icon.png')
        self.icon.blit(iconImage, (0, 0))
    
        pygame.display.set_caption('Infinite Maze')
        pygame.display.set_icon(self.icon)
   
        # Font Config
        self.font = pygame.font.SysFont('', 20)
        self.bgColor = pygame.Color(255, 255, 255)
        self.fgColor = pygame.Color(0, 0, 0) 

        self.pace = 0
        self.score = 0
        self.scoreIncrement = 1
        
        self.paused = False
        self.over = False
        self.shutdown = False

        self.clock = Clock()

    def updateScreen(self, player, lines):
        # Paint Screen
        self.screen.fill(self.bgColor)
        self.screen.blit(player.getCursor(), player.getPosition())
        for line in lines:
            pygame.draw.line(self.getScreen(), self.fgColor, line.getStart(), line.getEnd(), 1)

        prevMillis = self.clock.getMillis()
        prevSeconds = self.clock.getSeconds()
        # Update Clock
        self.clock.update()

        if self.paused:
            self.clock.rollbackMillis(self.clock.getMillis() - prevMillis)
        
        # Update Pace
        if not self.paused and self.clock.getMillis() > 10000 and self.clock.getSeconds() % 30 == 0 and prevSeconds != self.clock.getSeconds():
            self.pace += 0.1

        # Print Border
        pygame.draw.line(self.getScreen(), self.fgColor, (self.xMin, self.yMin), (self.width, self.yMin), 2)
        pygame.draw.line(self.getScreen(), self.fgColor, (self.xMin, self.yMax + 10), (self.width, self.yMax + 10), 2)
        pygame.draw.line(self.getScreen(), self.fgColor, (80, self.yMin), (80, self.yMax + 10), 2)
        
        # Print Display Text
        timeText = self.font.render('Time: ' + self.clock.getTimeString(), 1, self.fgColor)
        self.screen.blit(timeText, (10, 10))
        scoreText = self.font.render('Score: ' + str(self.score), 1, self.fgColor)
        self.screen.blit(scoreText, (10, 25))
      
        if (self.paused):
            scoreText = self.font.render('Paused (press space to continue)', 1, self.fgColor)
            self.screen.blit(scoreText, (100, 10))

        pygame.display.flip()
   
    def printEndDisplay(self):
        # Paint Screen
        self.screen.fill(self.bgColor)
        endText = self.font.render('Continue? (y/n)', 1, self.fgColor)
        scoreText = self.font.render('Score: ' + str(self.score), 1, self.fgColor)

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
        self.score += self.scoreIncrement
        
    def decrementScore(self):
        self.score -= self.scoreIncrement if self.score > 0 else 0

    def setScore(self, newScore):
        self.score = newScore

    def isPaused(self):
        return (self.paused)

    def changePaused(self, player):
        self.paused = not self.paused
        if (self.paused):
            self.fgColor = pygame.Color(128, 128, 128)
            player.setCursor('img/player_paused.png')
        else:
            self.fgColor = pygame.Color(0, 0, 0) 
            player.setCursor('img/player.png')

    def getWidth(self):
        return (self.width)

    def getHeight(self):
        return (self.height)

    def getXMin(self):
        return (self.xMin)

    def getYMin(self):
        return (self.yMin)

    def getXMax(self):
        return (self.xMax)

    def getYMax(self):
        return (self.yMax)

    def getFont(self):
        return (self.font)

    def getBGColor(self):
        return (self.bgColor)
    
    def getFGColor(self):
        return (self.fgColor)

    def getPace(self):
        return (self.pace)

    def setPace(self, newPace):
        self.pace = newPace
