import pygame
from pygame.locals import *
from Clock import Clock

class Game:
    def __init__(self):
        pygame.init()
        
        self.width = 640
        self.height = 480
        self.xMin = 80
        self.xMax = (self.width / 2)
        self.yMin = 40
        self.yMax = (self.height - 40)

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
        self.over = 0
        self.shutdown = 0

        self.clock = Clock()

    def updateScreen(self, player, lines, lineCount):
        # Paint Screen
        self.screen.fill(self.bgColor)
        self.screen.blit(player.getCursor(), player.getPosition())
        for row in range(lineCount):
            pygame.draw.line(self.getScreen(), self.fgColor, lines[row][0], lines[row][1], 10)

        # Update Clock
        self.clock.update()

        # Print Display Text
        timeText = self.font.render('Time: ' + self.clock.getTimeString(), 1, self.fgColor)
        self.screen.blit(timeText, (10, 10))
        scoreText = self.font.render('Score: ' + str(self.score), 1, self.fgColor)
        self.screen.blit(scoreText, (10, 25))

        pygame.display.flip()
   
    def printEndDisplay(self):
        # Paint Screen
        self.screen.fill(self.bgColor)
        endText = self.font.render('Continue? (y/n)', 1, self.fgColor)

        # Print Display Text
        self.screen.blit(endText, (10, 10))
        
        pygame.display.flip()

    def end(self):
        self.over = 1

    def cleanup(self):
        pygame.quit() 

    def isActive(self):
        return not self.over
    
    def quit(self):
        self.shutdown = 1

    def isPlaying(self):
        return not self.shutdown

    def reset(self):
        self.pace = 0
        self.score = 0
        self.over = 0

        self.clock.reset()

    def getClock(self):
        return self.clock

    def getScreen(self):
        return self.screen
  
    def getScore(self):
        return self.score

    def updateScore(self, amount):
        self.score += amount

    def setScore(self, newScore):
        self.score = newScore

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getXMin(self):
        return self.xMin

    def getYMin(self):
        return self.yMin

    def getXMax(self):
        return self.xMax

    def getYMax(self):
        return self.yMax

    def getFont(self):
        return self.font

    def getBGColor(self):
        return self.bgColor
    
    def getFGColor(self):
        return self.fgColor
