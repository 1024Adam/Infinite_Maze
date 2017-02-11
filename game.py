import pygame
from pygame.locals import *
 
pygame.init()

width = 640
height = 480
screen = pygame.display.set_mode((width, height))
bgColor = pygame.Color(255, 255, 255)
lineColor = pygame.Color(0, 0, 0) 
 
player = pygame.image.load('img/player.png')
positionx = 5
positiony = height / 2
position = (positionx, positiony)

moveSpeed = 10

gameOver = 0

while (not gameOver):
    screen.fill(bgColor)
    screen.blit(player, position)
    pygame.display.flip()
  
    keys = pygame.key.get_pressed()
    if (keys[pygame.K_UP]):
        positiony -= moveSpeed
    if (keys[pygame.K_DOWN]):
        positiony += moveSpeed
    if (keys[pygame.K_LEFT]):
        positionx -= moveSpeed
    if (keys[pygame.K_RIGHT]):
        positionx += moveSpeed
    
    for event in pygame.event.get():
        if (event.type == pygame.QUIT):
            gameOver = 1
    
    if (positionx < 5):
        positionx = 5
    if (positionx > 620):
        positionx = 620
    if (positiony < 5):
        positiony = 5
    if (positiony > 460):
        positiony = 460    
    position = (positionx, positiony)

pygame.quit() 
exit(0)
