import pygame
from pygame.locals import *
 
pygame.init()

# Screen Details
width = 640
height = 480
screen = pygame.display.set_mode((width, height))
icon = pygame.Surface((32, 32))
iconImage = pygame.image.load('img/icon.png')
icon.blit(iconImage, (0, 0))

pygame.display.set_caption('Infinite Maze')
pygame.display.set_icon(icon)

# Colors
bgColor = pygame.Color(255, 255, 255)
lineColor = pygame.Color(0, 0, 0) 

# Player/Position Details
player = pygame.image.load('img/player.png')
positionx = 5
positiony = height / 2
position = (positionx, positiony)

# Game Config/Stats
moveSpeed = 8
xcollisionSpace = 3 * moveSpeed
ycollisionSpace = 3 * moveSpeed
font = pygame.font.SysFont('', 20)
score = 0
time = pygame.time.Clock()
startTime = time.get_time()
currMills = 0
currSecs = 0
currMins = 0
gameOver = 0

# Maze Details
lineCount = 5
lines = [[0 for x in range(2)] for y in range(20)]
lines[0][0] = (100, 230)
lines[0][1] = (105, 230)
lines[1][0] = (105, 230)
lines[1][1] = (110, 230)
lines[2][0] = (110, 230)
lines[2][1] = (115, 230)
lines[3][0] = (115, 230)
lines[3][1] = (120, 230)
lines[4][0] = (100, 265)
lines[4][1] = (105, 265)

while (not gameOver):
    # Paint Screen
    screen.fill(bgColor)
    screen.blit(player, position)
    for row in range(lineCount):
        pygame.draw.line(screen, Color(0, 0, 0), lines[row][0], lines[row][1], 10)
    time.tick()
    currMills += time.get_time()
    currSecs = currMills / 1000
    if (currSecs >= 60):
        currMins += 1
        currSecs -= 60
        currMills -= 60000
    timeText = font.render('Time: ' + '{:02d}'.format(currMins) + ':' + '{:02d}'.format(currSecs), 1, Color(0, 0, 0))
    screen.blit(timeText, (10, 10))
    scoreText = font.render('Score: ' + str(score), 1, Color(0, 0, 0))
    screen.blit(scoreText, (10, 25))
    pygame.display.flip()
  
    # Arrow Move Events
    keys = pygame.key.get_pressed()
    if (keys[pygame.K_RIGHT] or keys[pygame.K_d]):
        blocked = 0
        for row in range(lineCount):
            if (((positionx + xcollisionSpace) > lines[row][0][0]) and ((positionx - xcollisionSpace) < lines[row][0][0]) and ((positiony - ycollisionSpace) < lines[row][0][1]) and ((positiony + ycollisionSpace) > lines[row][0][1])):
                blocked = 1
        if (not blocked):
            positionx += moveSpeed
            score += 1
    if (keys[pygame.K_DOWN] or keys[pygame.K_s]):
        blocked = 0
        for row in range(lineCount):
            if (((positionx + xcollisionSpace) > lines[row][0][0]) and ((positionx - xcollisionSpace) < lines[row][0][0]) and ((positiony - ycollisionSpace) < lines[row][0][1]) and ((positiony + ycollisionSpace) > lines[row][0][1])):
                blocked = 1
        if (not blocked):
            positiony += moveSpeed
    if (keys[pygame.K_UP] or keys[pygame.K_w]):
        positiony -= moveSpeed
    if (keys[pygame.K_LEFT] or keys[pygame.K_a]):
        positionx -= moveSpeed
        score -= 1
    
    # Quit Events
    if (keys[pygame.K_LMETA] and keys[pygame.K_q]):
        gameOver = 1
    if (keys[pygame.K_RMETA] and keys[pygame.K_q]):
        gameOver = 1
    if (keys[pygame.K_LALT] and keys[pygame.K_F4]):
        gameOver = 1
    if (keys[pygame.K_ESCAPE]):
        gameOver = 1
    
    # Process Game Events
    for event in pygame.event.get():
        if (event.type == pygame.QUIT):
            gameOver = 1
   
    # Position Adjustments (to prevent screen overflow) 
    if (positionx < 5):
        positionx = 5
        for row in range(lineCount):
            lines[row][0] = (lines[row][0][0] + moveSpeed, lines[row][0][1])
            lines[row][1] = (lines[row][1][0] + moveSpeed, lines[row][1][1])
        score += 1
    if (positionx > 310):
        positionx = 310
        for row in range(lineCount):
            lines[row][0] = (lines[row][0][0] - moveSpeed, lines[row][0][1])
            lines[row][1] = (lines[row][1][0] - moveSpeed, lines[row][1][1])
    if (positiony < 5):
        positiony = 5
    if (positiony > 460):
        positiony = 460    
    position = (positionx, positiony)

pygame.quit() 
exit(0)
