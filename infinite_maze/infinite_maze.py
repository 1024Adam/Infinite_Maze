import pygame
from pygame.locals import *

from Player import Player

def maze(): 
    pygame.init()

    # Screen Details
    width = 640
    height = 480
    maxWidthR = (width / 2)
    maxWidthL = 80
    maxHeightU = 40
    maxHeightD = (height - 40)
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
    player = Player('img/player.png', 80, (height / 2), 10)
    
    # Game Config/Stats
    font = pygame.font.SysFont('', 20)
    score = 0
    time = pygame.time.Clock()
    startTime = time.get_time()
    currMills = 0
    currSecs = 0
    currMins = 0
    gameOver = 0
    
    # Maze Details
    lineCount = 6
    lines = [[0 for x in range(2)] for y in range(20)]
    lines[0][0] = (120, 230)
    lines[0][1] = (125, 230)
    lines[1][0] = (125, 230)
    lines[1][1] = (130, 230)
    lines[2][0] = (130, 230)
    lines[2][1] = (135, 230)
    lines[3][0] = (135, 230)
    lines[3][1] = (140, 230)
    lines[4][0] = (190, 265)
    lines[4][1] = (195, 265)
    lines[5][0] = (195, 265)
    lines[5][1] = (200, 265)
    
    while (not gameOver):
        # Paint Screen
        screen.fill(bgColor)
        screen.blit(player.cursor, player.position)
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
                if ((player.position[0] + 10 + player.speed >= lines[row][0][0]) and (player.position[0] + 10 <= lines[row][0][0]) and (player.position[1] + 20 > lines[row][0][1]) and (player.position[1] - 5 < lines[row][0][1])):
                    blocked = 1
            if (not blocked):
                player.xMove(1)
                score += 1
        if (keys[pygame.K_DOWN] or keys[pygame.K_s]):
            blocked = 0
            for row in range(lineCount):
                if ((player.position[1] + 15 + player.speed >= lines[row][0][1]) and (player.position[1] + 15 <= lines[row][0][1]) and (player.position[0] + 15 > lines[row][0][0]) and (player.position[0] - 15 < lines[row][0][0])):
                    blocked = 1
            if (not blocked):
                player.yMove(1)
        if (keys[pygame.K_UP] or keys[pygame.K_w]):
            blocked = 0
            for row in range(lineCount):
                if ((player.position[1] - player.speed <= lines[row][0][1]) and (player.position[1] >= lines[row][0][1]) and (player.position[0] + 15 > lines[row][0][0]) and (player.position[0] - 15 < lines[row][0][0])):
                    blocked = 1
            if (not blocked):
                player.yMove(-1)
        if (keys[pygame.K_LEFT] or keys[pygame.K_a]):
            blocked = 0
            for row in range(lineCount):
                if ((player.position[0] - 5 - player.speed <= lines[row][0][0]) and (player.position[0] - 5 >= lines[row][0][0]) and (player.position[1] + 20 > lines[row][0][1]) and (player.position[1] - 5 < lines[row][0][1])):
                    blocked = 1
            if (not blocked):
                player.xMove(-1)
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
        if (player.position[0] < maxWidthL):
            player.xSet(maxWidthL)
            score = max(score, 0)
            if (score > 0):
                for row in range(lineCount):
                    lines[row][0] = (lines[row][0][0] + player.speed, lines[row][0][1])
                    lines[row][1] = (lines[row][1][0] + player.speed, lines[row][1][1])
        if (player.position[0] > maxWidthR):
            player.xSet(maxWidthR)
            for row in range(lineCount):
                lines[row][0] = (lines[row][0][0] - player.speed, lines[row][0][1])
                lines[row][1] = (lines[row][1][0] - player.speed, lines[row][1][1])
        player.ySet(max(player.position[1], maxHeightU))
        player.ySet(min(player.position[1], maxHeightD))
        
    pygame.quit() 
    exit(0)
