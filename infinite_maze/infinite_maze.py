import pygame
from pygame.locals import *
from Player import Player
from Game import Game

def maze(): 
    # Contains All Game Stats/Config
    game = Game() 
     
    # Player Stats/Position/Details
    player = Player(80, (game.getHeight() / 2))
    
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
    
    while (game.isActive()):
        game.updateScreen(player, lines, lineCount)     
 
        # Arrow Move Events
        keys = pygame.key.get_pressed()
        if (keys[pygame.K_RIGHT] or keys[pygame.K_d]):
            blocked = 0
            for row in range(lineCount):
                if ((player.getX() + 10 + player.getSpeed() >= lines[row][0][0]) and (player.getX() + 10 <= lines[row][0][0]) and (player.getY() + 20 > lines[row][0][1]) and (player.getY() - 5 < lines[row][0][1])):
                    blocked = 1
            if (not blocked):
                player.moveX(1)
                game.updateScore(1)
        if (keys[pygame.K_DOWN] or keys[pygame.K_s]):
            blocked = 0
            for row in range(lineCount):
                if ((player.getY() + 15 + player.getSpeed() >= lines[row][0][1]) and (player.getY() + 15 <= lines[row][0][1]) and (player.getX() + 15 > lines[row][0][0]) and (player.getX() - 15 < lines[row][0][0])):
                    blocked = 1
            if (not blocked):
                player.moveY(1)
        if (keys[pygame.K_UP] or keys[pygame.K_w]):
            blocked = 0
            for row in range(lineCount):
                if ((player.getY() - player.getSpeed() <= lines[row][0][1]) and (player.getY() >= lines[row][0][1]) and (player.getX() + 15 > lines[row][0][0]) and (player.getX() - 15 < lines[row][0][0])):
                    blocked = 1
            if (not blocked):
                player.moveY(-1)
        if (keys[pygame.K_LEFT] or keys[pygame.K_a]):
            blocked = 0
            for row in range(lineCount):
                if ((player.getX() - 5 - player.getSpeed() <= lines[row][0][0]) and (player.getX() - 5 >= lines[row][0][0]) and (player.getY() + 20 > lines[row][0][1]) and (player.getY() - 5 < lines[row][0][1])):
                    blocked = 1
            if (not blocked):
                player.moveX(-1)
                game.updateScore(-1)
        
        # Quit Events
        if (keys[pygame.K_LMETA] and keys[pygame.K_q]):
            game.end()
        if (keys[pygame.K_RMETA] and keys[pygame.K_q]):
            game.end()
        if (keys[pygame.K_LALT] and keys[pygame.K_F4]):
            game.end()
        if (keys[pygame.K_ESCAPE]):
            game.end()
        
        # Process Game Events
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                game.end()
       
        # Position Adjustments (to prevent screen overflow) 
        if (player.getX() < game.getXMin()):
            player.setX(game.getXMin())
            game.setScore(max(game.getScore(), 0))
            if (game.getScore() > 0):
                for row in range(lineCount):
                    lines[row][0] = (lines[row][0][0] + player.getSpeed(), lines[row][0][1])
                    lines[row][1] = (lines[row][1][0] + player.getSpeed(), lines[row][1][1])
        if (player.getX() > game.getXMax()):
            player.setX(game.getXMax())
            for row in range(lineCount):
                lines[row][0] = (lines[row][0][0] - player.getSpeed(), lines[row][0][1])
                lines[row][1] = (lines[row][1][0] - player.getSpeed(), lines[row][1][1])
        player.setY(max(player.getY(), game.getYMin()))
        player.setY(min(player.getY(), game.getYMax()))
        
    game.cleanup()
    exit(0)
