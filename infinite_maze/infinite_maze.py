import pygame
from pygame.locals import *
from pygame.time import *
from Player import Player
from Game import Game
from Line import Line

def maze(): 
    # Contains All Game Stats/Config
    game = Game() 
 
    # Player Stats/Position/Details
    player = Player(80, 223)
    
    # Maze Details
    lines = Line.generateMaze(game, 15, 20)

    game.getClock().reset()
    keys = pygame.key.get_pressed()

    while (game.isPlaying()): 
        while (game.isActive()):
            mills = game.getClock().getFullTime()
            game.updateScreen(player, lines)     
            
            prevKeys = keys
            keys = pygame.key.get_pressed()

            if (not game.isPaused()): 
                # Arrow Move Events
                if (keys[pygame.K_RIGHT] or keys[pygame.K_d]):
                    blocked = 0
                    for line in lines:
                        if ((player.getX() + player.getSpeed() >= line.getXStart()) and (player.getX() <= line.getXStart()) and (player.getY() + 6 >= line.getYStart()) and (player.getY() - 6 <= line.getYStart())):
                            blocked = 1
                        if ((player.getX() + 6 + player.getSpeed() >= line.getXStart()) and (player.getX() <= line.getXStart()) and (player.getY() >= line.getYStart()) and (player.getY() <= line.getYEnd())):
                            blocked = 1
                    if (not blocked):
                        player.moveX(1)
                        game.updateScore(1)
                if (keys[pygame.K_DOWN] or keys[pygame.K_s]):
                    blocked = 0
                    for line in lines:
                        if ((player.getY() + 5 + player.getSpeed() >= line.getYStart()) and (player.getY() <= line.getYStart()) and (player.getX() + 6 > line.getXStart()) and (player.getX() - 6 < line.getXStart())):
                            blocked = 1
                        if ((player.getY() + 6 + player.getSpeed() >= line.getYStart()) and (player.getY() <= line.getYStart()) and (player.getX() >= line.getXStart()) and (player.getX() <= line.getXEnd())):
                            blocked = 1
                    if (not blocked):
                        player.moveY(1)
                if (keys[pygame.K_UP] or keys[pygame.K_w]):
                    blocked = 0
                    for line in lines:
                        if ((player.getY() - player.getSpeed() <= line.getYEnd()) and (player.getY() >= line.getYEnd()) and (player.getX() + 6 > line.getXStart()) and (player.getX() - 6 < line.getXStart())):
                            blocked = 1
                        if ((player.getY() - 6 - player.getSpeed() <= line.getYEnd()) and (player.getY() >= line.getYEnd()) and (player.getX() >= line.getXStart()) and (player.getX() <= line.getXEnd())):
                            blocked = 1
                    if (not blocked):
                        player.moveY(-1)
                if (keys[pygame.K_LEFT] or keys[pygame.K_a]):
                    blocked = 0
                    for line in lines:
                        if ((player.getX() + 1 - player.getSpeed() <= line.getXEnd()) and (player.getX() >= line.getXEnd()) and (player.getY() + 6 >= line.getYStart()) and (player.getY() - 6 <= line.getYStart())):
                            blocked = 1
                        if ((player.getX() - player.getSpeed() <= line.getXEnd()) and (player.getX() >= line.getXEnd()) and (player.getY() >= line.getYStart()) and (player.getY() <= line.getYEnd())):
                            blocked = 1
                    if (not blocked):
                        player.moveX(-1)
                        game.updateScore(-1)
            
                # Process game pace adjustments
                player.setX(player.getX() - game.getPace())
                for line in lines:
                    line.setXStart(line.getXStart() - game.getPace())
                    line.setXEnd(line.getXEnd() - game.getPace())

                # Position Adjustments (to prevent screen overflow) 
                if (player.getX() < game.getXMin()):
                    game.end()
                if (player.getX() > game.getXMax()):
                    player.setX(game.getXMax())
                    for line in lines:
                        line.setXStart(line.getXStart() - player.getSpeed())
                        line.setXEnd(line.getXEnd() - player.getSpeed())
                player.setY(max(player.getY(), game.getYMin()))
                player.setY(min(player.getY(), game.getYMax()))

                # Reposition lines that have been passed 
                xMax = Line.getXMax(lines)
                for line in lines:
                    start = line.getXStart()
                    end = line.getXEnd()
                    if (start < 0):
                        line.setXStart(xMax)
                        if (start == end):
                            line.setXEnd(xMax)
                        else: 
                            line.setXEnd(xMax + 22)
    
            # Pause Event
            if (prevKeys[pygame.K_SPACE] and not keys[pygame.K_SPACE]):
                game.changePaused(player)
    
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

            # Process FPS
            processTime = game.getClock().getFullTime() - mills
            if (processTime < 16):
               delay(16 - processTime)
       

        # Game has ended
        game.printEndDisplay() 
        # Quit Events
        keys = pygame.key.get_pressed()
        if (keys[pygame.K_y]):
            game.reset();
            player.reset(80, (game.getHeight() / 2))

            # Maze Details
            lines = Line.generateMaze(game, 15, 20)
            
            game.getClock().reset()
        
        if (keys[pygame.K_n]):
            game.quit()
        if (keys[pygame.K_LMETA] and keys[pygame.K_q]):
            game.quit()
        if (keys[pygame.K_RMETA] and keys[pygame.K_q]):
            game.quit()
        if (keys[pygame.K_LALT] and keys[pygame.K_F4]):
            game.quit()
        
        # Process Game Events
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                game.quit()
        
    game.cleanup()
    exit(0)
