import pygame
from pygame.locals import *
from pygame import time
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
            game.updateScreen(player, lines)     
            prevKeys = keys
            keys = pygame.key.get_pressed()

            if (not game.isPaused()): 
                # Arrow Move Events
                if (keys[pygame.K_RIGHT] or keys[pygame.K_d]):
                    blocked = False
                    for line in lines:
                        if line.getIsHorizontal():
                            blocked = blocked or (
                                player.getY() <= line.getYStart() and
                                player.getY() + player.getHeight() >= line.getYStart() and
                                player.getX() + player.getWidth() + player.getSpeed() == line.getXStart()
                            )  
                        else: # vertical line
                            blocked = blocked or (
                                player.getX() + player.getWidth() <= line.getXStart() and
                                player.getX() + player.getWidth() + player.getSpeed() >= line.getXStart() and
                                (
                                    (player.getY() >= line.getYStart() and player.getY() <= line.getYEnd()) or
                                    (player.getY() + player.getHeight() >= line.getYStart() and player.getY() + player.getHeight() <= line.getYEnd())
                                )
                            )
                    if (not blocked):
                        player.moveX(player.getSpeed())
                        game.incrementScore()
                elif (keys[pygame.K_LEFT] or keys[pygame.K_a]):
                    blocked = False
                    for line in lines:
                        if line.getIsHorizontal():
                            blocked = blocked or (
                                player.getY() <= line.getYStart() and
                                player.getY() + player.getHeight() >= line.getYStart() and
                                player.getX() - player.getSpeed() == line.getXEnd()
                            )  
                        else: # vertical line
                            blocked = blocked or (
                                player.getX() >= line.getXEnd() and
                                player.getX() - player.getSpeed() <= line.getXEnd() and
                                (
                                    (player.getY() >= line.getYStart() and player.getY() <= line.getYEnd()) or
                                    (player.getY() + player.getHeight() >= line.getYStart() and player.getY() + player.getHeight() <= line.getYEnd())
                                )                            
                            )
                    if (not blocked):
                        player.moveX(-player.getSpeed())
                        game.decrementScore()
                if (keys[pygame.K_DOWN] or keys[pygame.K_s]):
                    blocked = False
                    for line in lines:
                        if line.getIsHorizontal():
                            blocked = blocked or (
                                player.getY() + player.getHeight() <= line.getYStart() and
                                player.getY() + player.getHeight() + player.getSpeed() >= line.getYStart() and
                                (
                                    (player.getX() >= line.getXStart() and player.getX() <= line.getXEnd()) or
                                    (player.getX() + player.getWidth() >= line.getXStart() and player.getX() + player.getWidth() <= line.getXEnd())
                                )
                            )  
                        else: # vertical line
                            blocked = blocked or (
                                player.getX() <= line.getXStart() and
                                player.getX() + player.getWidth() >= line.getXStart() and
                                player.getY() + player.getHeight() + player.getSpeed() == line.getYStart()
                            )
                    if (not blocked):
                        player.moveY(player.getSpeed())
                elif (keys[pygame.K_UP] or keys[pygame.K_w]):
                    blocked = False
                    for line in lines:
                        if line.getIsHorizontal():
                            blocked = blocked or (
                                player.getY() >= line.getYStart() and
                                player.getY() - player.getSpeed() <= line.getYStart() and
                                (
                                    (player.getX() >= line.getXStart() and player.getX() <= line.getXEnd()) or
                                    (player.getX() + player.getWidth() >= line.getXStart() and player.getX() + player.getWidth() <= line.getXEnd())
                                )
                            )  
                        else: # vertical line
                            blocked = blocked or (
                                player.getX() <= line.getXStart() and
                                player.getX() + player.getWidth() >= line.getXStart() and
                                player.getY() - player.getSpeed() == line.getYEnd()
                            )
                    if (not blocked):
                        player.moveY(-player.getSpeed())

                # Process game pace adjustments
                if game.getClock().getTicks() % 10 == 0:
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
                    if (start < 80):
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
            if any(event.type == pygame.QUIT for event in pygame.event.get()):
                game.end()

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
        if any(event == pygame.QUIT for event in pygame.event.get()):
            game.quit()

    game.cleanup()
    exit(0)
