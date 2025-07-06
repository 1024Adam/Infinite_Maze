import pygame
from pygame import time
from ..entities.player import Player
from .game import Game
from ..entities.maze import Line
from ..utils.config import config

DO_NOTHING = config.get_movement_constant("DO_NOTHING")
RIGHT = config.get_movement_constant("RIGHT")
LEFT = config.get_movement_constant("LEFT")
UP = config.get_movement_constant("UP")
DOWN = config.get_movement_constant("DOWN")


def maze():
    # Contains All Game Stats/Config
    game = Game()

    # Player Stats/Position/Details
    player = Player(config.PLAYER_START_X, config.PLAYER_START_Y)

    # Maze Details
    lines = Line.generateMaze(game, config.MAZE_ROWS, config.MAZE_COLS)

    game.getClock().reset()
    keys = pygame.key.get_pressed()
    while game.isPlaying():
        while game.isActive():
            game.updateScreen(player, lines)
            prevKeys = keys
            keys = pygame.key.get_pressed()

            if not game.isPaused():
                # Arrow Move Events
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    blocked = False
                    for line in lines:
                        if line.getIsHorizontal():
                            blocked = blocked or (
                                player.getY() <= line.getYStart()
                                and player.getY() + player.getHeight()
                                >= line.getYStart()
                                and player.getX()
                                + player.getWidth()
                                + player.getSpeed()
                                == line.getXStart()
                            )
                        else:  # vertical line
                            blocked = blocked or (
                                player.getX() + player.getWidth() <= line.getXStart()
                                and player.getX()
                                + player.getWidth()
                                + player.getSpeed()
                                >= line.getXStart()
                                and (
                                    (
                                        player.getY() >= line.getYStart()
                                        and player.getY() <= line.getYEnd()
                                    )
                                    or (
                                        player.getY() + player.getHeight()
                                        >= line.getYStart()
                                        and player.getY() + player.getHeight()
                                        <= line.getYEnd()
                                    )
                                )
                            )
                    if not blocked:
                        player.moveX(player.getSpeed())
                        game.incrementScore()
                elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    blocked = False
                    for line in lines:
                        if line.getIsHorizontal():
                            blocked = blocked or (
                                player.getY() <= line.getYStart()
                                and player.getY() + player.getHeight()
                                >= line.getYStart()
                                and player.getX() - player.getSpeed() == line.getXEnd()
                            )
                        else:  # vertical line
                            blocked = blocked or (
                                player.getX() >= line.getXEnd()
                                and player.getX() - player.getSpeed() <= line.getXEnd()
                                and (
                                    (
                                        player.getY() >= line.getYStart()
                                        and player.getY() <= line.getYEnd()
                                    )
                                    or (
                                        player.getY() + player.getHeight()
                                        >= line.getYStart()
                                        and player.getY() + player.getHeight()
                                        <= line.getYEnd()
                                    )
                                )
                            )
                    if not blocked:
                        player.moveX(-player.getSpeed())
                        game.decrementScore()
                if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    blocked = False
                    for line in lines:
                        if line.getIsHorizontal():
                            blocked = blocked or (
                                player.getY() + player.getHeight() <= line.getYStart()
                                and player.getY()
                                + player.getHeight()
                                + player.getSpeed()
                                >= line.getYStart()
                                and (
                                    (
                                        player.getX() >= line.getXStart()
                                        and player.getX() <= line.getXEnd()
                                    )
                                    or (
                                        player.getX() + player.getWidth()
                                        >= line.getXStart()
                                        and player.getX() + player.getWidth()
                                        <= line.getXEnd()
                                    )
                                )
                            )
                        else:  # vertical line
                            blocked = blocked or (
                                player.getX() <= line.getXStart()
                                and player.getX() + player.getWidth()
                                >= line.getXStart()
                                and player.getY()
                                + player.getHeight()
                                + player.getSpeed()
                                == line.getYStart()
                            )
                    if not blocked:
                        player.moveY(player.getSpeed())
                elif keys[pygame.K_UP] or keys[pygame.K_w]:
                    blocked = False
                    for line in lines:
                        if line.getIsHorizontal():
                            blocked = blocked or (
                                player.getY() >= line.getYStart()
                                and player.getY() - player.getSpeed()
                                <= line.getYStart()
                                and (
                                    (
                                        player.getX() >= line.getXStart()
                                        and player.getX() <= line.getXEnd()
                                    )
                                    or (
                                        player.getX() + player.getWidth()
                                        >= line.getXStart()
                                        and player.getX() + player.getWidth()
                                        <= line.getXEnd()
                                    )
                                )
                            )
                        else:  # vertical line
                            blocked = blocked or (
                                player.getX() <= line.getXStart()
                                and player.getX() + player.getWidth()
                                >= line.getXStart()
                                and player.getY() - player.getSpeed() == line.getYEnd()
                            )
                    if not blocked:
                        player.moveY(-player.getSpeed())

                # Process game pace adjustments
                if game.getClock().getTicks() % 10 == 0:
                    player.setX(player.getX() - game.getPace())
                    for line in lines:
                        line.setXStart(line.getXStart() - game.getPace())
                        line.setXEnd(line.getXEnd() - game.getPace())

                # Position Adjustments (to prevent screen overflow)
                if player.getX() < config.X_MIN:
                    game.end()
                if player.getX() > config.X_MAX:
                    player.setX(config.X_MAX)
                    for line in lines:
                        line.setXStart(line.getXStart() - player.getSpeed())
                        line.setXEnd(line.getXEnd() - player.getSpeed())
                player.setY(max(player.getY(), config.Y_MIN))
                player.setY(min(player.getY(), config.Y_MAX))

                # Reposition lines that have been passed
                xMax = Line.getXMax(lines)
                for line in lines:
                    start = line.getXStart()
                    end = line.getXEnd()
                    if start < config.PLAYER_START_X:
                        line.setXStart(xMax)
                        if start == end:
                            line.setXEnd(xMax)
                        else:
                            line.setXEnd(xMax + config.MAZE_CELL_SIZE)

            # Pause Event
            if prevKeys[pygame.K_SPACE] and not keys[pygame.K_SPACE]:
                game.changePaused(player)

            # Quit Events
            if keys[pygame.K_LMETA] and keys[pygame.K_q]:
                game.end()
            if keys[pygame.K_RMETA] and keys[pygame.K_q]:
                game.end()
            if keys[pygame.K_LALT] and keys[pygame.K_F4]:
                game.end()
            if keys[pygame.K_ESCAPE]:
                game.end()

            # Process Game Events
            if any(event.type == pygame.QUIT for event in pygame.event.get()):
                game.end()

            # Process FPS
            processTime = int(
                game.getClock().getMillis() - game.getClock().getPrevMillis()
            )
            if processTime <= config.FPS_DELAY_MS:
                time.delay(config.FPS_DELAY_MS - processTime)

        # Game has ended
        game.printEndDisplay()

        # Quit Events
        keys = pygame.key.get_pressed()
        if keys[pygame.K_y]:
            game.reset()
            player.reset(config.PLAYER_START_X, (config.SCREEN_HEIGHT / 2))

            # Maze Details
            lines = Line.generateMaze(game, config.MAZE_ROWS, config.MAZE_COLS)

            game.getClock().reset()
        if keys[pygame.K_n]:
            game.quit()
        if keys[pygame.K_LMETA] and keys[pygame.K_q]:
            game.quit()
        if keys[pygame.K_RMETA] and keys[pygame.K_q]:
            game.quit()
        if keys[pygame.K_LALT] and keys[pygame.K_F4]:
            game.quit()

        # Process Game Events
        if any(event == pygame.QUIT for event in pygame.event.get()):
            game.quit()

    game.cleanup()
    exit(0)


def controlled_run(wrapper, counter):
    # Contains All Game Stats/Config
    game = Game()

    # Player Stats/Position/Details
    player = Player(config.PLAYER_START_X, config.PLAYER_START_Y)

    # Maze Details
    lines = Line.generateMaze(game, config.MAZE_ROWS, config.MAZE_COLS)

    game.getClock().reset()
    keys = pygame.key.get_pressed()
    action = None
    while game.isPlaying():
        while game.isActive():
            game.updateScreen(player, lines)
            keys = pygame.key.get_pressed()

            if not game.isPaused():
                values = {}
                scoreIncreased = False

                # Arrow Move Events
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    action = RIGHT
                elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    action = LEFT
                elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    action = DOWN
                elif keys[pygame.K_UP] or keys[pygame.K_w]:
                    action = UP
                elif action is None:
                    action = DO_NOTHING

                values["action"] = action

                if action == RIGHT:
                    blocked = False
                    for line in lines:
                        if line.getIsHorizontal():
                            blocked = blocked or (
                                player.getY() <= line.getYStart()
                                and player.getY() + player.getHeight()
                                >= line.getYStart()
                                and player.getX()
                                + player.getWidth()
                                + player.getSpeed()
                                == line.getXStart()
                            )
                        else:  # vertical line
                            blocked = blocked or (
                                player.getX() + player.getWidth() <= line.getXStart()
                                and player.getX()
                                + player.getWidth()
                                + player.getSpeed()
                                >= line.getXStart()
                                and (
                                    (
                                        player.getY() >= line.getYStart()
                                        and player.getY() <= line.getYEnd()
                                    )
                                    or (
                                        player.getY() + player.getHeight()
                                        >= line.getYStart()
                                        and player.getY() + player.getHeight()
                                        <= line.getYEnd()
                                    )
                                )
                            )
                    if not blocked:
                        player.moveX(player.getSpeed())
                        game.incrementScore()
                        scoreIncreased = True
                elif action == LEFT:
                    blocked = False
                    for line in lines:
                        if line.getIsHorizontal():
                            blocked = blocked or (
                                player.getY() <= line.getYStart()
                                and player.getY() + player.getHeight()
                                >= line.getYStart()
                                and player.getX() - player.getSpeed() == line.getXEnd()
                            )
                        else:  # vertical line
                            blocked = blocked or (
                                player.getX() >= line.getXEnd()
                                and player.getX() - player.getSpeed() <= line.getXEnd()
                                and (
                                    (
                                        player.getY() >= line.getYStart()
                                        and player.getY() <= line.getYEnd()
                                    )
                                    or (
                                        player.getY() + player.getHeight()
                                        >= line.getYStart()
                                        and player.getY() + player.getHeight()
                                        <= line.getYEnd()
                                    )
                                )
                            )
                    if not blocked:
                        player.moveX(-player.getSpeed())
                        game.decrementScore()
                if action == DOWN:
                    blocked = False
                    for line in lines:
                        if line.getIsHorizontal():
                            blocked = blocked or (
                                player.getY() + player.getHeight() <= line.getYStart()
                                and player.getY()
                                + player.getHeight()
                                + player.getSpeed()
                                >= line.getYStart()
                                and (
                                    (
                                        player.getX() >= line.getXStart()
                                        and player.getX() <= line.getXEnd()
                                    )
                                    or (
                                        player.getX() + player.getWidth()
                                        >= line.getXStart()
                                        and player.getX() + player.getWidth()
                                        <= line.getXEnd()
                                    )
                                )
                            )
                        else:  # vertical line
                            blocked = blocked or (
                                player.getX() <= line.getXStart()
                                and player.getX() + player.getWidth()
                                >= line.getXStart()
                                and player.getY()
                                + player.getHeight()
                                + player.getSpeed()
                                == line.getYStart()
                            )
                    if not blocked:
                        player.moveY(player.getSpeed())
                elif action == UP:
                    blocked = False
                    for line in lines:
                        if line.getIsHorizontal():
                            blocked = blocked or (
                                player.getY() >= line.getYStart()
                                and player.getY() - player.getSpeed()
                                <= line.getYStart()
                                and (
                                    (
                                        player.getX() >= line.getXStart()
                                        and player.getX() <= line.getXEnd()
                                    )
                                    or (
                                        player.getX() + player.getWidth()
                                        >= line.getXStart()
                                        and player.getX() + player.getWidth()
                                        <= line.getXEnd()
                                    )
                                )
                            )
                        else:  # vertical line
                            blocked = blocked or (
                                player.getX() <= line.getXStart()
                                and player.getX() + player.getWidth()
                                >= line.getXStart()
                                and player.getY() - player.getSpeed() == line.getYEnd()
                            )
                    if not blocked:
                        player.moveY(-player.getSpeed())

                values["score_increased"] = scoreIncreased

                # Process game pace adjustments
                if game.getClock().getTicks() % 10 == 0:
                    player.setX(player.getX() - game.getPace())
                    for line in lines:
                        line.setXStart(line.getXStart() - game.getPace())
                        line.setXEnd(line.getXEnd() - game.getPace())

                # Position Adjustments (to prevent screen overflow)
                if player.getX() < config.X_MIN:
                    game.end()
                if player.getX() > config.X_MAX:
                    player.setX(config.X_MAX)
                    for line in lines:
                        line.setXStart(line.getXStart() - player.getSpeed())
                        line.setXEnd(line.getXEnd() - player.getSpeed())
                player.setY(max(player.getY(), config.Y_MIN))
                player.setY(min(player.getY(), config.Y_MAX))

                # Reposition lines that have been passed
                xMax = Line.getXMax(lines)
                for line in lines:
                    start = line.getXStart()
                    end = line.getXEnd()
                    if start < config.PLAYER_START_X:
                        line.setXStart(xMax)
                        if start == end:
                            line.setXEnd(xMax)
                        else:
                            line.setXEnd(xMax + config.MAZE_CELL_SIZE)

            # Process FPS
            processTime = int(
                game.getClock().getMillis() - game.getClock().getPrevMillis()
            )
            if processTime <= config.FPS_DELAY_MS:
                time.delay(config.FPS_DELAY_MS - processTime)

            closestLine = 1000
            for line in lines:
                if player.getX() < line.getXStart() and line.getXStart() < closestLine:
                    closestLine = line.getXStart()
            values["closest_line"] = closestLine

            response = wrapper.control(values)
            action = response

        game.quit()

    wrapper.gameover(game.getScore())
