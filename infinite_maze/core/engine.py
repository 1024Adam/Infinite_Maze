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
    lines = Line.generate_maze(game, config.MAZE_ROWS, config.MAZE_COLS)

    game.get_clock().reset()
    keys = pygame.key.get_pressed()
    while game.is_playing():
        while game.is_active():
            game.update_screen(player, lines)
            prev_keys = keys
            keys = pygame.key.get_pressed()

            if not game.is_paused():
                # Arrow Move Events
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    if not player.is_movement_blocked(RIGHT, lines):
                        player.move_x(player.get_speed())
                        game.increment_score()
                elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    if not player.is_movement_blocked(LEFT, lines):
                        player.move_x(-player.get_speed())
                        game.decrement_score()
                if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    if not player.is_movement_blocked(DOWN, lines):
                        player.move_y(player.get_speed())
                elif keys[pygame.K_UP] or keys[pygame.K_w]:
                    if not player.is_movement_blocked(UP, lines):
                        player.move_y(-player.get_speed())

                # Process game pace adjustments
                if game.get_clock().get_ticks() % 10 == 0:
                    player.set_x(player.get_x() - game.get_pace())
                    for line in lines:
                        line.set_x_start(line.get_x_start() - game.get_pace())
                        line.set_x_end(line.get_x_end() - game.get_pace())

                # Position Adjustments (to prevent screen overflow)
                if player.get_x() < config.X_MIN:
                    game.end()
                if player.get_x() > config.X_MAX:
                    player.set_x(config.X_MAX)
                    for line in lines:
                        line.set_x_start(line.get_x_start() - player.get_speed())
                        line.set_x_end(line.get_x_end() - player.get_speed())
                player.set_y(max(player.get_y(), config.Y_MIN))
                player.set_y(min(player.get_y(), config.Y_MAX))

                # Reposition lines that have been passed
                x_max = Line.get_x_max(lines)
                for line in lines:
                    start = line.get_x_start()
                    end = line.get_x_end()
                    if start < config.PLAYER_START_X:
                        line.set_x_start(x_max)
                        if start == end:
                            line.set_x_end(x_max)
                        else:
                            line.set_x_end(x_max + config.MAZE_CELL_SIZE)

            # Pause Event
            if prev_keys[pygame.K_SPACE] and not keys[pygame.K_SPACE]:
                game.change_paused(player)

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
            process_time = int(
                game.get_clock().get_millis() - game.get_clock().get_prev_millis()
            )
            if process_time <= config.FPS_DELAY_MS:
                time.delay(config.FPS_DELAY_MS - process_time)

        # Game has ended
        game.print_end_display()

        # Quit Events
        keys = pygame.key.get_pressed()
        if keys[pygame.K_y]:
            game.reset()
            player.reset(config.PLAYER_START_X, (config.SCREEN_HEIGHT / 2))

            # Maze Details
            lines = Line.generate_maze(game, config.MAZE_ROWS, config.MAZE_COLS)

            game.get_clock().reset()
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
    lines = Line.generate_maze(game, config.MAZE_ROWS, config.MAZE_COLS)

    game.get_clock().reset()
    keys = pygame.key.get_pressed()
    action = None
    while game.is_playing():
        while game.is_active():
            game.update_screen(player, lines)
            keys = pygame.key.get_pressed()

            if not game.is_paused():
                values = {}
                score_increased = False

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
                    if not player.is_movement_blocked(RIGHT, lines):
                        player.move_x(player.get_speed())
                        game.increment_score()
                        score_increased = True
                elif action == LEFT:
                    if not player.is_movement_blocked(LEFT, lines):
                        player.move_x(-player.get_speed())
                        game.decrement_score()
                if action == DOWN:
                    if not player.is_movement_blocked(DOWN, lines):
                        player.move_y(player.get_speed())
                elif action == UP:
                    if not player.is_movement_blocked(UP, lines):
                        player.move_y(-player.get_speed())

                values["score_increased"] = score_increased

                # Process game pace adjustments
                if game.get_clock().get_ticks() % 10 == 0:
                    player.set_x(player.get_x() - game.get_pace())
                    for line in lines:
                        line.set_x_start(line.get_x_start() - game.get_pace())
                        line.set_x_end(line.get_x_end() - game.get_pace())

                # Position Adjustments (to prevent screen overflow)
                if player.get_x() < config.X_MIN:
                    game.end()
                if player.get_x() > config.X_MAX:
                    player.set_x(config.X_MAX)
                    for line in lines:
                        line.set_x_start(line.get_x_start() - player.get_speed())
                        line.set_x_end(line.get_x_end() - player.get_speed())
                player.set_y(max(player.get_y(), config.Y_MIN))
                player.set_y(min(player.get_y(), config.Y_MAX))

                # Reposition lines that have been passed
                x_max = Line.get_x_max(lines)
                for line in lines:
                    start = line.get_x_start()
                    end = line.get_x_end()
                    if start < config.PLAYER_START_X:
                        line.set_x_start(x_max)
                        if start == end:
                            line.set_x_end(x_max)
                        else:
                            line.set_x_end(x_max + config.MAZE_CELL_SIZE)

            # Process FPS
            process_time = int(
                game.get_clock().get_millis() - game.get_clock().get_prev_millis()
            )
            if process_time <= config.FPS_DELAY_MS:
                time.delay(config.FPS_DELAY_MS - process_time)

            closest_line = 1000
            for line in lines:
                if player.get_x() < line.get_x_start() and line.get_x_start() < closest_line:
                    closest_line = line.get_x_start()
            values["closest_line"] = closest_line

            response = wrapper.control(values)
            action = response

        game.quit()

    wrapper.gameover(game.get_score())
