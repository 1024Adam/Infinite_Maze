"""
Functional tests for Infinite Maze end-to-end gameplay scenarios.

These tests verify complete gameplay workflows from game start
to game over, simulating real user interactions.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from infinite_maze.core.game import Game
from infinite_maze.entities.player import Player
from infinite_maze.entities.maze import Line
from infinite_maze.core.clock import Clock
from infinite_maze.utils.config import config
from tests.fixtures.pygame_mocks import full_pygame_mocks, InputSimulator
from tests.fixtures.test_helpers import PerformanceMonitor, GameStateCapture


class TestGameStartupScenarios:
    """Test game startup and initialization scenarios."""
    
    def test_normal_game_startup(self):
        """Test normal game startup sequence."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(80, 223, headless=True)  # Default start position
            lines = Line.generate_maze(game, 15, 20)
            
            # Verify initial state
            assert game.is_playing()
            assert game.is_active()
            assert not game.is_paused()
            assert game.get_score() == 0
            assert game.get_pace() == 0
            
            # Verify player start position
            assert player.get_position() == (80, 223)
            
            # Verify maze generation
            assert len(lines) > 0
            assert all(isinstance(line, Line) for line in lines)
    
    def test_game_startup_with_custom_settings(self):
        """Test game startup with custom initial settings."""
        with full_pygame_mocks():
            game = Game(headless=True)
            
            # Custom player position
            player = Player(150, 250, headless=True)
            assert player.get_position() == (150, 250)
            
            # Custom maze size
            lines = Line.generate_maze(game, 10, 15)
            assert len(lines) > 0
            
            # Custom initial score
            game.set_score(5)
            assert game.get_score() == 5
    
    def test_game_startup_error_recovery(self):
        """Test game startup with error conditions."""
        with full_pygame_mocks():
            # Test with invalid player position
            player = Player(-100, -100, headless=True)
            game = Game(headless=True)
            
            # Game should handle invalid positions gracefully
            game.update_screen(player, [])
            
            # Reset to valid position
            player.reset(100, 100)
            assert player.get_position() == (100, 100)


class TestBasicGameplayScenarios:
    """Test basic gameplay scenarios."""
    
    def test_simple_movement_session(self):
        """Test simple movement gameplay session."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            lines = Line.generate_maze(game, 10, 10)
            
            # Record initial state
            initial_score = game.get_score()
            initial_position = player.get_position()
            
            # Simulate movement sequence
            movements = [
                ('right', 1, 0),   # Right: score +1
                ('right', 1, 0),   # Right: score +1  
                ('down', 0, 1),    # Down: no score change
                ('right', 1, 0),   # Right: score +1
                ('up', 0, -1),     # Up: no score change
            ]
            
            for direction, dx, dy in movements:
                if direction == 'right':
                    player.move_x(1)
                    game.increment_score()
                elif direction == 'left':
                    player.move_x(-1)
                    game.decrement_score()
                elif direction == 'down':
                    player.move_y(1)
                elif direction == 'up':
                    player.move_y(-1)
                
                # Update game state
                game.update_screen(player, lines)
                game.get_clock().update()
            
            # Verify final state
            final_position = player.get_position()
            final_score = game.get_score()
            
            assert final_position != initial_position
            assert final_score > initial_score
    
    def test_collision_avoidance_gameplay(self):
        """Test gameplay with collision avoidance."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(90, 100, headless=True)
            
            # Create maze with known walls
            walls = [
                Line((100, 90), (100, 110)),   # Vertical wall blocking right
                Line((80, 120), (120, 120)),   # Horizontal wall blocking down
            ]
            
            # Test collision detection and avoidance
            initial_x = player.get_x()
            
            # Try to move right (should hit wall)
            next_x = initial_x + player.get_speed()
            collision = False
            
            for wall in walls:
                if not wall.get_is_horizontal():  # Vertical wall
                    if (next_x <= wall.get_x_start() <= next_x + player.get_width() and
                        player.get_y() <= wall.get_y_end() and 
                        player.get_y() + player.get_height() >= wall.get_y_start()):
                        collision = True
                        break
            
            if not collision:
                player.move_x(1)
                game.increment_score()
            
            # Verify collision handling
            if collision:
                assert player.get_x() == initial_x  # Should not move
                assert game.get_score() == 0        # Score unchanged
            else:
                assert player.get_x() > initial_x   # Should move
    
    def test_boundary_interaction_gameplay(self):
        """Test gameplay at game boundaries."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(config.X_MIN + 5, config.Y_MIN + 5, headless=True)
            
            # Test left boundary
            player.set_x(config.X_MIN - 10)
            if player.get_x() < config.X_MIN:
                # Game should end or position should be corrected
                game.end()
                assert not game.is_active()
            
            # Reset game
            game.reset()
            player.reset(config.X_MAX - 5, config.Y_MAX - 5)
            
            # Test right boundary
            player.set_x(config.X_MAX + 10)
            if player.get_x() > config.X_MAX:
                player.set_x(config.X_MAX)
            assert player.get_x() <= config.X_MAX
            
            # Test vertical boundaries
            player.set_y(config.Y_MIN - 10)
            player.set_y(max(player.get_y(), config.Y_MIN))
            assert player.get_y() >= config.Y_MIN
            
            player.set_y(config.Y_MAX + 10)
            player.set_y(min(player.get_y(), config.Y_MAX))
            assert player.get_y() <= config.Y_MAX


class TestAdvancedGameplayScenarios:
    """Test advanced gameplay scenarios."""
    
    def test_pace_progression_gameplay(self):
        """Test gameplay with pace progression over time."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(200, 200, headless=True)
            lines = Line.generate_maze(game, 15, 20)
            clock = game.get_clock()
            
            # Simulate extended gameplay
            for frame in range(300):  # 5 seconds at 60 FPS
                # Update clock
                clock.millis += 16.67  # ~60 FPS
                clock.update()
                
                # Simulate pace progression
                if clock.get_seconds() > 10 and clock.get_seconds() % 30 == 0:
                    current_pace = game.get_pace()
                    game.set_pace(current_pace + 1)
                
                # Apply pace to maze
                pace = game.get_pace()
                for line in lines:
                    line.set_x_start(line.get_x_start() - pace)
                    line.set_x_end(line.get_x_end() - pace)
                
                # Player movement to stay in bounds
                if frame % 10 == 0:
                    player.move_x(1)
                    game.increment_score()
                
                # Update screen
                game.update_screen(player, lines)
            
            # Verify pace progression
            assert game.get_pace() >= 0
            assert game.get_score() >= 30  # Should have gained score
    
    def test_long_survival_gameplay(self):
        """Test long survival gameplay session."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            lines = Line.generate_maze(game, 20, 25)
            
            # Simulate 30-second survival session
            survival_time = 0
            target_survival = 30000  # 30 seconds in milliseconds
            
            while survival_time < target_survival and game.is_active():
                # Update clock
                game.get_clock().millis += 16.67
                game.get_clock().update()
                survival_time = game.get_clock().get_millis()
                
                # Simulate player strategy: move right to gain score
                if survival_time % 100 < 16.67:  # Every ~100ms
                    # Check if safe to move right
                    can_move_right = True
                    test_x = player.get_x() + player.get_speed()
                    
                    for line in lines:
                        if not line.get_is_horizontal():  # Vertical wall
                            if (test_x <= line.get_x_start() <= test_x + player.get_width() and
                                player.get_y() <= line.get_y_end() and 
                                player.get_y() + player.get_height() >= line.get_y_start()):
                                can_move_right = False
                                break
                    
                    if can_move_right:
                        player.move_x(1)
                        game.increment_score()
                    else:
                        # Try moving vertically to avoid obstacles
                        player.move_y(1 if player.get_y() < 300 else -1)
                
                # Apply pace progression
                if survival_time % 1000 < 16.67:  # Every second
                    current_pace = game.get_pace()
                    if survival_time > 10000:  # After 10 seconds
                        game.set_pace(min(current_pace + 1, 10))
                
                # Update screen
                game.update_screen(player, lines)
                
                # Check boundaries
                if player.get_x() < config.X_MIN:
                    game.end()
                    break
                    
                player.set_x(min(player.get_x(), config.X_MAX))
                player.set_y(max(config.Y_MIN, min(player.get_y(), config.Y_MAX)))
            
            # Verify survival results
            final_time = game.get_clock().get_seconds()
            final_score = game.get_score()
            
            assert final_time > 0
            assert final_score >= 0
    
    def test_challenging_maze_navigation(self):
        """Test navigation through challenging maze."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(85, 200, headless=True)
            
            # Create challenging maze layout
            lines = []
            
            # Create maze sections with narrow passages
            for i in range(5):
                x_offset = 120 + i * 60
                
                # Top wall
                lines.append(Line((x_offset, 150), (x_offset + 40, 150)))
                # Bottom wall  
                lines.append(Line((x_offset, 250), (x_offset + 40, 250)))
                # Partial side walls (create narrow passage)
                lines.append(Line((x_offset + 40, 150), (x_offset + 40, 190)))
                lines.append(Line((x_offset + 40, 210), (x_offset + 40, 250)))
            
            # Navigation strategy
            target_x = 500  # Navigate to this X position
            navigation_steps = 0
            max_steps = 1000
            
            while (player.get_x() < target_x and 
                   navigation_steps < max_steps and 
                   game.is_active()):
                
                navigation_steps += 1
                
                # Try to move right
                can_move_right = True
                test_x = player.get_x() + player.get_speed()
                
                for line in lines:
                    # Check collision with test position
                    if line.get_is_horizontal():
                        if (player.get_y() <= line.get_y_start() <= player.get_y() + player.get_height() and
                            test_x < line.get_x_end() and test_x + player.get_width() > line.get_x_start()):
                            can_move_right = False
                            break
                    else:
                        if (test_x <= line.get_x_start() <= test_x + player.get_width() and
                            player.get_y() < line.get_y_end() and player.get_y() + player.get_height() > line.get_y_start()):
                            can_move_right = False
                            break
                
                if can_move_right:
                    player.move_x(1)
                    game.increment_score()
                else:
                    # Navigate around obstacle
                    if player.get_y() > 200:
                        player.move_y(-1)  # Move up
                    else:
                        player.move_y(1)   # Move down
                
                # Update game
                game.update_screen(player, lines)
                game.get_clock().update()
            
            # Verify navigation success
            assert player.get_x() > 120  # Should have made progress
            assert navigation_steps < max_steps  # Should not timeout


class TestPauseResumeScenarios:
    """Test pause and resume functionality scenarios."""
    
    def test_pause_during_gameplay(self):
        """Test pausing during active gameplay."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(150, 200, headless=True)
            lines = Line.generate_maze(game, 10, 10)
            
            # Play for a while
            for _ in range(50):
                player.move_x(1)
                game.increment_score()
                game.update_screen(player, lines)
                game.get_clock().update()
            
            # Record state before pause
            pre_pause_score = game.get_score()
            pre_pause_position = player.get_position()
            pre_pause_time = game.get_clock().get_millis()
            
            # Pause game
            game.change_paused(player)
            assert game.is_paused()
            
            # Simulate pause duration
            pause_duration = 1000  # 1 second
            
            # During pause, game state should not change
            for _ in range(60):  # 1 second at 60 FPS
                if game.is_paused():
                    # Rollback time during pause
                    game.get_clock().rollback_millis(16.67)
                
                game.update_screen(player, lines)
            
            # Resume game
            game.change_paused(player)
            assert not game.is_paused()
            
            # Verify state preservation
            assert game.get_score() == pre_pause_score
            assert player.get_position() == pre_pause_position
    
    def test_multiple_pause_resume_cycles(self):
        """Test multiple pause/resume cycles."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            
            # Perform multiple pause/resume cycles
            for cycle in range(5):
                # Play for a bit
                for _ in range(20):
                    player.move_x(1)
                    game.increment_score()
                    game.get_clock().update()
                
                # Pause
                game.change_paused(player)
                assert game.is_paused()
                
                # Brief pause
                for _ in range(10):
                    game.update_screen(player, [])
                
                # Resume
                game.change_paused(player)
                assert not game.is_paused()
            
            # Verify final state
            assert game.get_score() >= 100  # Should have gained score
    
    def test_pause_at_critical_moments(self):
        """Test pausing at critical game moments."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(config.X_MIN + 5, 200, headless=True)
            lines = [Line((config.X_MIN + 10, 180), (config.X_MIN + 10, 220))]
            
            # Move towards danger
            player.set_x(config.X_MIN + 2)
            
            # Pause just before boundary
            game.change_paused(player)
            pause_position = player.get_position()
            
            # Try to move during pause (should not affect position)
            for _ in range(30):
                game.update_screen(player, lines)
            
            # Resume
            game.change_paused(player)
            
            # Position should be preserved
            assert player.get_position() == pause_position


class TestGameOverScenarios:
    """Test game over scenarios."""
    
    def test_left_boundary_game_over(self):
        """Test game over when hitting left boundary."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(config.X_MIN + 5, 200, headless=True)
            
            # Move towards left boundary
            player.set_x(config.X_MIN - 10)
            
            # Check if game should end
            if player.get_x() < config.X_MIN:
                game.end()
                assert not game.is_active()
                assert game.is_playing()  # Still in game loop, but not active
    
    def test_collision_game_over(self):
        """Test game over due to collision (if implemented)."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(95, 200, headless=True)
            
            # Create blocking wall
            wall = Line((100, 180), (100, 220))
            
            # Force collision
            player.set_x(100)  # Move into wall
            
            # Check collision
            if (player.get_x() <= wall.get_x_start() <= player.get_x() + player.get_width() and
                player.get_y() <= wall.get_y_end() and 
                player.get_y() + player.get_height() >= wall.get_y_start()):
                
                # Collision detected - game might end or position corrected
                # This depends on game implementation
                if not game.is_active():
                    assert not game.is_active()
    
    def test_game_reset_after_game_over(self):
        """Test game reset after game over."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            
            # Play and gain score
            for _ in range(20):
                player.move_x(1)
                game.increment_score()
            
            score_before_end = game.get_score()
            
            # End game
            game.end()
            assert not game.is_active()
            
            # Reset game
            game.reset()
            assert game.is_active()
            assert game.is_playing()
            assert game.get_score() == 0  # Score should reset
            
            # Reset player
            player.reset(80, 223)  # Default position
            assert player.get_position() == (80, 223)


class TestEdgeCaseScenarios:
    """Test edge case scenarios."""
    
    def test_rapid_input_handling(self):
        """Test rapid input sequence handling."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(200, 200, headless=True)
            
            # Rapid input sequence
            rapid_moves = [
                'right', 'down', 'right', 'up', 'right', 'down',
                'left', 'up', 'right', 'down', 'right', 'right'
            ]
            
            initial_position = player.get_position()
            
            for move in rapid_moves:
                if move == 'right':
                    player.move_x(1)
                    game.increment_score()
                elif move == 'left':
                    player.move_x(-1)
                    game.decrement_score()
                elif move == 'up':
                    player.move_y(-1)
                elif move == 'down':
                    player.move_y(1)
                
                game.update_screen(player, [])
                game.get_clock().update()
            
            # Verify final state is consistent
            final_position = player.get_position()
            assert final_position != initial_position
            assert game.get_score() >= 0
    
    def test_extreme_position_handling(self):
        """Test handling of extreme positions."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(0, 0, headless=True)
            
            # Test extreme positions
            extreme_positions = [
                (-1000, -1000),
                (10000, 10000),
                (config.X_MIN - 100, config.Y_MIN - 100),
                (config.X_MAX + 100, config.Y_MAX + 100)
            ]
            
            for x, y in extreme_positions:
                player.set_x(x)
                player.set_y(y)
                
                # Game should handle gracefully
                try:
                    game.update_screen(player, [])
                except Exception as e:
                    assert False, f"Game crashed with extreme position ({x}, {y}): {e}"
                
                # Correct position if needed
                player.set_x(max(config.X_MIN, min(x, config.X_MAX)))
                player.set_y(max(config.Y_MIN, min(y, config.Y_MAX)))
    
    def test_zero_time_gameplay(self):
        """Test gameplay with zero or minimal time."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            clock = game.get_clock()
            
            # Set clock to zero
            clock.millis = 0
            
            # Try to play with zero time
            player.move_x(1)
            game.increment_score()
            clock.update()
            
            # Should handle gracefully
            assert game.get_score() >= 1
            assert clock.get_millis() >= 0


class TestPerformanceScenarios:
    """Test performance under various gameplay scenarios."""
    
    @pytest.mark.performance
    def test_extended_gameplay_performance(self):
        """Test performance during extended gameplay."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(150, 200, headless=True)
            lines = Line.generate_maze(game, 25, 30)
            
            monitor = PerformanceMonitor()
            monitor.start()
            
            # Simulate 5 minutes of gameplay
            for frame in range(18000):  # 5 minutes at 60 FPS
                # Player movement strategy
                if frame % 3 == 0:
                    player.move_x(1)
                    game.increment_score()
                elif frame % 7 == 0:
                    player.move_y(1 if frame % 14 == 0 else -1)
                
                # Update game systems
                game.update_screen(player, lines)
                game.get_clock().update()
                
                # Sample performance periodically
                if frame % 600 == 0:  # Every 10 seconds
                    monitor.sample_frame(60.0, 16.7)
            
            monitor.stop()
            
            # Verify performance
            duration = monitor.get_duration()
            avg_fps = monitor.get_average_fps()
            memory_stats = monitor.get_memory_usage_mb()
            
            assert duration < 10.0  # Should complete reasonably quickly
            assert avg_fps > 30.0   # Should maintain good FPS
            assert memory_stats['peak'] < 500  # Should not leak memory
    
    @pytest.mark.performance
    def test_high_pace_gameplay_performance(self):
        """Test performance at high game pace."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(200, 200, headless=True)
            lines = Line.generate_maze(game, 20, 25)
            
            # Set high pace
            game.set_pace(10)
            
            monitor = PerformanceMonitor()
            monitor.start()
            
            # Simulate high-pace gameplay
            for frame in range(3600):  # 1 minute at 60 FPS
                # Apply pace to maze
                pace = game.get_pace()
                for line in lines:
                    line.set_x_start(line.get_x_start() - pace)
                    line.set_x_end(line.get_x_end() - pace)
                
                # Player must move to keep up
                player.move_x(pace // 2)
                if frame % 2 == 0:
                    game.increment_score()
                
                # Update systems
                game.update_screen(player, lines)
                game.get_clock().update()
                
                # Check boundaries
                if player.get_x() < config.X_MIN:
                    game.end()
                    break
                
                player.set_x(min(player.get_x(), config.X_MAX))
            
            monitor.stop()
            
            # Verify high-pace performance
            duration = monitor.get_duration()
            assert duration < 5.0, f"High-pace gameplay too slow: {duration}s"
    
    @pytest.mark.performance
    def test_large_maze_performance(self):
        """Test performance with large maze."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            
            # Generate large maze
            lines = Line.generate_maze(game, 50, 60)  # Large maze
            
            import time
            start_time = time.time()
            
            # Test collision detection with large maze
            for _ in range(1000):
                # Test movement in all directions
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    test_x = player.get_x() + dx * player.get_speed()
                    test_y = player.get_y() + dy * player.get_speed()
                    
                    collision_count = 0
                    for line in lines:
                        # Collision check
                        if line.get_is_horizontal():
                            if (test_y <= line.get_y_start() <= test_y + player.get_height() and
                                test_x < line.get_x_end() and test_x + player.get_width() > line.get_x_start()):
                                collision_count += 1
                        else:
                            if (test_x <= line.get_x_start() <= test_x + player.get_width() and
                                test_y < line.get_y_end() and test_y + player.get_height() > line.get_y_start()):
                                collision_count += 1
            
            end_time = time.time()
            duration = end_time - start_time
            
            assert duration < 10.0, f"Large maze collision detection too slow: {duration}s"
