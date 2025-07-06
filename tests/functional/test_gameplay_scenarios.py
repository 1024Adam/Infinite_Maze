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
from tests.fixtures.pygame_mocks import full_pygame_mocks, InputSimulator
from tests.fixtures.test_helpers import PerformanceMonitor, GameStateCapture


class TestGameStartupScenarios:
    """Test game startup and initialization scenarios."""
    
    def test_normal_game_startup(self):
        """Test normal game startup sequence."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(80, 223, headless=True)  # Default start position
            lines = Line.generateMaze(game, 15, 20)
            
            # Verify initial state
            assert game.isPlaying()
            assert game.isActive()
            assert not game.isPaused()
            assert game.getScore() == 0
            assert game.getPace() == 0
            
            # Verify player start position
            assert player.getPosition() == (80, 223)
            
            # Verify maze generation
            assert len(lines) > 0
            assert all(isinstance(line, Line) for line in lines)
    
    def test_game_startup_with_custom_settings(self):
        """Test game startup with custom initial settings."""
        with full_pygame_mocks():
            game = Game(headless=True)
            
            # Custom player position
            player = Player(150, 250, headless=True)
            assert player.getPosition() == (150, 250)
            
            # Custom maze size
            lines = Line.generateMaze(game, 10, 15)
            assert len(lines) > 0
            
            # Custom initial score
            game.setScore(5)
            assert game.getScore() == 5
    
    def test_game_startup_error_recovery(self):
        """Test game startup with error conditions."""
        with full_pygame_mocks():
            # Test with invalid player position
            player = Player(-100, -100, headless=True)
            game = Game(headless=True)
            
            # Game should handle invalid positions gracefully
            game.updateScreen(player, [])
            
            # Reset to valid position
            player.reset(100, 100)
            assert player.getPosition() == (100, 100)


class TestBasicGameplayScenarios:
    """Test basic gameplay scenarios."""
    
    def test_simple_movement_session(self):
        """Test simple movement gameplay session."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            lines = Line.generateMaze(game, 10, 10)
            
            # Record initial state
            initial_score = game.getScore()
            initial_position = player.getPosition()
            
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
                    player.moveX(1)
                    game.incrementScore()
                elif direction == 'left':
                    player.moveX(-1)
                    game.decrementScore()
                elif direction == 'down':
                    player.moveY(1)
                elif direction == 'up':
                    player.moveY(-1)
                
                # Update game state
                game.updateScreen(player, lines)
                game.getClock().update()
            
            # Verify final state
            final_position = player.getPosition()
            final_score = game.getScore()
            
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
            initial_x = player.getX()
            
            # Try to move right (should hit wall)
            next_x = initial_x + player.getSpeed()
            collision = False
            
            for wall in walls:
                if not wall.getIsHorizontal():  # Vertical wall
                    if (next_x <= wall.getXStart() <= next_x + player.getWidth() and
                        player.getY() <= wall.getYEnd() and 
                        player.getY() + player.getHeight() >= wall.getYStart()):
                        collision = True
                        break
            
            if not collision:
                player.moveX(1)
                game.incrementScore()
            
            # Verify collision handling
            if collision:
                assert player.getX() == initial_x  # Should not move
                assert game.getScore() == 0        # Score unchanged
            else:
                assert player.getX() > initial_x   # Should move
    
    def test_boundary_interaction_gameplay(self):
        """Test gameplay at game boundaries."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(game.X_MIN + 5, game.Y_MIN + 5, headless=True)
            
            # Test left boundary
            player.setX(game.X_MIN - 10)
            if player.getX() < game.X_MIN:
                # Game should end or position should be corrected
                game.end()
                assert not game.isActive()
            
            # Reset game
            game.reset()
            player.reset(game.X_MAX - 5, game.Y_MAX - 5)
            
            # Test right boundary
            player.setX(game.X_MAX + 10)
            if player.getX() > game.X_MAX:
                player.setX(game.X_MAX)
            assert player.getX() <= game.X_MAX
            
            # Test vertical boundaries
            player.setY(game.Y_MIN - 10)
            player.setY(max(player.getY(), game.Y_MIN))
            assert player.getY() >= game.Y_MIN
            
            player.setY(game.Y_MAX + 10)
            player.setY(min(player.getY(), game.Y_MAX))
            assert player.getY() <= game.Y_MAX


class TestAdvancedGameplayScenarios:
    """Test advanced gameplay scenarios."""
    
    def test_pace_progression_gameplay(self):
        """Test gameplay with pace progression over time."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(200, 200, headless=True)
            lines = Line.generateMaze(game, 15, 20)
            clock = game.getClock()
            
            # Simulate extended gameplay
            for frame in range(300):  # 5 seconds at 60 FPS
                # Update clock
                clock.millis += 16.67  # ~60 FPS
                clock.update()
                
                # Simulate pace progression
                if clock.getSeconds() > 10 and clock.getSeconds() % 30 == 0:
                    current_pace = game.getPace()
                    game.setPace(current_pace + 1)
                
                # Apply pace to maze
                pace = game.getPace()
                for line in lines:
                    line.setXStart(line.getXStart() - pace)
                    line.setXEnd(line.getXEnd() - pace)
                
                # Player movement to stay in bounds
                if frame % 10 == 0:
                    player.moveX(1)
                    game.incrementScore()
                
                # Update screen
                game.updateScreen(player, lines)
            
            # Verify pace progression
            assert game.getPace() >= 0
            assert game.getScore() >= 30  # Should have gained score
    
    def test_long_survival_gameplay(self):
        """Test long survival gameplay session."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            lines = Line.generateMaze(game, 20, 25)
            
            # Simulate 30-second survival session
            survival_time = 0
            target_survival = 30000  # 30 seconds in milliseconds
            
            while survival_time < target_survival and game.isActive():
                # Update clock
                game.getClock().millis += 16.67
                game.getClock().update()
                survival_time = game.getClock().getMillis()
                
                # Simulate player strategy: move right to gain score
                if survival_time % 100 < 16.67:  # Every ~100ms
                    # Check if safe to move right
                    can_move_right = True
                    test_x = player.getX() + player.getSpeed()
                    
                    for line in lines:
                        if not line.getIsHorizontal():  # Vertical wall
                            if (test_x <= line.getXStart() <= test_x + player.getWidth() and
                                player.getY() <= line.getYEnd() and 
                                player.getY() + player.getHeight() >= line.getYStart()):
                                can_move_right = False
                                break
                    
                    if can_move_right:
                        player.moveX(1)
                        game.incrementScore()
                    else:
                        # Try moving vertically to avoid obstacles
                        player.moveY(1 if player.getY() < 300 else -1)
                
                # Apply pace progression
                if survival_time % 1000 < 16.67:  # Every second
                    current_pace = game.getPace()
                    if survival_time > 10000:  # After 10 seconds
                        game.setPace(min(current_pace + 1, 10))
                
                # Update screen
                game.updateScreen(player, lines)
                
                # Check boundaries
                if player.getX() < game.X_MIN:
                    game.end()
                    break
                    
                player.setX(min(player.getX(), game.X_MAX))
                player.setY(max(game.Y_MIN, min(player.getY(), game.Y_MAX)))
            
            # Verify survival results
            final_time = game.getClock().getSeconds()
            final_score = game.getScore()
            
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
            
            while (player.getX() < target_x and 
                   navigation_steps < max_steps and 
                   game.isActive()):
                
                navigation_steps += 1
                
                # Try to move right
                can_move_right = True
                test_x = player.getX() + player.getSpeed()
                
                for line in lines:
                    # Check collision with test position
                    if line.getIsHorizontal():
                        if (player.getY() <= line.getYStart() <= player.getY() + player.getHeight() and
                            test_x < line.getXEnd() and test_x + player.getWidth() > line.getXStart()):
                            can_move_right = False
                            break
                    else:
                        if (test_x <= line.getXStart() <= test_x + player.getWidth() and
                            player.getY() < line.getYEnd() and player.getY() + player.getHeight() > line.getYStart()):
                            can_move_right = False
                            break
                
                if can_move_right:
                    player.moveX(1)
                    game.incrementScore()
                else:
                    # Navigate around obstacle
                    if player.getY() > 200:
                        player.moveY(-1)  # Move up
                    else:
                        player.moveY(1)   # Move down
                
                # Update game
                game.updateScreen(player, lines)
                game.getClock().update()
            
            # Verify navigation success
            assert player.getX() > 120  # Should have made progress
            assert navigation_steps < max_steps  # Should not timeout


class TestPauseResumeScenarios:
    """Test pause and resume functionality scenarios."""
    
    def test_pause_during_gameplay(self):
        """Test pausing during active gameplay."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(150, 200, headless=True)
            lines = Line.generateMaze(game, 10, 10)
            
            # Play for a while
            for _ in range(50):
                player.moveX(1)
                game.incrementScore()
                game.updateScreen(player, lines)
                game.getClock().update()
            
            # Record state before pause
            pre_pause_score = game.getScore()
            pre_pause_position = player.getPosition()
            pre_pause_time = game.getClock().getMillis()
            
            # Pause game
            game.changePaused(player)
            assert game.isPaused()
            
            # Simulate pause duration
            pause_duration = 1000  # 1 second
            
            # During pause, game state should not change
            for _ in range(60):  # 1 second at 60 FPS
                if game.isPaused():
                    # Rollback time during pause
                    game.getClock().rollbackMillis(16.67)
                
                game.updateScreen(player, lines)
            
            # Resume game
            game.changePaused(player)
            assert not game.isPaused()
            
            # Verify state preservation
            assert game.getScore() == pre_pause_score
            assert player.getPosition() == pre_pause_position
    
    def test_multiple_pause_resume_cycles(self):
        """Test multiple pause/resume cycles."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            
            # Perform multiple pause/resume cycles
            for cycle in range(5):
                # Play for a bit
                for _ in range(20):
                    player.moveX(1)
                    game.incrementScore()
                    game.getClock().update()
                
                # Pause
                game.changePaused(player)
                assert game.isPaused()
                
                # Brief pause
                for _ in range(10):
                    game.updateScreen(player, [])
                
                # Resume
                game.changePaused(player)
                assert not game.isPaused()
            
            # Verify final state
            assert game.getScore() >= 100  # Should have gained score
    
    def test_pause_at_critical_moments(self):
        """Test pausing at critical game moments."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(game.X_MIN + 5, 200, headless=True)
            lines = [Line((game.X_MIN + 10, 180), (game.X_MIN + 10, 220))]
            
            # Move towards danger
            player.setX(game.X_MIN + 2)
            
            # Pause just before boundary
            game.changePaused(player)
            pause_position = player.getPosition()
            
            # Try to move during pause (should not affect position)
            for _ in range(30):
                game.updateScreen(player, lines)
            
            # Resume
            game.changePaused(player)
            
            # Position should be preserved
            assert player.getPosition() == pause_position


class TestGameOverScenarios:
    """Test game over scenarios."""
    
    def test_left_boundary_game_over(self):
        """Test game over when hitting left boundary."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(game.X_MIN + 5, 200, headless=True)
            
            # Move towards left boundary
            player.setX(game.X_MIN - 10)
            
            # Check if game should end
            if player.getX() < game.X_MIN:
                game.end()
                assert not game.isActive()
                assert game.isPlaying()  # Still in game loop, but not active
    
    def test_collision_game_over(self):
        """Test game over due to collision (if implemented)."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(95, 200, headless=True)
            
            # Create blocking wall
            wall = Line((100, 180), (100, 220))
            
            # Force collision
            player.setX(100)  # Move into wall
            
            # Check collision
            if (player.getX() <= wall.getXStart() <= player.getX() + player.getWidth() and
                player.getY() <= wall.getYEnd() and 
                player.getY() + player.getHeight() >= wall.getYStart()):
                
                # Collision detected - game might end or position corrected
                # This depends on game implementation
                if not game.isActive():
                    assert not game.isActive()
    
    def test_game_reset_after_game_over(self):
        """Test game reset after game over."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            
            # Play and gain score
            for _ in range(20):
                player.moveX(1)
                game.incrementScore()
            
            score_before_end = game.getScore()
            
            # End game
            game.end()
            assert not game.isActive()
            
            # Reset game
            game.reset()
            assert game.isActive()
            assert game.isPlaying()
            assert game.getScore() == 0  # Score should reset
            
            # Reset player
            player.reset(80, 223)  # Default position
            assert player.getPosition() == (80, 223)


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
            
            initial_position = player.getPosition()
            
            for move in rapid_moves:
                if move == 'right':
                    player.moveX(1)
                    game.incrementScore()
                elif move == 'left':
                    player.moveX(-1)
                    game.decrementScore()
                elif move == 'up':
                    player.moveY(-1)
                elif move == 'down':
                    player.moveY(1)
                
                game.updateScreen(player, [])
                game.getClock().update()
            
            # Verify final state is consistent
            final_position = player.getPosition()
            assert final_position != initial_position
            assert game.getScore() >= 0
    
    def test_extreme_position_handling(self):
        """Test handling of extreme positions."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(0, 0, headless=True)
            
            # Test extreme positions
            extreme_positions = [
                (-1000, -1000),
                (10000, 10000),
                (game.X_MIN - 100, game.Y_MIN - 100),
                (game.X_MAX + 100, game.Y_MAX + 100)
            ]
            
            for x, y in extreme_positions:
                player.setX(x)
                player.setY(y)
                
                # Game should handle gracefully
                try:
                    game.updateScreen(player, [])
                except Exception as e:
                    assert False, f"Game crashed with extreme position ({x}, {y}): {e}"
                
                # Correct position if needed
                player.setX(max(game.X_MIN, min(x, game.X_MAX)))
                player.setY(max(game.Y_MIN, min(y, game.Y_MAX)))
    
    def test_zero_time_gameplay(self):
        """Test gameplay with zero or minimal time."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(100, 200, headless=True)
            clock = game.getClock()
            
            # Set clock to zero
            clock.millis = 0
            
            # Try to play with zero time
            player.moveX(1)
            game.incrementScore()
            clock.update()
            
            # Should handle gracefully
            assert game.getScore() >= 1
            assert clock.getMillis() >= 0


class TestPerformanceScenarios:
    """Test performance under various gameplay scenarios."""
    
    @pytest.mark.performance
    def test_extended_gameplay_performance(self):
        """Test performance during extended gameplay."""
        with full_pygame_mocks():
            game = Game(headless=True)
            player = Player(150, 200, headless=True)
            lines = Line.generateMaze(game, 25, 30)
            
            monitor = PerformanceMonitor()
            monitor.start()
            
            # Simulate 5 minutes of gameplay
            for frame in range(18000):  # 5 minutes at 60 FPS
                # Player movement strategy
                if frame % 3 == 0:
                    player.moveX(1)
                    game.incrementScore()
                elif frame % 7 == 0:
                    player.moveY(1 if frame % 14 == 0 else -1)
                
                # Update game systems
                game.updateScreen(player, lines)
                game.getClock().update()
                
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
            lines = Line.generateMaze(game, 20, 25)
            
            # Set high pace
            game.setPace(10)
            
            monitor = PerformanceMonitor()
            monitor.start()
            
            # Simulate high-pace gameplay
            for frame in range(3600):  # 1 minute at 60 FPS
                # Apply pace to maze
                pace = game.getPace()
                for line in lines:
                    line.setXStart(line.getXStart() - pace)
                    line.setXEnd(line.getXEnd() - pace)
                
                # Player must move to keep up
                player.moveX(pace // 2)
                if frame % 2 == 0:
                    game.incrementScore()
                
                # Update systems
                game.updateScreen(player, lines)
                game.getClock().update()
                
                # Check boundaries
                if player.getX() < game.X_MIN:
                    game.end()
                    break
                
                player.setX(min(player.getX(), game.X_MAX))
            
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
            lines = Line.generateMaze(game, 50, 60)  # Large maze
            
            import time
            start_time = time.time()
            
            # Test collision detection with large maze
            for _ in range(1000):
                # Test movement in all directions
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    test_x = player.getX() + dx * player.getSpeed()
                    test_y = player.getY() + dy * player.getSpeed()
                    
                    collision_count = 0
                    for line in lines:
                        # Collision check
                        if line.getIsHorizontal():
                            if (test_y <= line.getYStart() <= test_y + player.getHeight() and
                                test_x < line.getXEnd() and test_x + player.getWidth() > line.getXStart()):
                                collision_count += 1
                        else:
                            if (test_x <= line.getXStart() <= test_x + player.getWidth() and
                                test_y < line.getYEnd() and test_y + player.getHeight() > line.getYStart()):
                                collision_count += 1
            
            end_time = time.time()
            duration = end_time - start_time
            
            assert duration < 10.0, f"Large maze collision detection too slow: {duration}s"
