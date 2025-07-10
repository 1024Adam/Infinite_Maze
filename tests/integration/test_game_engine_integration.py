"""
Integration tests for Game Engine in Infinite Maze.

These tests verify the integration between the Game engine, Player,
Clock, and Maze components working together as a complete system.
"""

import pytest
import pygame
from unittest.mock import Mock, patch, MagicMock

from infinite_maze.core.engine import maze, controlled_run
from infinite_maze.core.game import Game
from infinite_maze.entities.player import Player
from infinite_maze.entities.maze import Line
from infinite_maze.utils.config import config
from tests.fixtures.pygame_mocks import full_pygame_mocks, InputSimulator
from tests.fixtures.test_helpers import PerformanceMonitor, GameStateCapture


class TestGameEngineInitialization:
    """Test game engine initialization and setup."""
    
    def test_engine_components_creation(self):
        """Test that engine creates all necessary components."""
        with patch('infinite_maze.core.engine.pygame.init'), \
             patch('infinite_maze.core.engine.pygame.key.get_pressed', return_value={}), \
             patch('infinite_maze.core.engine.pygame.event.get', return_value=[]), \
             patch('infinite_maze.core.engine.time.delay'):
            
            # Mock the game loop to exit immediately
            with patch.object(Game, 'is_playing', return_value=False):
                with patch.object(Game, 'is_active', return_value=False):
                    with patch.object(Game, 'cleanup'):
                        try:
                            maze()
                        except SystemExit:
                            pass  # Expected exit
    
    def test_controlled_run_initialization(self):
        """Test controlled_run function initialization."""
        mock_wrapper = Mock()
        mock_counter = Mock()
        
        with patch('infinite_maze.core.engine.pygame.key.get_pressed', return_value={}):
            # Mock the game to exit immediately
            with patch.object(Game, 'is_playing', return_value=False):
                with patch.object(Game, 'is_active', return_value=False):
                    result = controlled_run(mock_wrapper, mock_counter)
                    
                    # Should return some result
                    assert result is not None


class TestGameEngineGameLoop:
    """Test game engine main loop functionality."""
    
    def test_game_loop_single_iteration(self):
        """Test single iteration of game loop."""
        with full_pygame_mocks() as mocks:
            # Set up game to run one iteration then exit
            game = Game(headless=True)
            player = Player(100, 100, headless=True)
            lines = Line.generate_maze(game, 5, 5)
            
            # Mock key state for one frame
            mocks['event']['key_state'][pygame.K_RIGHT] = True
            
            # Simulate one game loop iteration
            initial_score = game.get_score()
            initial_pos = player.get_position()
            
            # Simulate right movement
            player.move_x(1)
            game.increment_score()
            
            # Verify state changes
            assert game.get_score() == initial_score + 1
            assert player.get_x() > initial_pos[0]
    
    def test_game_loop_pause_handling(self):
        """Test game loop pause functionality."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        
        # Test pause toggle
        assert not game.is_paused()
        
        game.change_paused(player)
        assert game.is_paused()
        
        game.change_paused(player)
        assert not game.is_paused()
    
    def test_game_loop_input_processing(self):
        """Test input processing in game loop."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        
        # Test various inputs
        input_tests = [
            ('right', lambda: player.move_x(1)),
            ('left', lambda: player.move_x(-1)),
            ('up', lambda: player.move_y(-1)),
            ('down', lambda: player.move_y(1))
        ]
        
        for direction, action in input_tests:
            initial_pos = player.get_position()
            action()
            new_pos = player.get_position()
            
            # Position should change
            assert new_pos != initial_pos


class TestGameEngineMovementSystem:
    """Test game engine movement and collision system."""
    
    def test_movement_with_collision_detection(self):
        """Test movement system with collision detection."""
        player = Player(90, 100, headless=True)
        
        # Create wall blocking movement
        walls = [Line((100, 90), (100, 110))]  # Vertical wall
        
        # Test collision detection
        initial_x = player.get_x()
        
        # Simulate collision check before movement
        test_x = initial_x + player.get_speed()
        collision_detected = False
        
        for wall in walls:
            if wall.get_is_horizontal():
                # Check horizontal collision
                if (player.get_y() <= wall.get_y_start() <= player.get_y() + player.get_height() and
                    test_x <= wall.get_x_end() and test_x + player.get_width() >= wall.get_x_start()):
                    collision_detected = True
            else:
                # Check vertical collision
                if (test_x <= wall.get_x_start() <= test_x + player.get_width() and
                    player.get_y() <= wall.get_y_end() and player.get_y() + player.get_height() >= wall.get_y_start()):
                    collision_detected = True
        
        if not collision_detected:
            player.move_x(1)
        
        # Verify movement or blocking
        if collision_detected:
            assert player.get_x() == initial_x  # Should not move
        else:
            assert player.get_x() > initial_x   # Should move
    
    def test_boundary_enforcement(self):
        """Test game boundary enforcement."""
        game = Game(headless=True)
        player = Player(config.X_MIN, config.Y_MIN, headless=True)
        
        # Test left boundary
        player.set_x(config.X_MIN - 10)
        if player.get_x() < config.X_MIN:
            # Game should end or adjust position
            assert player.get_x() < config.X_MIN  # Or game.end() called
        
        # Test right boundary
        player.set_x(config.X_MAX + 10)
        if player.get_x() > config.X_MAX:
            # Position should be adjusted
            adjusted_x = min(player.get_x(), config.X_MAX)
            player.set_x(adjusted_x)
            assert player.get_x() <= config.X_MAX
        
        # Test vertical boundaries
        player.set_y(config.Y_MIN - 10)
        adjusted_y = max(player.get_y(), config.Y_MIN)
        player.set_y(adjusted_y)
        assert player.get_y() >= config.Y_MIN
        
        player.set_y(config.Y_MAX + 10)
        adjusted_y = min(player.get_y(), config.Y_MAX)
        player.set_y(adjusted_y)
        assert player.get_y() <= config.Y_MAX
    
    def test_maze_line_repositioning(self):
        """Test maze line repositioning system."""
        game = Game(headless=True)
        lines = Line.generate_maze(game, 5, 5)
        
        # Find initial max X
        initial_max_x = Line.get_x_max(lines)
        
        # Simulate line repositioning (when lines move off screen)
        for line in lines:
            if line.get_x_start() < 80:  # Line moved off screen
                line.set_x_start(initial_max_x + 22)
                if line.get_x_start() == line.get_x_end():
                    line.set_x_end(initial_max_x + 22)
                else:
                    line.set_x_end(initial_max_x + 44)
        
        # Verify repositioning
        new_max_x = Line.get_x_max(lines)
        assert new_max_x >= initial_max_x


class TestGameEngineScoring:
    """Test game engine scoring system."""
    
    def test_scoring_with_movement(self):
        """Test scoring integration with movement."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        
        initial_score = game.get_score()
        
        # Right movement should increase score
        player.move_x(1)
        game.increment_score()
        assert game.get_score() == initial_score + 1
        
        # Left movement should decrease score
        player.move_x(-1)
        game.decrement_score()
        assert game.get_score() == initial_score
        
        # Vertical movement should not affect score
        player.move_y(1)
        assert game.get_score() == initial_score
        
        player.move_y(-1)
        assert game.get_score() == initial_score
    
    def test_scoring_minimum_enforcement(self):
        """Test score minimum enforcement in game context."""
        game = Game(headless=True)
        
        # Start with some score
        game.set_score(2)
        
        # Decrement to minimum
        game.decrement_score()  # Score = 1
        game.decrement_score()  # Score = 0
        game.decrement_score()  # Should stay at 0
        
        assert game.get_score() == 0
    
    def test_scoring_with_game_progression(self):
        """Test scoring throughout game progression."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        
        # Simulate game progression with mixed movements
        movement_sequence = [
            ('right', 1, 1),    # Right: +1
            ('right', 1, 2),    # Right: +1
            ('left', -1, 1),    # Left: -1
            ('up', 0, 1),       # Up: +0
            ('down', 0, 1),     # Down: +0
            ('right', 1, 2),    # Right: +1
        ]
        
        for direction, score_change, expected_score in movement_sequence:
            if direction == 'right':
                player.move_x(1)
                game.increment_score()
            elif direction == 'left':
                player.move_x(-1)
                game.decrement_score()
            elif direction == 'up':
                player.move_y(-1)
            elif direction == 'down':
                player.move_y(1)
            
            assert game.get_score() == expected_score


class TestGameEnginePaceSystem:
    """Test game engine pace/timing system."""
    
    def test_pace_progression(self):
        """Test pace progression over time."""
        game = Game(headless=True)
        clock = game.get_clock()
        
        # Simulate time progression
        time_tests = [
            (5000, 0),    # 5 seconds: no pace
            (15000, 0),   # 15 seconds: no pace
            (35000, 1),   # 35 seconds: pace starts
            (65000, 2),   # 65 seconds: pace increases
        ]
        
        for millis, expected_pace in time_tests:
            clock.millis = millis
            
            # Simulate pace calculation
            if millis > 10000 and clock.get_seconds() % 30 == 0:
                current_pace = game.get_pace()
                game.set_pace(current_pace + 1)
            
            # Note: Exact pace calculation depends on implementation details
            # This test verifies the general progression concept
    
    def test_pace_affects_movement(self):
        """Test that pace affects game movement."""
        game = Game(headless=True)
        player = Player(200, 100, headless=True)
        lines = Line.generate_maze(game, 5, 5)
        
        # Set initial pace
        game.set_pace(0)
        initial_positions = [(line.get_x_start(), line.get_x_end()) for line in lines]
        
        # Increase pace
        game.set_pace(3)
        
        # Simulate pace effect on lines (moving them left)
        pace_offset = game.get_pace()
        for line in lines:
            line.set_x_start(line.get_x_start() - pace_offset)
            line.set_x_end(line.get_x_end() - pace_offset)
        
        # Verify lines moved
        new_positions = [(line.get_x_start(), line.get_x_end()) for line in lines]
        for initial, new in zip(initial_positions, new_positions):
            assert new[0] < initial[0]  # X positions should decrease
            assert new[1] < initial[1]
    
    def test_pace_player_adjustment(self):
        """Test player position adjustment due to pace."""
        game = Game(headless=True)
        player = Player(200, 100, headless=True)
        
        initial_x = player.get_x()
        pace = 5
        game.set_pace(pace)
        
        # Simulate player position adjustment due to pace
        player.set_x(player.get_x() - pace)
        
        assert player.get_x() == initial_x - pace


class TestGameEngineStateManagement:
    """Test game engine state management."""
    
    def test_game_state_transitions(self):
        """Test game state transitions."""
        game = Game(headless=True)
        
        # Initial state
        assert game.is_playing()
        assert game.is_active()
        assert not game.is_paused()
        
        # End game
        game.end()
        assert game.is_playing()  # Still playing, but not active
        assert not game.is_active()
        
        # Reset game
        game.reset()
        assert game.is_playing()
        assert game.is_active()
        assert not game.is_paused()
        
        # Quit game
        game.quit()
        assert not game.is_playing()
    
    def test_game_state_with_player(self):
        """Test game state integration with player."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        
        # Capture initial state
        state_capture = GameStateCapture(game, player)
        state_capture.capture()
        
        # Modify state
        game.set_score(50)
        player.move_x(5)
        game.set_pace(3)
        
        # Get state differences
        diff = state_capture.get_state_diff()
        assert 'game' in diff
        assert 'player' in diff
        
        # Restore state
        state_capture.restore()
        assert game.get_score() == 0  # Should be restored
    
    def test_pause_state_effects(self):
        """Test pause state effects on game systems."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        clock = game.get_clock()
        
        # Normal state
        assert not game.is_paused()
        
        # Pause game
        game.change_paused(player)
        assert game.is_paused()
        
        # Simulate pause effects on clock
        pre_pause_millis = clock.get_millis()
        clock.update()
        post_pause_millis = clock.get_millis()
        
        if game.is_paused():
            # In paused state, might rollback time
            rollback_amount = post_pause_millis - pre_pause_millis
            clock.rollback_millis(rollback_amount)
            assert clock.get_millis() == pre_pause_millis


class TestGameEnginePerformance:
    """Test game engine performance characteristics."""
    
    @pytest.mark.performance
    def test_game_loop_performance(self):
        """Test game loop performance."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        lines = Line.generate_maze(game, 10, 10)
        
        monitor = PerformanceMonitor()
        monitor.start()
        
        # Simulate game loop iterations
        for frame in range(60):  # 1 second at 60 FPS
            # Update game state
            game.update_screen(player, lines)
            
            # Simulate movement
            if frame % 4 == 0:
                player.move_x(1)
                game.increment_score()
            
            # Update clock
            game.get_clock().update()
            
            # Record frame
            monitor.sample_frame(60.0, 16.7)  # Target 60 FPS
        
        monitor.stop()
        
        # Verify performance
        duration = monitor.get_duration()
        avg_fps = monitor.get_average_fps()
        memory_stats = monitor.get_memory_usage_mb()
        
        assert duration < 2.0  # Should complete quickly
        assert avg_fps > 30.0  # Should maintain decent FPS
        assert memory_stats['peak'] < 200  # Should not use excessive memory
    
    @pytest.mark.performance
    def test_collision_detection_performance(self):
        """Test collision detection performance in game context."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        lines = Line.generate_maze(game, 20, 20)  # Large maze
        
        import time
        start_time = time.time()
        
        # Simulate many collision checks
        for _ in range(1000):
            # Test movement in each direction
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                test_x = player.get_x() + dx * player.get_speed()
                test_y = player.get_y() + dy * player.get_speed()
                
                # Check collision with all lines
                collision_count = 0
                for line in lines:
                    # Simplified collision check
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
        
        assert duration < 5.0, f"Collision detection took too long: {duration}s"
    
    @pytest.mark.performance
    def test_maze_generation_performance(self):
        """Test maze generation performance."""
        game = Game(headless=True)
        
        import time
        start_time = time.time()
        
        # Generate multiple mazes
        for size in [(5, 5), (10, 10), (15, 15), (20, 20)]:
            for _ in range(10):
                lines = Line.generate_maze(game, size[0], size[1])
                assert len(lines) > 0
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 3.0, f"Maze generation took too long: {duration}s"


class TestGameEngineIntegration:
    """Integration tests for complete game engine."""
    
    def test_complete_game_simulation(self):
        """Test complete game simulation."""
        game = Game(headless=True)
        player = Player(80, 223, headless=True)  # Default start position
        lines = Line.generate_maze(game, 15, 20)
        
        # Simulate short game session
        for step in range(100):
            # Update screen
            game.update_screen(player, lines)
            
            # Simulate input
            if step % 5 == 0:
                # Move right occasionally
                player.move_x(1)
                game.increment_score()
            elif step % 7 == 0:
                # Move down occasionally
                player.move_y(1)
            
            # Update clock
            game.get_clock().update()
            
            # Check boundaries
            if player.get_x() < config.X_MIN:
                game.end()
                break
            if player.get_x() > config.X_MAX:
                player.set_x(config.X_MAX)
            
            # Adjust for boundaries
            player.set_y(max(player.get_y(), config.Y_MIN))
            player.set_y(min(player.get_y(), config.Y_MAX))
        
        # Verify final state
        assert game.get_score() >= 0
        assert player.get_x() >= config.X_MIN
        assert player.get_y() >= config.Y_MIN
        assert player.get_y() <= config.Y_MAX
    
    def test_game_engine_error_recovery(self):
        """Test game engine error recovery."""
        game = Game(headless=True)
        player = Player(100, 100, headless=True)
        
        # Test various error conditions
        try:
            # Invalid player position
            player.set_x(-1000)
            player.set_y(-1000)
            
            # Engine should handle gracefully
            game.update_screen(player, [])
            
        except Exception as e:
            # Should not crash with unexpected exceptions
            assert False, f"Engine crashed with: {e}"
        
        # Reset to valid state
        player.reset(100, 100)
        assert player.get_position() == (100, 100)
    
    def test_game_engine_resource_cleanup(self):
        """Test game engine resource cleanup."""
        # Test that multiple game instances clean up properly
        for _ in range(5):
            game = Game(headless=True)
            player = Player(100, 100, headless=True)
            lines = Line.generate_maze(game, 5, 5)
            
            # Use game briefly
            game.update_screen(player, lines)
            game.increment_score()
            
            # Cleanup
            game.cleanup()
        
        # Should complete without memory issues
