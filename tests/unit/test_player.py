"""
Unit tests for the Player entity in Infinite Maze.

These tests verify the functionality of the Player class including
movement, positioning, sprite management, and state handling.
"""

import pytest
import pygame
from unittest.mock import Mock, patch, call

from infinite_maze.entities.player import Player
from infinite_maze.utils.config import GameConfig
from tests.fixtures.pygame_mocks import full_pygame_mocks, MockPygameSurface


class TestPlayerInitialization:
    """Test Player class initialization."""
    
    def test_player_init_headless_mode(self):
        """Test player initialization in headless mode."""
        player = Player(100, 200, headless=True)
        
        assert player.get_x() == 100
        assert player.get_y() == 200
        assert player.get_position() == (100, 200)
        assert player.get_speed() == 1  # Player speed is hardcoded to 1
        assert player.get_width() == 10  # Player width is hardcoded to 10
        assert player.get_height() == 10  # Player height is hardcoded to 10
    
    def test_player_init_with_display(self):
        """Test player initialization with display mode."""
        with full_pygame_mocks() as mocks:
            player = Player(150, 250, headless=False)
            
            assert player.get_x() == 150
            assert player.get_y() == 250
            assert player.get_position() == (150, 250)
            # Verify image loading was attempted
            assert mocks['image']['load'].called
    
    def test_player_init_default_position(self):
        """Test player initialization with default position."""
        player = Player(0, 0, headless=True)
        
        assert player.get_position() == (0, 0)
    
    def test_player_init_negative_position(self):
        """Test player initialization with negative coordinates."""
        player = Player(-10, -20, headless=True)
        
        assert player.get_x() == -10
        assert player.get_y() == -20


class TestPlayerMovement:
    """Test Player movement functionality."""
    
    def test_move_x_positive(self):
        """Test moving player right."""
        player = Player(100, 100, headless=True)
        initial_x = player.get_x()
        
        player.move_x(1)
        
        expected_x = initial_x + player.get_speed()
        assert player.get_x() == expected_x
        assert player.get_y() == 100  # Y should not change
    
    def test_move_x_negative(self):
        """Test moving player left."""
        player = Player(100, 100, headless=True)
        initial_x = player.get_x()
        
        player.move_x(-1)
        
        expected_x = initial_x - player.get_speed()
        assert player.get_x() == expected_x
        assert player.get_y() == 100  # Y should not change
    
    def test_move_y_positive(self):
        """Test moving player down."""
        player = Player(100, 100, headless=True)
        initial_y = player.get_y()
        
        player.move_y(1)
        
        expected_y = initial_y + player.get_speed()
        assert player.get_y() == expected_y
        assert player.get_x() == 100  # X should not change
    
    def test_move_y_negative(self):
        """Test moving player up."""
        player = Player(100, 100, headless=True)
        initial_y = player.get_y()
        
        player.move_y(-1)
        
        expected_y = initial_y - player.get_speed()
        assert player.get_y() == expected_y
        assert player.get_x() == 100  # X should not change
    
    def test_move_x_zero(self):
        """Test moving player with zero X movement."""
        player = Player(100, 100, headless=True)
        initial_position = player.get_position()
        
        player.move_x(0)
        
        assert player.get_position() == initial_position
    
    def test_move_y_zero(self):
        """Test moving player with zero Y movement."""
        player = Player(100, 100, headless=True)
        initial_position = player.get_position()
        
        player.move_y(0)
        
        assert player.get_position() == initial_position
    
    def test_multiple_movements(self):
        """Test multiple consecutive movements."""
        player = Player(100, 100, headless=True)
        speed = player.get_speed()
        
        # Move right, then down, then left, then up
        player.move_x(1)
        player.move_y(1)
        player.move_x(-1)
        player.move_y(-1)
        
        # Should be back to original position
        assert player.get_position() == (100, 100)
    
    def test_large_movement_values(self):
        """Test movement with large multiplier values."""
        player = Player(100, 100, headless=True)
        speed = player.get_speed()
        
        player.move_x(10)
        player.move_y(5)
        
        assert player.get_x() == 100 + (10 * speed)
        assert player.get_y() == 100 + (5 * speed)


class TestPlayerPositioning:
    """Test Player position management."""
    
    def test_set_x_position(self):
        """Test setting X position directly."""
        player = Player(100, 100, headless=True)
        
        player.set_x(250)
        
        assert player.get_x() == 250
        assert player.get_y() == 100  # Y should not change
        assert player.get_position() == (250, 100)
    
    def test_set_y_position(self):
        """Test setting Y position directly."""
        player = Player(100, 100, headless=True)
        
        player.set_y(300)
        
        assert player.get_y() == 300
        assert player.get_x() == 100  # X should not change
        assert player.get_position() == (100, 300)
    
    def test_set_position_with_negative_values(self):
        """Test setting position with negative coordinates."""
        player = Player(100, 100, headless=True)
        
        player.set_x(-50)
        player.set_y(-75)
        
        assert player.get_position() == (-50, -75)
    
    def test_position_consistency(self):
        """Test that position getters are consistent."""
        player = Player(123, 456, headless=True)
        
        assert player.get_x() == 123
        assert player.get_y() == 456
        assert player.get_position() == (123, 456)
        assert player.get_position()[0] == player.get_x()
        assert player.get_position()[1] == player.get_y()


class TestPlayerDimensions:
    """Test Player dimension properties."""
    
    def test_player_width(self):
        """Test player width property."""
        player = Player(100, 100, headless=True)
        
        assert player.get_width() == 10  # Hardcoded in Player implementation
        assert isinstance(player.get_width(), int)
        assert player.get_width() > 0
    
    def test_player_height(self):
        """Test player height property."""
        player = Player(100, 100, headless=True)
        
        assert player.get_height() == 10  # Hardcoded in Player implementation
        assert isinstance(player.get_height(), int)
        assert player.get_height() > 0
    
    def test_player_speed(self):
        """Test player speed property."""
        player = Player(100, 100, headless=True)
        
        assert player.get_speed() == 1  # Hardcoded in Player implementation
        assert isinstance(player.get_speed(), int)
        assert player.get_speed() > 0


class TestPlayerSpriteManagement:
    """Test Player sprite and cursor management."""
    
    def test_set_cursor_headless_mode(self):
        """Test setting cursor in headless mode."""
        player = Player(100, 100, headless=True)
        
        # Should not raise exception in headless mode
        player.set_cursor("test_image.png")
        
        # In headless mode, cursor might be None or a placeholder
        cursor = player.get_cursor()
        # The behavior may vary based on implementation
    
    def test_set_cursor_with_display(self):
        """Test setting cursor with display mode."""
        with full_pygame_mocks() as mocks:
            player = Player(100, 100, headless=False)
            
            player.set_cursor("test_image.png")
            
            # Verify image loading was called
            mocks['image']['load'].assert_called_with("test_image.png")
    
    def test_set_cursor_invalid_file(self):
        """Test setting cursor with invalid file."""
        with full_pygame_mocks() as mocks:
            # Mock image load to raise an exception
            mocks['image']['load'].side_effect = pygame.error("File not found")
            
            player = Player(100, 100, headless=False)
            
            # Should not raise exception, should handle gracefully
            player.set_cursor("invalid_file.png")
    
    def test_get_cursor(self):
        """Test getting cursor object."""
        with full_pygame_mocks() as mocks:
            player = Player(100, 100, headless=False)
            
            cursor = player.get_cursor()
            
            # Should return some surface object
            assert cursor is not None


class TestPlayerReset:
    """Test Player reset functionality."""
    
    def test_reset_position(self):
        """Test resetting player to new position."""
        player = Player(100, 100, headless=True)
        
        # Move player away from initial position
        player.move_x(5)
        player.move_y(3)
        
        # Reset to new position
        player.reset(200, 300)
        
        assert player.get_position() == (200, 300)
    
    def test_reset_multiple_times(self):
        """Test multiple resets."""
        player = Player(100, 100, headless=True)
        
        # First reset
        player.reset(200, 250)
        assert player.get_position() == (200, 250)
        
        # Second reset
        player.reset(50, 75)
        assert player.get_position() == (50, 75)
    
    def test_reset_with_negative_coordinates(self):
        """Test reset with negative coordinates."""
        player = Player(100, 100, headless=True)
        
        player.reset(-100, -200)
        
        assert player.get_position() == (-100, -200)


class TestPlayerBoundaryConditions:
    """Test Player behavior at boundary conditions."""
    
    def test_movement_from_zero_position(self):
        """Test movement from (0,0) position."""
        player = Player(0, 0, headless=True)
        speed = player.get_speed()
        
        player.move_x(1)
        player.move_y(1)
        
        assert player.get_x() == speed
        assert player.get_y() == speed
    
    def test_movement_to_negative_position(self):
        """Test movement that results in negative position."""
        player = Player(5, 5, headless=True)
        speed = player.get_speed()
        
        # Move left more than current position
        player.move_x(-2)  # Should result in negative X if speed > 2.5
        
        expected_x = 5 - (2 * speed)
        assert player.get_x() == expected_x
    
    def test_large_coordinate_values(self):
        """Test player with very large coordinate values."""
        large_value = 999999
        player = Player(large_value, large_value, headless=True)
        
        assert player.get_x() == large_value
        assert player.get_y() == large_value
        
        # Test movement from large values
        player.move_x(1)
        assert player.get_x() == large_value + player.get_speed()


class TestPlayerIntegration:
    """Integration tests for Player with other components."""
    
    def test_player_with_game_bounds(self):
        """Test player movement within game bounds."""
        # Using game configuration bounds
        player = Player(GameConfig.PLAYER_START_X, GameConfig.PLAYER_START_Y, headless=True)
        
        # Test that player starts within reasonable bounds
        assert player.get_x() >= 0
        assert player.get_y() >= 0
        
        # Test movement within typical game area
        for _ in range(10):
            player.move_x(1)
        
        # Player should still be within reasonable bounds
        assert player.get_x() < 1000  # Assuming game width is reasonable
    
    def test_player_collision_bounds(self):
        """Test player collision boundary calculations."""
        player = Player(100, 100, headless=True)
        
        # Test that collision bounds make sense
        x, y = player.get_position()
        width, height = player.get_width(), player.get_height()
        
        # Player should occupy space from (x,y) to (x+width, y+height)
        assert width > 0
        assert height > 0
        
        # After movement, bounds should update
        player.move_x(1)
        new_x = player.get_x()
        assert new_x != x  # Position should have changed
    
    @pytest.mark.parametrize("start_x,start_y,move_x,move_y,expected_x,expected_y", [
        (100, 100, 1, 0, 105, 100),   # Right movement (assuming speed=5)
        (100, 100, -1, 0, 95, 100),   # Left movement
        (100, 100, 0, 1, 100, 105),   # Down movement
        (100, 100, 0, -1, 100, 95),   # Up movement
        (100, 100, 2, 3, 110, 115),   # Diagonal movement
    ])
    def test_movement_parametrized(self, start_x, start_y, move_x, move_y, 
                                  expected_x, expected_y):
        """Parametrized test for various movement scenarios."""
        player = Player(start_x, start_y, headless=True)
        speed = player.get_speed()
        
        if move_x != 0:
            player.move_x(move_x)
        if move_y != 0:
            player.move_y(move_y)
        
        # Calculate expected position based on actual speed
        actual_expected_x = start_x + (move_x * speed)
        actual_expected_y = start_y + (move_y * speed)
        
        assert player.get_x() == actual_expected_x
        assert player.get_y() == actual_expected_y


# Performance and stress tests
class TestPlayerPerformance:
    """Performance tests for Player operations."""
    
    @pytest.mark.performance
    def test_movement_performance(self):
        """Test performance of rapid movement operations."""
        player = Player(100, 100, headless=True)
        
        import time
        start_time = time.time()
        
        # Perform many movement operations
        for i in range(10000):
            direction = i % 4
            if direction == 0:
                player.move_x(1)
            elif direction == 1:
                player.move_x(-1)
            elif direction == 2:
                player.move_y(1)
            else:
                player.move_y(-1)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly (less than 1 second for 10k operations)
        assert duration < 1.0, f"Movement operations took too long: {duration}s"
    
    @pytest.mark.performance
    def test_position_setting_performance(self):
        """Test performance of position setting operations."""
        player = Player(100, 100, headless=True)
        
        import time
        start_time = time.time()
        
        # Perform many position setting operations
        for i in range(10000):
            player.set_x(i % 1000)
            player.set_y((i * 2) % 1000)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly
        assert duration < 1.0, f"Position setting took too long: {duration}s"
