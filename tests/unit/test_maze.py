"""
Unit tests for the Line/Maze classes in Infinite Maze.

These tests verify the functionality of the Line class including
line positioning, maze generation, collision boundaries, and
maze management operations.
"""

import pytest
from unittest.mock import Mock, patch

from infinite_maze.entities.maze import Line
from infinite_maze.core.game import Game


class TestLineInitialization:
    """Test Line class initialization."""
    
    def test_line_init_default(self):
        """Test line initialization with default parameters."""
        line = Line()
        
        assert line.get_start() == (0, 0)
        assert line.get_end() == (0, 0)
        assert line.get_side_a() == 0
        assert line.get_side_b() == 0
        assert line.get_is_horizontal() is True  # Same Y coordinates
    
    def test_line_init_with_parameters(self):
        """Test line initialization with specific parameters."""
        start_pos = (10, 20)
        end_pos = (30, 40)
        side_a = 5
        side_b = 7
        
        line = Line(start_pos, end_pos, side_a, side_b)
        
        assert line.get_start() == start_pos
        assert line.get_end() == end_pos
        assert line.get_side_a() == side_a
        assert line.get_side_b() == side_b
    
    def test_line_horizontal_detection(self):
        """Test horizontal line detection."""
        # Horizontal line (same Y coordinates)
        horizontal_line = Line((10, 20), (30, 20))
        assert horizontal_line.get_is_horizontal() is True
        
        # Vertical line (different Y coordinates)
        vertical_line = Line((10, 20), (10, 40))
        assert vertical_line.get_is_horizontal() is False
        
        # Diagonal line
        diagonal_line = Line((10, 20), (30, 40))
        assert diagonal_line.get_is_horizontal() is False
    
    def test_line_negative_coordinates(self):
        """Test line with negative coordinates."""
        line = Line((-10, -20), (-5, -15))
        
        assert line.get_start() == (-10, -20)
        assert line.get_end() == (-5, -15)


class TestLinePositionGetters:
    """Test Line position getter methods."""
    
    def test_get_start_end(self):
        """Test getting start and end positions."""
        line = Line((10, 20), (30, 40))
        
        assert line.get_start() == (10, 20)
        assert line.get_end() == (30, 40)
    
    def test_get_individual_coordinates(self):
        """Test getting individual coordinate values."""
        line = Line((15, 25), (35, 45))
        
        assert line.get_x_start() == 15
        assert line.get_y_start() == 25
        assert line.get_x_end() == 35
        assert line.get_y_end() == 45
    
    def test_coordinate_consistency(self):
        """Test that coordinates are consistent across methods."""
        line = Line((100, 200), (300, 400))
        
        start = line.get_start()
        end = line.get_end()
        
        assert start[0] == line.get_x_start()
        assert start[1] == line.get_y_start()
        assert end[0] == line.get_x_end()
        assert end[1] == line.get_y_end()


class TestLinePositionSetters:
    """Test Line position setter methods."""
    
    def test_set_start_position(self):
        """Test setting start position."""
        line = Line((10, 20), (30, 40))
        
        new_start = (50, 60)
        line.set_start(new_start)
        
        assert line.get_start() == new_start
        assert line.get_end() == (30, 40)  # End should not change
    
    def test_set_end_position(self):
        """Test setting end position."""
        line = Line((10, 20), (30, 40))
        
        new_end = (70, 80)
        line.set_end(new_end)
        
        assert line.get_start() == (10, 20)  # Start should not change
        assert line.get_end() == new_end
    
    def test_set_individual_coordinates(self):
        """Test setting individual coordinates."""
        line = Line((10, 20), (30, 40))
        
        # Set start coordinates
        line.set_x_start(100)
        assert line.get_x_start() == 100
        assert line.get_y_start() == 20  # Y should not change
        
        line.set_y_start(200)
        assert line.get_x_start() == 100  # X should not change
        assert line.get_y_start() == 200
        
        # Set end coordinates
        line.set_x_end(300)
        assert line.get_x_end() == 300
        assert line.get_y_end() == 40  # Y should not change
        
        line.set_y_end(400)
        assert line.get_x_end() == 300  # X should not change
        assert line.get_y_end() == 400
    
    def test_coordinate_setting_updates_position(self):
        """Test that setting coordinates updates position tuples."""
        line = Line((10, 20), (30, 40))
        
        line.set_x_start(50)
        line.set_y_start(60)
        
        assert line.get_start() == (50, 60)
        
        line.set_x_end(70)
        line.set_y_end(80)
        
        assert line.get_end() == (70, 80)


class TestLineSideManagement:
    """Test Line side management for maze generation."""
    
    def test_get_sides(self):
        """Test getting side values."""
        line = Line((0, 0), (10, 10), 5, 7)
        
        assert line.get_side_a() == 5
        assert line.get_side_b() == 7
    
    def test_set_sides(self):
        """Test setting side values."""
        line = Line((0, 0), (10, 10), 1, 2)
        
        line.set_side_a(10)
        line.set_side_b(15)
        
        assert line.get_side_a() == 10
        assert line.get_side_b() == 15
    
    def test_sides_independence(self):
        """Test that side A and B are independent."""
        line = Line((0, 0), (10, 10), 1, 2)
        
        line.set_side_a(100)
        assert line.get_side_b() == 2  # Should not change
        
        line.set_side_b(200)
        assert line.get_side_a() == 100  # Should not change


class TestLineOrientationHandling:
    """Test Line orientation detection and management."""
    
    def test_horizontal_line_detection(self):
        """Test detection of horizontal lines."""
        horizontal_line = Line((10, 50), (100, 50))  # Same Y
        assert horizontal_line.get_is_horizontal() is True
    
    def test_vertical_line_detection(self):
        """Test detection of vertical lines."""
        vertical_line = Line((50, 10), (50, 100))  # Same X
        assert vertical_line.get_is_horizontal() is False
    
    def test_orientation_after_position_change(self):
        """Test orientation detection after position changes."""
        line = Line((10, 20), (30, 40))  # Initially diagonal
        assert line.get_is_horizontal() is False
        
        # Change to horizontal
        line.set_y_end(20)  # Make Y coordinates same
        # Note: reset_is_horizontal would need to be called or implemented differently
        # The current implementation doesn't automatically update is_horizontal
    
    def test_reset_is_horizontal(self):
        """Test reset_is_horizontal method."""
        line = Line((10, 20), (30, 40))
        
        # Change position to horizontal
        line.set_y_end(20)
        
        # Reset orientation detection
        line.reset_is_horizontal()
        
        # Should now detect as horizontal
        assert line.get_is_horizontal() is True


class TestLineMazeGeneration:
    """Test Line maze generation functionality."""
    
    def test_generate_maze_basic(self):
        """Test basic maze generation."""
        # Create a mock game object
        mock_game = Mock()
        mock_game.X_MAX = 100
        mock_game.Y_MIN = 50
        
        # Generate small maze
        lines = Line.generate_maze(mock_game, 3, 3)
        
        # Should return a list of lines
        assert isinstance(lines, list)
        assert len(lines) > 0
        
        # All items should be Line objects
        for line in lines:
            assert isinstance(line, Line)
    
    def test_generate_maze_dimensions(self):
        """Test maze generation with different dimensions."""
        mock_game = Mock()
        mock_game.X_MAX = 200
        mock_game.Y_MIN = 100
        
        # Test different sizes
        sizes = [(2, 2), (5, 5), (10, 8)]
        
        for width, height in sizes:
            lines = Line.generate_maze(mock_game, width, height)
            
            # Should generate lines
            assert len(lines) > 0
            
            # Number of lines should be related to maze size
            # (exact formula depends on implementation)
            assert len(lines) > width + height
    
    def test_generate_maze_line_properties(self):
        """Test properties of generated maze lines."""
        mock_game = Mock()
        mock_game.X_MAX = 100
        mock_game.Y_MIN = 50
        
        lines = Line.generate_maze(mock_game, 3, 3)
        
        for line in lines:
            # Each line should have valid coordinates
            start = line.get_start()
            end = line.get_end()
            
            assert isinstance(start[0], int)
            assert isinstance(start[1], int)
            assert isinstance(end[0], int)
            assert isinstance(end[1], int)
            
            # Lines should have valid side values
            assert isinstance(line.get_side_a(), int)
            assert isinstance(line.get_side_b(), int)
    
    def test_generate_maze_with_game_bounds(self):
        """Test that generated maze respects game bounds."""
        mock_game = Mock()
        mock_game.X_MAX = 300
        mock_game.Y_MIN = 40
        
        lines = Line.generate_maze(mock_game, 5, 5)
        
        for line in lines:
            # Lines should be positioned relative to game bounds
            assert line.get_x_start() >= mock_game.X_MAX
            assert line.get_y_start() >= mock_game.Y_MIN


class TestLineUtilityMethods:
    """Test Line utility methods."""
    
    def test_get_x_max_empty_list(self):
        """Test get_x_max with empty line list."""
        max_x = Line.get_x_max([])
        assert max_x == 0
    
    def test_get_x_max_single_line(self):
        """Test get_x_max with single line."""
        line = Line((10, 20), (50, 30))
        max_x = Line.get_x_max([line])
        assert max_x == 50
    
    def test_get_x_max_multiple_lines(self):
        """Test get_x_max with multiple lines."""
        lines = [
            Line((10, 20), (50, 30)),   # max X = 50
            Line((30, 40), (80, 50)),   # max X = 80
            Line((5, 10), (25, 15)),    # max X = 25
            Line((60, 70), (120, 80))   # max X = 120
        ]
        
        max_x = Line.get_x_max(lines)
        assert max_x == 120
    
    def test_get_x_max_negative_coordinates(self):
        """Test get_x_max with negative coordinates."""
        lines = [
            Line((-50, 0), (-20, 10)),  # max X = -20
            Line((-30, 0), (-10, 10)),  # max X = -10
            Line((-40, 0), (-35, 10))   # max X = -35
        ]
        
        max_x = Line.get_x_max(lines)
        assert max_x == -10
    
    def test_get_x_max_mixed_coordinates(self):
        """Test get_x_max with mixed positive/negative coordinates."""
        lines = [
            Line((-50, 0), (30, 10)),   # max X = 30
            Line((10, 0), (-5, 10)),    # max X = 10
            Line((-10, 0), (50, 10))    # max X = 50
        ]
        
        max_x = Line.get_x_max(lines)
        assert max_x == 50


class TestLineCollisionBoundaries:
    """Test Line collision boundary calculations."""
    
    def test_horizontal_line_boundaries(self):
        """Test horizontal line collision boundaries."""
        line = Line((100, 200), (200, 200))  # Horizontal line
        
        # Line should span from X=100 to X=200 at Y=200
        assert line.get_x_start() == 100
        assert line.get_x_end() == 200
        assert line.get_y_start() == 200
        assert line.get_y_end() == 200
        assert line.get_is_horizontal() is True
    
    def test_vertical_line_boundaries(self):
        """Test vertical line collision boundaries."""
        line = Line((150, 100), (150, 300))  # Vertical line
        
        # Line should span from Y=100 to Y=300 at X=150
        assert line.get_x_start() == 150
        assert line.get_x_end() == 150
        assert line.get_y_start() == 100
        assert line.get_y_end() == 300
        assert line.get_is_horizontal() is False
    
    def test_line_length_calculation(self):
        """Test calculating line length."""
        # Horizontal line
        h_line = Line((10, 50), (110, 50))
        h_length = h_line.get_x_end() - h_line.get_x_start()
        assert h_length == 100
        
        # Vertical line
        v_line = Line((50, 10), (50, 60))
        v_length = v_line.get_y_end() - v_line.get_y_start()
        assert v_length == 50


class TestLineEdgeCases:
    """Test Line edge cases and error conditions."""
    
    def test_zero_length_line(self):
        """Test line with zero length."""
        line = Line((50, 50), (50, 50))
        
        assert line.get_start() == line.get_end()
        assert line.get_x_start() == line.get_x_end()
        assert line.get_y_start() == line.get_y_end()
        assert line.get_is_horizontal() is True  # Same Y coordinates
    
    def test_negative_coordinate_line(self):
        """Test line with negative coordinates."""
        line = Line((-100, -50), (-20, -30))
        
        assert line.get_x_start() == -100
        assert line.get_y_start() == -50
        assert line.get_x_end() == -20
        assert line.get_y_end() == -30
    
    def test_large_coordinate_values(self):
        """Test line with very large coordinates."""
        large_val = 999999
        line = Line((0, 0), (large_val, large_val))
        
        assert line.get_x_end() == large_val
        assert line.get_y_end() == large_val
    
    def test_line_coordinate_modification_sequence(self):
        """Test sequence of coordinate modifications."""
        line = Line((10, 20), (30, 40))
        
        # Sequence of modifications
        modifications = [
            ('set_x_start', 100),
            ('set_y_start', 200),
            ('set_x_end', 300),
            ('set_y_end', 400),
            ('set_side_a', 5),
            ('set_side_b', 7)
        ]
        
        for method_name, value in modifications:
            method = getattr(line, method_name)
            method(value)
        
        # Check final state
        assert line.get_start() == (100, 200)
        assert line.get_end() == (300, 400)
        assert line.get_side_a() == 5
        assert line.get_side_b() == 7


class TestLineMazeIntegration:
    """Integration tests for Line with maze generation."""
    
    def test_maze_generation_with_real_game(self):
        """Test maze generation with real Game object."""
        # This test uses the actual Game class
        game = Game(headless=True)
        
        lines = Line.generate_maze(game, 5, 5)
        
        # Should generate valid maze
        assert len(lines) > 0
        
        # Lines should be positioned relative to game boundaries
        for line in lines:
            # All lines should be within reasonable bounds
            assert line.get_x_start() >= 0
            assert line.get_y_start() >= 0
            assert line.get_x_end() >= line.get_x_start()
            
            if not line.get_is_horizontal():
                assert line.get_y_end() >= line.get_y_start()
    
    def test_generated_maze_connectivity(self):
        """Test that generated maze has proper connectivity."""
        mock_game = Mock()
        mock_game.X_MAX = 100
        mock_game.Y_MIN = 50
        
        lines = Line.generate_maze(mock_game, 4, 4)
        
        # Generated maze should form a connected structure
        # (specific connectivity tests would depend on maze algorithm)
        assert len(lines) > 0
        
        # Should have both horizontal and vertical lines
        horizontal_lines = [line for line in lines if line.get_is_horizontal()]
        vertical_lines = [line for line in lines if not line.get_is_horizontal()]
        
        # In a proper maze, we expect both types of lines
        # (unless it's a very simple maze)
        assert len(horizontal_lines) > 0 or len(vertical_lines) > 0
    
    def test_maze_line_positioning_pattern(self):
        """Test that maze lines follow expected positioning pattern."""
        mock_game = Mock()
        mock_game.X_MAX = 100
        mock_game.Y_MIN = 50
        
        lines = Line.generate_maze(mock_game, 3, 3)
        
        # Lines should follow grid-based positioning
        # (specific pattern depends on maze generation algorithm)
        x_positions = set()
        y_positions = set()
        
        for line in lines:
            x_positions.add(line.get_x_start())
            x_positions.add(line.get_x_end())
            y_positions.add(line.get_y_start())
            y_positions.add(line.get_y_end())
        
        # Should have multiple distinct positions
        assert len(x_positions) > 1
        assert len(y_positions) > 1


@pytest.mark.performance
class TestLinePerformance:
    """Performance tests for Line operations."""
    
    def test_line_creation_performance(self):
        """Test performance of creating many lines."""
        import time
        start_time = time.time()
        
        lines = []
        for i in range(10000):
            line = Line((i, i), (i+10, i+10), i, i+1)
            lines.append(line)
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 1.0, f"Line creation took too long: {duration}s"
        assert len(lines) == 10000
    
    def test_coordinate_access_performance(self):
        """Test performance of coordinate access operations."""
        line = Line((100, 200), (300, 400))
        
        import time
        start_time = time.time()
        
        # Perform many coordinate access operations
        for _ in range(100000):
            _ = line.get_x_start()
            _ = line.get_y_start()
            _ = line.get_x_end()
            _ = line.get_y_end()
            _ = line.get_start()
            _ = line.get_end()
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 1.0, f"Coordinate access took too long: {duration}s"
    
    def test_maze_generation_performance(self):
        """Test performance of maze generation."""
        mock_game = Mock()
        mock_game.X_MAX = 100
        mock_game.Y_MIN = 50
        
        import time
        start_time = time.time()
        
        # Generate multiple mazes
        for _ in range(100):
            lines = Line.generate_maze(mock_game, 10, 10)
            assert len(lines) > 0
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 5.0, f"Maze generation took too long: {duration}s"
