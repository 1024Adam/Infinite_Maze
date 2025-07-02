"""
Unit tests for the configuration module in Infinite Maze.

These tests verify the functionality of the GameConfig class including
configuration settings, asset paths, color management, and constants.
"""

import pytest
import os
from unittest.mock import patch, Mock

from infinite_maze.utils.config import GameConfig, config


class TestGameConfigConstants:
    """Test GameConfig constant values."""
    
    def test_display_settings(self):
        """Test display configuration constants."""
        assert GameConfig.SCREEN_WIDTH > 0
        assert GameConfig.SCREEN_HEIGHT > 0
        assert GameConfig.FPS > 0
        
        # Common screen resolutions should be reasonable
        assert 400 <= GameConfig.SCREEN_WIDTH <= 3840
        assert 300 <= GameConfig.SCREEN_HEIGHT <= 2160
        assert 30 <= GameConfig.FPS <= 144
    
    def test_player_settings(self):
        """Test player configuration constants."""
        assert GameConfig.PLAYER_START_X >= 0
        assert GameConfig.PLAYER_START_Y >= 0
        assert GameConfig.PLAYER_SPEED > 0
        assert GameConfig.PLAYER_WIDTH > 0
        assert GameConfig.PLAYER_HEIGHT > 0
        
        # Player should start within screen bounds
        assert GameConfig.PLAYER_START_X < GameConfig.SCREEN_WIDTH
        assert GameConfig.PLAYER_START_Y < GameConfig.SCREEN_HEIGHT
    
    def test_maze_settings(self):
        """Test maze configuration constants."""
        assert GameConfig.MAZE_ROWS > 0
        assert GameConfig.MAZE_COLS > 0
        
        # Reasonable maze sizes
        assert 1 <= GameConfig.MAZE_ROWS <= 100
        assert 1 <= GameConfig.MAZE_COLS <= 100
    
    def test_asset_directories(self):
        """Test asset directory configuration."""
        assert isinstance(GameConfig.ASSETS_DIR, str)
        assert isinstance(GameConfig.IMAGES_DIR, str)
        assert len(GameConfig.ASSETS_DIR) > 0
        assert len(GameConfig.IMAGES_DIR) > 0


class TestGameConfigColors:
    """Test GameConfig color management."""
    
    def test_colors_dictionary(self):
        """Test colors dictionary structure."""
        colors = GameConfig.COLORS
        
        assert isinstance(colors, dict)
        assert len(colors) > 0
        
        # Check for basic colors
        basic_colors = ['BLACK', 'WHITE', 'RED', 'GREEN', 'BLUE']
        for color in basic_colors:
            assert color in colors
    
    def test_color_format(self):
        """Test color format (RGB tuples)."""
        for color_name, color_value in GameConfig.COLORS.items():
            assert isinstance(color_value, tuple)
            assert len(color_value) == 3
            
            # Each RGB component should be 0-255
            for component in color_value:
                assert isinstance(component, int)
                assert 0 <= component <= 255
    
    def test_get_color_valid(self):
        """Test getting valid colors."""
        # Test existing colors
        black = GameConfig.get_color('BLACK')
        assert black == (0, 0, 0)
        
        white = GameConfig.get_color('WHITE')
        assert white == (255, 255, 255)
        
        # Test case insensitive
        red = GameConfig.get_color('red')
        assert red == GameConfig.COLORS['RED']
    
    def test_get_color_invalid(self):
        """Test getting invalid colors returns default."""
        invalid_color = GameConfig.get_color('INVALID_COLOR')
        assert invalid_color == GameConfig.COLORS['WHITE']  # Default
        
        # Test empty string
        empty_color = GameConfig.get_color('')
        assert empty_color == GameConfig.COLORS['WHITE']
        
        # Test None (should handle gracefully)
        try:
            none_color = GameConfig.get_color(None)
            # Should either return default or handle gracefully
        except (TypeError, AttributeError):
            # Acceptable if it raises an appropriate exception
            pass
    
    def test_color_constants(self):
        """Test specific color constant values."""
        colors = GameConfig.COLORS
        
        # Test basic color values
        assert colors['BLACK'] == (0, 0, 0)
        assert colors['WHITE'] == (255, 255, 255)
        assert colors['RED'] == (255, 0, 0)
        assert colors['GREEN'] == (0, 255, 0)
        assert colors['BLUE'] == (0, 0, 255)


class TestGameConfigControls:
    """Test GameConfig control mappings."""
    
    def test_controls_structure(self):
        """Test controls dictionary structure."""
        controls = GameConfig.CONTROLS
        
        assert isinstance(controls, dict)
        assert len(controls) > 0
        
        # Check for required control mappings
        required_controls = ['MOVE_RIGHT', 'MOVE_LEFT', 'MOVE_UP', 'MOVE_DOWN', 'PAUSE', 'QUIT']
        for control in required_controls:
            assert control in controls
    
    def test_control_values(self):
        """Test control mapping values."""
        for control_name, control_keys in GameConfig.CONTROLS.items():
            assert isinstance(control_keys, list)
            assert len(control_keys) > 0
            
            # Each key should be a string
            for key in control_keys:
                assert isinstance(key, str)
                assert len(key) > 0
    
    def test_movement_controls(self):
        """Test movement control mappings."""
        controls = GameConfig.CONTROLS
        
        # Test that movement controls have expected keys
        assert 'RIGHT' in controls['MOVE_RIGHT'] or 'd' in controls['MOVE_RIGHT']
        assert 'LEFT' in controls['MOVE_LEFT'] or 'a' in controls['MOVE_LEFT']
        assert 'UP' in controls['MOVE_UP'] or 'w' in controls['MOVE_UP']
        assert 'DOWN' in controls['MOVE_DOWN'] or 's' in controls['MOVE_DOWN']
    
    def test_system_controls(self):
        """Test system control mappings."""
        controls = GameConfig.CONTROLS
        
        # Test pause control
        pause_keys = controls['PAUSE']
        assert 'SPACE' in pause_keys
        
        # Test quit controls
        quit_keys = controls['QUIT']
        assert 'ESCAPE' in quit_keys or 'q' in quit_keys


class TestGameConfigImages:
    """Test GameConfig image asset management."""
    
    def test_images_dictionary(self):
        """Test images dictionary structure."""
        images = GameConfig.IMAGES
        
        assert isinstance(images, dict)
        assert len(images) > 0
        
        # Check for required images
        required_images = ['player', 'player_paused', 'icon']
        for image in required_images:
            assert image in images
    
    def test_image_paths(self):
        """Test image path formats."""
        for image_name, image_path in GameConfig.IMAGES.items():
            assert isinstance(image_path, str)
            assert len(image_path) > 0
            
            # Should be relative paths starting with assets
            assert image_path.startswith('assets/')
            
            # Should end with image extension
            valid_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
            assert any(image_path.lower().endswith(ext) for ext in valid_extensions)
    
    def test_get_image_path_valid(self):
        """Test getting valid image paths."""
        player_path = GameConfig.get_image_path('player')
        assert player_path == GameConfig.IMAGES['player']
        
        icon_path = GameConfig.get_image_path('icon')
        assert icon_path == GameConfig.IMAGES['icon']
    
    def test_get_image_path_invalid(self):
        """Test getting invalid image paths."""
        invalid_path = GameConfig.get_image_path('invalid_image')
        assert invalid_path == ""  # Should return empty string
        
        # Test empty string
        empty_path = GameConfig.get_image_path('')
        assert empty_path == ""


class TestGameConfigAssetPaths:
    """Test GameConfig asset path utilities."""
    
    def test_get_asset_path_single(self):
        """Test getting asset path with single component."""
        path = GameConfig.get_asset_path('test.png')
        expected = os.path.join(GameConfig.ASSETS_DIR, 'test.png')
        assert path == expected
    
    def test_get_asset_path_multiple(self):
        """Test getting asset path with multiple components."""
        path = GameConfig.get_asset_path('images', 'player.png')
        expected = os.path.join(GameConfig.ASSETS_DIR, 'images', 'player.png')
        assert path == expected
    
    def test_get_asset_path_nested(self):
        """Test getting asset path with nested directories."""
        path = GameConfig.get_asset_path('sounds', 'effects', 'jump.wav')
        expected = os.path.join(GameConfig.ASSETS_DIR, 'sounds', 'effects', 'jump.wav')
        assert path == expected
    
    def test_get_asset_path_empty(self):
        """Test getting asset path with empty components."""
        path = GameConfig.get_asset_path()
        assert path == GameConfig.ASSETS_DIR


class TestGameConfigMovementConstants:
    """Test GameConfig movement constants."""
    
    def test_movement_constants_structure(self):
        """Test movement constants dictionary structure."""
        constants = GameConfig.MOVEMENT_CONSTANTS
        
        assert isinstance(constants, dict)
        assert len(constants) > 0
        
        # Check for required constants
        required_constants = ['DO_NOTHING', 'RIGHT', 'LEFT', 'UP', 'DOWN']
        for constant in required_constants:
            assert constant in constants
    
    def test_movement_constant_values(self):
        """Test movement constant values."""
        constants = GameConfig.MOVEMENT_CONSTANTS
        
        # Should be integers
        for constant_name, constant_value in constants.items():
            assert isinstance(constant_value, int)
        
        # Should have unique values
        values = list(constants.values())
        assert len(values) == len(set(values))  # No duplicates
        
        # Should include 0 for DO_NOTHING
        assert constants['DO_NOTHING'] == 0
    
    def test_get_movement_constant_valid(self):
        """Test getting valid movement constants."""
        right_const = GameConfig.get_movement_constant('RIGHT')
        assert right_const == GameConfig.MOVEMENT_CONSTANTS['RIGHT']
        
        # Test case insensitive
        left_const = GameConfig.get_movement_constant('left')
        assert left_const == GameConfig.MOVEMENT_CONSTANTS['LEFT']
    
    def test_get_movement_constant_invalid(self):
        """Test getting invalid movement constants."""
        invalid_const = GameConfig.get_movement_constant('INVALID')
        assert invalid_const == GameConfig.MOVEMENT_CONSTANTS['DO_NOTHING']  # Default
        
        # Test empty string
        empty_const = GameConfig.get_movement_constant('')
        assert empty_const == GameConfig.MOVEMENT_CONSTANTS['DO_NOTHING']


class TestConfigInstanceValidation:
    """Test the default config instance."""
    
    def test_config_instance_exists(self):
        """Test that default config instance exists."""
        assert config is not None
        assert isinstance(config, GameConfig)
    
    def test_config_instance_functionality(self):
        """Test that default config instance works."""
        # Test color access
        black = config.get_color('BLACK')
        assert black == (0, 0, 0)
        
        # Test image path access
        player_path = config.get_image_path('player')
        assert isinstance(player_path, str)
        
        # Test movement constant access
        right_const = config.get_movement_constant('RIGHT')
        assert isinstance(right_const, int)
    
    def test_config_constants_accessible(self):
        """Test that constants are accessible through instance."""
        assert config.SCREEN_WIDTH > 0
        assert config.SCREEN_HEIGHT > 0
        assert config.PLAYER_SPEED > 0
        assert len(config.COLORS) > 0


class TestGameConfigValidation:
    """Test GameConfig value validation."""
    
    def test_screen_dimensions_reasonable(self):
        """Test that screen dimensions are reasonable."""
        # Should be common aspect ratios
        width = GameConfig.SCREEN_WIDTH
        height = GameConfig.SCREEN_HEIGHT
        
        # Common aspect ratios: 4:3, 16:9, 16:10
        aspect_ratio = width / height
        common_ratios = [4/3, 16/9, 16/10, 5/4, 3/2]
        
        # Should be close to a common aspect ratio (within 0.1)
        assert any(abs(aspect_ratio - ratio) < 0.1 for ratio in common_ratios)
    
    def test_player_settings_consistency(self):
        """Test that player settings are consistent."""
        # Player should fit within screen
        assert GameConfig.PLAYER_START_X + GameConfig.PLAYER_WIDTH <= GameConfig.SCREEN_WIDTH
        assert GameConfig.PLAYER_START_Y + GameConfig.PLAYER_HEIGHT <= GameConfig.SCREEN_HEIGHT
        
        # Speed should be reasonable relative to player size
        assert GameConfig.PLAYER_SPEED <= min(GameConfig.PLAYER_WIDTH, GameConfig.PLAYER_HEIGHT)
    
    def test_maze_settings_reasonable(self):
        """Test that maze settings are reasonable."""
        # Maze should not be too large for screen
        total_maze_width = GameConfig.MAZE_COLS * 22  # Assuming 22 pixel spacing
        total_maze_height = GameConfig.MAZE_ROWS * 22
        
        # Should fit reasonably within screen (allowing for UI)
        assert total_maze_width <= GameConfig.SCREEN_WIDTH * 3  # Allow for scrolling
        assert total_maze_height <= GameConfig.SCREEN_HEIGHT * 2


class TestGameConfigEdgeCases:
    """Test GameConfig edge cases and error conditions."""
    
    def test_color_case_variations(self):
        """Test color name case variations."""
        # Test various case combinations
        test_cases = ['BLACK', 'black', 'Black', 'bLaCk']
        
        for case_variant in test_cases:
            color = GameConfig.get_color(case_variant)
            assert color == (0, 0, 0)  # Should all return black
    
    def test_movement_constant_case_variations(self):
        """Test movement constant case variations."""
        test_cases = ['RIGHT', 'right', 'Right', 'rIgHt']
        
        expected_value = GameConfig.MOVEMENT_CONSTANTS['RIGHT']
        for case_variant in test_cases:
            constant = GameConfig.get_movement_constant(case_variant)
            assert constant == expected_value
    
    def test_asset_path_edge_cases(self):
        """Test asset path with edge cases."""
        # Test with empty components
        path1 = GameConfig.get_asset_path('', 'test.png')
        path2 = GameConfig.get_asset_path('test.png', '')
        
        # Should handle gracefully
        assert isinstance(path1, str)
        assert isinstance(path2, str)
    
    def test_invalid_input_types(self):
        """Test methods with invalid input types."""
        # Test None inputs where strings expected
        try:
            color = GameConfig.get_color(None)
            # Should either return default or raise appropriate exception
        except (TypeError, AttributeError):
            pass  # Acceptable
        
        try:
            constant = GameConfig.get_movement_constant(None)
            # Should either return default or raise appropriate exception
        except (TypeError, AttributeError):
            pass  # Acceptable
    
    def test_special_characters_in_names(self):
        """Test handling of special characters in names."""
        # Test with special characters that shouldn't exist
        special_names = ['COLOR@#$', 'MOVE_!@#', '123COLOR']
        
        for name in special_names:
            color = GameConfig.get_color(name)
            assert color == GameConfig.COLORS['WHITE']  # Should return default
            
            constant = GameConfig.get_movement_constant(name)
            assert constant == GameConfig.MOVEMENT_CONSTANTS['DO_NOTHING']  # Should return default


class TestGameConfigInheritance:
    """Test GameConfig class behavior and inheritance."""
    
    def test_config_class_methods(self):
        """Test that all methods are class methods."""
        # Test that methods can be called on class
        color = GameConfig.get_color('RED')
        assert color == (255, 0, 0)
        
        path = GameConfig.get_asset_path('test')
        assert isinstance(path, str)
        
        constant = GameConfig.get_movement_constant('UP')
        assert isinstance(constant, int)
    
    def test_config_instance_isolation(self):
        """Test that config instances are isolated."""
        config1 = GameConfig()
        config2 = GameConfig()
        
        # Should be separate instances
        assert config1 is not config2
        
        # But should have same values
        assert config1.SCREEN_WIDTH == config2.SCREEN_WIDTH
        assert config1.get_color('RED') == config2.get_color('RED')
    
    def test_config_immutability(self):
        """Test that config values behave as constants."""
        original_width = GameConfig.SCREEN_WIDTH
        original_colors = GameConfig.COLORS.copy()
        
        # Attempting to modify (if possible) shouldn't affect other accesses
        # Note: Python doesn't enforce true immutability, but behavior should be consistent
        assert GameConfig.SCREEN_WIDTH == original_width
        assert GameConfig.COLORS == original_colors


@pytest.mark.integration
class TestGameConfigIntegration:
    """Integration tests for GameConfig with other components."""
    
    def test_config_with_game_initialization(self):
        """Test config integration with game initialization."""
        # Test that config values work with game setup
        width = GameConfig.SCREEN_WIDTH
        height = GameConfig.SCREEN_HEIGHT
        
        # Should be suitable for pygame display
        assert width >= 320  # Minimum reasonable width
        assert height >= 240  # Minimum reasonable height
    
    def test_config_image_paths_exist_structure(self):
        """Test that config image paths have proper structure for file system."""
        for image_name, image_path in GameConfig.IMAGES.items():
            # Path should be valid for file system
            assert '\\' not in image_path or os.sep == '\\'  # Windows compatibility
            assert '//' not in image_path  # No double slashes
            
            # Should have proper directory structure
            parts = image_path.split('/')
            assert len(parts) >= 2  # At least 'assets/filename'
    
    def test_config_player_settings_game_compatibility(self):
        """Test that player settings are compatible with game mechanics."""
        # Player speed should allow reasonable movement
        speed = GameConfig.PLAYER_SPEED
        assert 1 <= speed <= 20  # Reasonable speed range
        
        # Player size should be reasonable for collision detection
        width = GameConfig.PLAYER_WIDTH
        height = GameConfig.PLAYER_HEIGHT
        assert 10 <= width <= 50  # Reasonable size range
        assert 10 <= height <= 50
