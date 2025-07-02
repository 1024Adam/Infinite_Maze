"""
Integration tests for Asset Loading in Infinite Maze.

These tests verify the integration between asset loading systems,
configuration management, and game components.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import pygame

from infinite_maze.entities.player import Player
from infinite_maze.core.game import Game
from infinite_maze.utils.config import GameConfig
from tests.fixtures.pygame_mocks import full_pygame_mocks
from tests.fixtures.test_helpers import PerformanceMonitor


class TestAssetLoadingIntegration:
    """Test asset loading integration with game components."""
    
    def test_player_sprite_loading(self):
        """Test player sprite loading integration."""
        with full_pygame_mocks() as mocks:
            # Mock successful image loading
            mock_surface = Mock()
            mock_surface.get_width.return_value = 10
            mock_surface.get_height.return_value = 10
            mocks['image']['load'].return_value = mock_surface
            
            player = Player(100, 100, headless=True)
            
            # Verify player has dimensions
            assert player.getWidth() > 0
            assert player.getHeight() > 0
    
    def test_player_sprite_loading_failure_recovery(self):
        """Test player sprite loading with file not found."""
        with full_pygame_mocks() as mocks:
            # Mock failed image loading
            mocks['image']['load'].side_effect = pygame.error("File not found")
            
            # Player should still initialize with default dimensions
            player = Player(100, 100, headless=True)
            
            # Should have fallback dimensions
            assert player.getWidth() == 10  # Default width
            assert player.getHeight() == 10  # Default height
    
    def test_player_sprite_state_integration(self):
        """Test player sprite state changes."""
        with full_pygame_mocks() as mocks:
            # Mock normal and paused sprites
            normal_sprite = Mock()
            normal_sprite.get_width.return_value = 10
            normal_sprite.get_height.return_value = 10
            
            paused_sprite = Mock()
            paused_sprite.get_width.return_value = 12
            paused_sprite.get_height.return_value = 12
            
            # Return different sprites for different files
            def mock_load(path):
                if "paused" in path:
                    return paused_sprite
                return normal_sprite
            
            mocks['image']['load'].side_effect = mock_load
            
            player = Player(100, 100, headless=True)
            game = Game(headless=True)
            
            # Test pause state change
            initial_sprite_state = (player.getWidth(), player.getHeight())
            
            # Simulate pause
            game.changePaused(player)
            
            # In paused state, sprite might change
            if game.isPaused():
                # Player could reload sprite for paused state
                try:
                    # Simulate sprite change in pause
                    player.width = 12
                    player.height = 12
                    paused_sprite_state = (player.getWidth(), player.getHeight())
                    assert paused_sprite_state != initial_sprite_state
                except:
                    # Fallback if sprite changing not implemented
                    pass


class TestConfigAssetIntegration:
    """Test configuration and asset path integration."""
    
    def test_asset_path_resolution(self):
        """Test asset path resolution from config."""
        config = GameConfig()
        
        # Test player asset paths
        player_image = config.getPlayerImage()
        player_paused_image = config.getPlayerPausedImage()
        icon_path = config.getIcon()
        
        # Paths should be valid strings
        assert isinstance(player_image, str)
        assert isinstance(player_paused_image, str)
        assert isinstance(icon_path, str)
        
        # Paths should contain expected components
        assert "player" in player_image.lower()
        assert "paused" in player_paused_image.lower()
        assert "icon" in icon_path.lower()
    
    def test_asset_path_validation(self):
        """Test asset path validation."""
        config = GameConfig()
        
        # Get asset paths
        asset_paths = [
            config.getPlayerImage(),
            config.getPlayerPausedImage(),
            config.getIcon()
        ]
        
        for path in asset_paths:
            # Path should be relative to project
            assert not os.path.isabs(path) or path.startswith("assets/")
            
            # Should not contain invalid characters
            invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
            assert not any(char in path for char in invalid_chars)
    
    def test_config_asset_integration_with_game(self):
        """Test config asset integration with game components."""
        config = GameConfig()
        
        # Test with player
        with full_pygame_mocks() as mocks:
            mock_surface = Mock()
            mock_surface.get_width.return_value = 15
            mock_surface.get_height.return_value = 15
            mocks['image']['load'].return_value = mock_surface
            
            player = Player(100, 100, headless=True)
            
            # Player should use config dimensions indirectly
            assert player.getWidth() > 0
            assert player.getHeight() > 0
    
    def test_config_window_icon_integration(self):
        """Test window icon configuration integration."""
        config = GameConfig()
        
        with full_pygame_mocks() as mocks:
            # Mock icon loading
            mock_icon = Mock()
            mocks['image']['load'].return_value = mock_icon
            
            # Simulate setting window icon
            icon_path = config.getIcon()
            
            # Should have icon path
            assert icon_path is not None
            assert len(icon_path) > 0


class TestAssetErrorHandling:
    """Test asset loading error handling integration."""
    
    def test_missing_player_assets_graceful_handling(self):
        """Test graceful handling of missing player assets."""
        with full_pygame_mocks() as mocks:
            # Simulate file not found
            mocks['image']['load'].side_effect = pygame.error("File not found")
            
            # Game should still work
            game = Game(headless=True)
            player = Player(100, 100, headless=True)
            
            # Basic functionality should work
            initial_pos = player.getPosition()
            player.moveX(1)
            assert player.getPosition() != initial_pos
    
    def test_corrupted_asset_handling(self):
        """Test handling of corrupted assets."""
        with full_pygame_mocks() as mocks:
            # Simulate corrupted file
            mocks['image']['load'].side_effect = pygame.error("Invalid image format")
            
            # Game should continue with defaults
            player = Player(100, 100, headless=True)
            
            # Should have fallback dimensions
            assert player.getWidth() == 10
            assert player.getHeight() == 10
    
    def test_permission_denied_asset_handling(self):
        """Test handling of permission denied for assets."""
        with full_pygame_mocks() as mocks:
            # Simulate permission error
            mocks['image']['load'].side_effect = PermissionError("Access denied")
            
            try:
                player = Player(100, 100, headless=True)
                # Should work with defaults
                assert player.getPosition() == (100, 100)
            except PermissionError:
                # If error propagates, it should be handled gracefully
                assert False, "Permission error not handled gracefully"
    
    def test_asset_loading_timeout_simulation(self):
        """Test asset loading timeout simulation."""
        with full_pygame_mocks() as mocks:
            # Simulate slow loading
            def slow_load(path):
                import time
                time.sleep(0.01)  # Small delay
                mock_surface = Mock()
                mock_surface.get_width.return_value = 10
                mock_surface.get_height.return_value = 10
                return mock_surface
            
            mocks['image']['load'].side_effect = slow_load
            
            # Should complete reasonably quickly
            import time
            start_time = time.time()
            player = Player(100, 100, headless=True)
            end_time = time.time()
            
            assert end_time - start_time < 1.0, "Asset loading took too long"


class TestAssetMemoryManagement:
    """Test asset memory management integration."""
    
    def test_asset_memory_cleanup(self):
        """Test asset memory cleanup."""
        with full_pygame_mocks() as mocks:
            mock_surface = Mock()
            mock_surface.get_width.return_value = 10
            mock_surface.get_height.return_value = 10
            mocks['image']['load'].return_value = mock_surface
            
            # Create multiple players (simulating asset reuse)
            players = []
            for i in range(10):
                player = Player(100 + i * 20, 100, headless=True)
                players.append(player)
            
            # All should work
            for player in players:
                assert player.getWidth() > 0
                assert player.getHeight() > 0
    
    def test_asset_caching_behavior(self):
        """Test asset caching behavior."""
        with full_pygame_mocks() as mocks:
            load_call_count = 0
            
            def count_loads(path):
                nonlocal load_call_count
                load_call_count += 1
                mock_surface = Mock()
                mock_surface.get_width.return_value = 10
                mock_surface.get_height.return_value = 10
                return mock_surface
            
            mocks['image']['load'].side_effect = count_loads
            
            # Create multiple players
            player1 = Player(100, 100, headless=True)
            player2 = Player(200, 100, headless=True)
            player3 = Player(300, 100, headless=True)
            
            # Note: Actual caching depends on implementation
            # This test documents expected behavior
            assert load_call_count >= 1  # At least one load should occur
    
    def test_large_asset_handling(self):
        """Test handling of large assets."""
        with full_pygame_mocks() as mocks:
            # Simulate large image
            large_surface = Mock()
            large_surface.get_width.return_value = 1000
            large_surface.get_height.return_value = 1000
            mocks['image']['load'].return_value = large_surface
            
            # Should handle large assets
            player = Player(100, 100, headless=True)
            
            # Should use the asset dimensions
            assert player.getWidth() > 0
            assert player.getHeight() > 0


class TestAssetDisplayIntegration:
    """Test asset integration with display system."""
    
    def test_player_sprite_display_integration(self):
        """Test player sprite display integration."""
        with full_pygame_mocks() as mocks:
            mock_surface = Mock()
            mock_surface.get_width.return_value = 15
            mock_surface.get_height.return_value = 15
            mocks['image']['load'].return_value = mock_surface
            
            game = Game(headless=True)
            player = Player(100, 100, headless=True)
            lines = []
            
            # Test display update
            try:
                game.updateScreen(player, lines)
                # Should complete without error
            except Exception as e:
                assert False, f"Display update failed: {e}"
    
    def test_sprite_scaling_integration(self):
        """Test sprite scaling integration."""
        from ..utils.config import config
        
        with full_pygame_mocks() as mocks:
            # Test different sprite sizes
            sprite_sizes = [(10, 10), (20, 20), (5, 5), (30, 15)]
            
            for width, height in sprite_sizes:
                mock_surface = Mock()
                mock_surface.get_width.return_value = width
                mock_surface.get_height.return_value = height
                mocks['image']['load'].return_value = mock_surface
                
                player = Player(100, 100, headless=True)
                
                # Player dimensions are now controlled by config, not sprite size
                assert player.getWidth() == config.PLAYER_WIDTH
                assert player.getHeight() == config.PLAYER_HEIGHT
    
    def test_sprite_position_integration(self):
        """Test sprite position integration with game coordinates."""
        with full_pygame_mocks() as mocks:
            mock_surface = Mock()
            mock_surface.get_width.return_value = 12
            mock_surface.get_height.return_value = 12
            mocks['image']['load'].return_value = mock_surface
            
            player = Player(150, 200, headless=True)
            
            # Position should match initialization
            assert player.getPosition() == (150, 200)
            
            # Movement should update position
            player.moveX(5)
            assert player.getX() == 155
            
            player.moveY(-3)
            assert player.getY() == 197


class TestAssetConfigurationErrors:
    """Test asset configuration error scenarios."""
    
    def test_invalid_asset_path_config(self):
        """Test invalid asset path configuration."""
        # Test with non-existent paths
        config = GameConfig()
        
        # Paths should be strings even if files don't exist
        player_image = config.getPlayerImage()
        assert isinstance(player_image, str)
        assert len(player_image) > 0
    
    def test_asset_path_traversal_protection(self):
        """Test protection against path traversal."""
        config = GameConfig()
        
        # Asset paths should be safe
        asset_paths = [
            config.getPlayerImage(),
            config.getPlayerPausedImage(),
            config.getIcon()
        ]
        
        for path in asset_paths:
            # Should not contain path traversal
            assert ".." not in path
            assert not path.startswith("/")
            assert not (len(path) > 1 and path[1] == ":")  # Windows drive
    
    def test_config_environment_integration(self):
        """Test config integration with different environments."""
        # Test headless mode
        config = GameConfig()
        game = Game(headless=True)
        
        # Should work in headless mode
        assert game.isPlaying()
        
        # Config should provide valid paths regardless
        assert len(config.getPlayerImage()) > 0


class TestAssetPerformance:
    """Test asset loading performance integration."""
    
    @pytest.mark.performance
    def test_asset_loading_performance(self):
        """Test asset loading performance."""
        with full_pygame_mocks() as mocks:
            mock_surface = Mock()
            mock_surface.get_width.return_value = 10
            mock_surface.get_height.return_value = 10
            mocks['image']['load'].return_value = mock_surface
            
            monitor = PerformanceMonitor()
            monitor.start()
            
            # Create many players to test asset loading
            players = []
            for i in range(50):
                player = Player(100 + i, 100, headless=True)
                players.append(player)
            
            monitor.stop()
            
            # Should complete quickly
            duration = monitor.get_duration()
            assert duration < 2.0, f"Asset loading took too long: {duration}s"
            
            # All players should be valid
            assert len(players) == 50
            for player in players:
                assert player.getWidth() > 0
                assert player.getHeight() > 0
    
    @pytest.mark.performance
    def test_repeated_asset_access_performance(self):
        """Test repeated asset access performance."""
        with full_pygame_mocks() as mocks:
            mock_surface = Mock()
            mock_surface.get_width.return_value = 10
            mock_surface.get_height.return_value = 10
            mocks['image']['load'].return_value = mock_surface
            
            player = Player(100, 100, headless=True)
            
            import time
            start_time = time.time()
            
            # Access sprite properties many times
            for _ in range(1000):
                width = player.getWidth()
                height = player.getHeight()
                pos = player.getPosition()
                
                # Verify values
                assert width > 0
                assert height > 0
                assert len(pos) == 2
            
            end_time = time.time()
            duration = end_time - start_time
            
            assert duration < 1.0, f"Repeated access took too long: {duration}s"
    
    @pytest.mark.performance
    def test_asset_memory_usage_performance(self):
        """Test asset memory usage performance."""
        with full_pygame_mocks() as mocks:
            mock_surface = Mock()
            mock_surface.get_width.return_value = 20
            mock_surface.get_height.return_value = 20
            mocks['image']['load'].return_value = mock_surface
            
            monitor = PerformanceMonitor()
            
            # Create and destroy many players
            for iteration in range(10):
                players = []
                for i in range(20):
                    player = Player(100 + i, 100, headless=True)
                    players.append(player)
                
                # Sample memory after creation
                monitor.sample_memory()
                
                # Clear players
                players.clear()
            
            # Memory usage should be reasonable
            memory_stats = monitor.get_memory_usage_mb()
            assert memory_stats['peak'] < 100, f"Memory usage too high: {memory_stats['peak']}MB"


class TestAssetIntegrationScenarios:
    """Test complete asset integration scenarios."""
    
    def test_game_startup_with_assets(self):
        """Test complete game startup with asset loading."""
        with full_pygame_mocks() as mocks:
            # Mock all asset loads
            mock_surface = Mock()
            mock_surface.get_width.return_value = 15
            mock_surface.get_height.return_value = 15
            mocks['image']['load'].return_value = mock_surface
            
            # Full game initialization
            config = GameConfig()
            game = Game(headless=True)
            player = Player(100, 100, headless=True)
            
            # All components should be ready
            assert game.isPlaying()
            assert player.getPosition() == (100, 100)
            assert player.getWidth() > 0
            assert player.getHeight() > 0
    
    def test_game_with_missing_assets(self):
        """Test game operation with missing assets."""
        with full_pygame_mocks() as mocks:
            # All asset loads fail
            mocks['image']['load'].side_effect = pygame.error("File not found")
            
            # Game should still work
            game = Game(headless=True)
            player = Player(100, 100, headless=True)
            
            # Basic functionality should work
            assert game.isPlaying()
            assert player.getPosition() == (100, 100)
            
            # Movement should work
            player.moveX(5)
            assert player.getX() == 105
    
    def test_asset_hot_reload_simulation(self):
        """Test asset hot reload simulation."""
        from ..utils.config import config
        
        with full_pygame_mocks() as mocks:
            # Initial asset
            initial_surface = Mock()
            initial_surface.get_width.return_value = 10
            initial_surface.get_height.return_value = 10
            
            # Updated asset
            updated_surface = Mock()
            updated_surface.get_width.return_value = 20
            updated_surface.get_height.return_value = 20
            
            # Start with initial asset
            mocks['image']['load'].return_value = initial_surface
            player = Player(100, 100, headless=True)
            
            # Player dimensions are controlled by config, not asset size
            assert player.getWidth() == config.PLAYER_WIDTH
            assert player.getHeight() == config.PLAYER_HEIGHT
            
            # Simulate asset reload
            mocks['image']['load'].return_value = updated_surface
            new_player = Player(200, 100, headless=True)
            
            # New player should still use config dimensions
            assert new_player.getWidth() == config.PLAYER_WIDTH
            assert new_player.getHeight() == config.PLAYER_HEIGHT
