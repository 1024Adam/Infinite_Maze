"""
Unit tests for the Game class in Infinite Maze.

These tests verify the functionality of the Game class including
game state management, scoring, timing, display handling, and configuration.
"""

import pytest
import pygame
from unittest.mock import Mock, patch, MagicMock

from infinite_maze.core.game import Game
from infinite_maze.core.clock import Clock
from infinite_maze.utils.config import config
from tests.fixtures.pygame_mocks import full_pygame_mocks, headless_pygame_mocks
from tests.fixtures.test_helpers import temporary_game_state


class TestGameInitialization:
    """Test Game class initialization."""
    
    def test_game_init_headless_mode(self):
        """Test game initialization in headless mode."""
        game = Game(headless=True)
        
        # Basic state checks
        assert game.get_score() == 0
        assert game.get_pace() == 0
        assert not game.is_paused()
        assert game.is_active()
        assert game.is_playing()
        
        # Headless mode should not initialize display components
        assert game.screen is None
        assert game.font is None
        assert game.icon is None
    
    def test_game_init_with_display(self):
        """Test game initialization with display mode."""
        with full_pygame_mocks() as mocks:
            game = Game(headless=False)
            
            # Verify pygame initialization calls
            assert mocks['display']['set_mode'].called
            assert mocks['display']['set_caption'].called
            assert mocks['display']['set_icon'].called
            
            # Game should be properly initialized
            assert game.get_score() == 0
            assert game.get_pace() == 0
            assert not game.is_paused()

class TestGameScoring:
    """Test Game scoring functionality."""
    
    def test_initial_score(self):
        """Test initial score is zero."""
        game = Game(headless=True)
        assert game.get_score() == 0
    
    def test_increment_score(self):
        """Test score increment functionality."""
        game = Game(headless=True)
        initial_score = game.get_score()
        
        game.increment_score()
        
        assert game.get_score() == initial_score + config.SCORE_INCREMENT
    
    def test_decrement_score(self):
        """Test score decrement functionality."""
        game = Game(headless=True)
        
        # First increment to have a positive score
        game.increment_score()
        game.increment_score()
        current_score = game.get_score()
        
        game.decrement_score()
        
        assert game.get_score() == current_score - config.SCORE_INCREMENT
    
    def test_decrement_score_at_zero(self):
        """Test that score cannot go below zero."""
        game = Game(headless=True)
        
        # Try to decrement from zero
        game.decrement_score()
        
        assert game.get_score() == 0
    
    def test_decrement_score_minimum_enforcement(self):
        """Test minimum score enforcement with repeated decrements."""
        game = Game(headless=True)
        
        # Increment once, then decrement multiple times
        game.increment_score()
        game.decrement_score()
        game.decrement_score()
        game.decrement_score()
        
        assert game.get_score() == 0
    
    def test_update_score(self):
        """Test update_score method with positive and negative values."""
        game = Game(headless=True)
        
        # Test positive update
        game.update_score(5)
        assert game.get_score() == 5
        
        # Test negative update
        game.update_score(-2)
        assert game.get_score() == 3
        
        # Test zero update
        game.update_score(0)
        assert game.get_score() == 3
    
    def test_set_score(self):
        """Test set_score method."""
        game = Game(headless=True)
        
        game.set_score(42)
        assert game.get_score() == 42
        
        game.set_score(0)
        assert game.get_score() == 0
        
        # Test setting negative score (should be allowed by set_score)
        game.set_score(-10)
        assert game.get_score() == -10
    
    def test_score_sequence(self):
        """Test a sequence of score operations."""
        game = Game(headless=True)
        
        operations = [
            ('increment', 1),
            ('increment', 2),
            ('increment', 3),
            ('decrement', 2),
            ('update', 5, 7),  # update_score(5) should result in 7
            ('set', 10, 10)    # set_score(10) should result in 10
        ]
        
        for operation in operations:
            if operation[0] == 'increment':
                game.increment_score()
                assert game.get_score() == operation[1]
            elif operation[0] == 'decrement':
                game.decrement_score()
                assert game.get_score() == operation[1]
            elif operation[0] == 'update':
                game.update_score(operation[1])
                assert game.get_score() == operation[2]
            elif operation[0] == 'set':
                game.set_score(operation[1])
                assert game.get_score() == operation[2]


class TestGamePace:
    """Test Game pace functionality."""
    
    def test_initial_pace(self):
        """Test initial pace is zero."""
        game = Game(headless=True)
        assert game.get_pace() == 0
    
    def test_set_pace(self):
        """Test setting pace value."""
        game = Game(headless=True)
        
        game.set_pace(3)
        assert game.get_pace() == 3
        
        game.set_pace(0)
        assert game.get_pace() == 0
        
        game.set_pace(10)
        assert game.get_pace() == 10
    
    def test_pace_progression(self):
        """Test pace progression through game time."""
        game = Game(headless=True)
        
        # Simulate pace increases
        for expected_pace in range(1, 6):
            game.set_pace(expected_pace)
            assert game.get_pace() == expected_pace


class TestGameState:
    """Test Game state management."""
    
    def test_initial_state(self):
        """Test initial game state."""
        game = Game(headless=True)
        
        assert game.is_active()
        assert game.is_playing()
        assert not game.is_paused()
    
    def test_pause_functionality(self):
        """Test pause/unpause functionality."""
        with full_pygame_mocks():
            game = Game(headless=True)
            # Create a mock player for pause functionality
            mock_player = Mock()
            mock_player.set_cursor = Mock()
            
            # Initially not paused
            assert not game.is_paused()
            
            # Pause the game
            game.change_paused(mock_player)
            assert game.is_paused()
            
            # Unpause the game
            game.change_paused(mock_player)
            assert not game.is_paused()
    
    def test_end_game(self):
        """Test ending the game."""
        game = Game(headless=True)
        
        assert game.is_active()
        
        game.end()
        
        assert not game.is_active()
    
    def test_quit_game(self):
        """Test quitting the game."""
        game = Game(headless=True)
        
        assert game.is_playing()
        
        game.quit()
        
        assert not game.is_playing()
    
    def test_reset_game(self):
        """Test resetting the game."""
        game = Game(headless=True)
        
        # Modify game state
        game.set_score(50)
        game.set_pace(5)
        game.end()
        
        # Reset
        game.reset()
        
        # Check state is reset
        assert game.get_score() == 0
        assert game.get_pace() == 0
        assert game.is_active()


class TestGameClock:
    """Test Game clock integration."""
    
    def test_get_clock(self):
        """Test getting the game clock."""
        game = Game(headless=True)
        
        clock = game.get_clock()
        
        assert clock is not None
        assert isinstance(clock, Clock)
    
    def test_clock_functionality(self):
        """Test clock integration with game."""
        game = Game(headless=True)
        clock = game.get_clock()
        
        # Clock should be initialized
        assert clock.get_millis() >= 0
        assert clock.get_ticks() >= 0


class TestGameDisplay:
    """Test Game display functionality."""
    
    def test_get_screen_headless(self):
        """Test getting screen in headless mode."""
        game = Game(headless=True)
        
        screen = game.get_screen()
        assert screen is None
    
    def test_get_screen_with_display(self):
        """Test getting screen with display mode."""
        with full_pygame_mocks() as mocks:
            game = Game(headless=False)
            
            screen = game.get_screen()
            assert screen is not None
    
    def test_update_screen_headless(self):
        """Test screen update in headless mode."""
        game = Game(headless=True)
        mock_player = Mock()
        mock_lines = []
        
        # Should not raise exception in headless mode
        game.update_screen(mock_player, mock_lines)
    
    def test_update_screen_with_display(self):
        """Test screen update with display mode."""
        with full_pygame_mocks() as mocks:
            game = Game(headless=False)
            
            mock_player = Mock()
            mock_player.get_cursor.return_value = mocks['image']['loaded_surface']
            mock_player.get_position.return_value = (100, 100)
            
            mock_lines = []
            
            # Should call display functions
            game.update_screen(mock_player, mock_lines)
            
            # Verify screen operations were called
            assert mocks['display']['surface'].fill.called
            assert mocks['display']['surface'].blit.called
    
    def test_print_end_display(self):
        """Test end game display."""
        with full_pygame_mocks() as mocks:
            game = Game(headless=False)
            
            # Should not raise exception
            game.print_end_display()
            
            # Verify display operations
            assert mocks['display']['surface'].fill.called
            assert mocks['display']['flip'].called


class TestGameCleanup:
    """Test Game cleanup functionality."""
    
    def test_cleanup_headless(self):
        """Test cleanup in headless mode."""
        game = Game(headless=True)
        
        # Should not raise exception
        game.cleanup()
    
    def test_cleanup_with_display(self):
        """Test cleanup with display mode."""
        with patch('pygame.quit') as mock_quit:
            game = Game(headless=False)
            
            game.cleanup()
            
            # Should call pygame.quit
            mock_quit.assert_called_once()


class TestGamePauseIntegration:
    """Test Game pause integration with player."""
    
    def test_pause_changes_player_sprite(self):
        """Test that pausing changes player sprite."""
        with full_pygame_mocks():
            game = Game(headless=True)
            mock_player = Mock()
            
            # Pause the game
            game.change_paused(mock_player)
            
            # Should call set_cursor on player
            mock_player.set_cursor.assert_called()
            
            # Unpause
            game.change_paused(mock_player)
            
            # Should call set_cursor again
            assert mock_player.set_cursor.call_count == 2
    
    def test_pause_affects_color(self):
        """Test that pausing affects foreground color."""
        game = Game(headless=True)
        mock_player = Mock()
        
        original_color = game.FG_COLOR
        
        # Pause
        game.change_paused(mock_player)
        paused_color = game.FG_COLOR
        
        # Unpause
        game.change_paused(mock_player)
        unpaused_color = game.FG_COLOR
        
        # Colors should be different when paused
        assert paused_color != original_color
        assert unpaused_color == original_color

class TestGameEdgeCases:
    """Test Game edge cases and error conditions."""
    
    def test_multiple_initializations(self):
        """Test multiple game initializations."""
        # Should be able to create multiple game instances
        game1 = Game(headless=True)
        game2 = Game(headless=True)
        
        assert game1.get_score() == 0
        assert game2.get_score() == 0
        
        # Modifying one shouldn't affect the other
        game1.set_score(10)
        assert game2.get_score() == 0
    
    def test_extreme_score_values(self):
        """Test game with extreme score values."""
        game = Game(headless=True)
        
        # Test very large positive score
        large_score = 999999
        game.set_score(large_score)
        assert game.get_score() == large_score
        
        # Test very large negative score
        negative_score = -999999
        game.set_score(negative_score)
        assert game.get_score() == negative_score
    
    def test_rapid_state_changes(self):
        """Test rapid state changes."""
        game = Game(headless=True)
        mock_player = Mock()
        
        # Rapid pause/unpause
        for _ in range(100):
            game.change_paused(mock_player)
        
        # Should end up unpaused (started unpaused, 100 toggles = even)
        assert not game.is_paused()
    
    def test_reset_after_end(self):
        """Test reset after game has ended."""
        game = Game(headless=True)
        
        # Modify state and end game
        game.set_score(50)
        game.set_pace(3)
        game.end()
        
        assert not game.is_active()
        
        # Reset should restore active state
        game.reset()
        
        assert game.is_active()
        assert game.get_score() == 0
        assert game.get_pace() == 0


@pytest.mark.integration
class TestGameIntegration:
    """Integration tests for Game with other components."""
    
    def test_game_with_real_clock(self):
        """Test game integration with real clock."""
        game = Game(headless=True)
        clock = game.get_clock()
        
        # Clock should be functional
        initial_millis = clock.get_millis()
        clock.update()
        updated_millis = clock.get_millis()
        
        # Time should have progressed
        assert updated_millis >= initial_millis
    
    def test_game_state_persistence(self):
        """Test that game state persists correctly."""
        game = Game(headless=True)
        
        # Set up some state
        game.set_score(25)
        game.set_pace(2)
        
        # State should persist
        assert game.get_score() == 25
        assert game.get_pace() == 2
        
        # After operations, state should be maintained
        game.increment_score()
        assert game.get_score() == 26
        assert game.get_pace() == 2  # Should not change


@pytest.mark.performance
class TestGamePerformance:
    """Performance tests for Game operations."""
    
    def test_score_operations_performance(self):
        """Test performance of score operations."""
        game = Game(headless=True)
        
        import time
        start_time = time.time()
        
        # Perform many score operations
        for i in range(10000):
            if i % 3 == 0:
                game.increment_score()
            elif i % 3 == 1:
                game.decrement_score()
            else:
                game.update_score(1)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly
        assert duration < 1.0, f"Score operations took too long: {duration}s"
    
    def test_state_change_performance(self):
        """Test performance of state changes."""
        game = Game(headless=True)
        mock_player = Mock()
        
        import time
        start_time = time.time()
        
        # Perform many state changes
        for _ in range(1000):
            game.change_paused(mock_player)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly
        assert duration < 1.0, f"State changes took too long: {duration}s"
