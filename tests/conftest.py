"""
Pytest configuration and shared fixtures for Infinite Maze tests.

This module provides common fixtures, configuration, and utilities
for testing the Infinite Maze game components.
"""

import pytest
import pygame
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Generator, Tuple, Any

# Add the infinite_maze package to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from infinite_maze.core.game import Game
from infinite_maze.entities.player import Player
from infinite_maze.entities.maze import Line
from infinite_maze.core.clock import Clock
from infinite_maze.utils.config import GameConfig


@pytest.fixture(scope="session", autouse=True)
def pygame_init():
    """Initialize pygame for all tests and clean up after session."""
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture
def mock_pygame_display():
    """Mock pygame display to avoid creating actual windows during tests."""
    with patch('pygame.display.set_mode') as mock_set_mode, \
         patch('pygame.display.set_caption') as mock_set_caption, \
         patch('pygame.display.set_icon') as mock_set_icon, \
         patch('pygame.display.flip') as mock_flip:
        
        # Create a mock surface
        mock_surface = Mock()
        mock_surface.fill = Mock()
        mock_surface.blit = Mock()
        mock_surface.get_size.return_value = (800, 600)
        mock_set_mode.return_value = mock_surface
        
        yield {
            'set_mode': mock_set_mode,
            'set_caption': mock_set_caption,
            'set_icon': mock_set_icon,
            'flip': mock_flip,
            'surface': mock_surface
        }


@pytest.fixture
def mock_pygame_image():
    """Mock pygame image loading to avoid file dependencies."""
    with patch('pygame.image.load') as mock_load:
        mock_surface = Mock()
        mock_surface.get_size.return_value = (20, 20)
        mock_load.return_value = mock_surface
        yield mock_load


@pytest.fixture
def mock_pygame_font():
    """Mock pygame font for text rendering tests."""
    with patch('pygame.font.SysFont') as mock_font:
        mock_font_obj = Mock()
        mock_font_obj.render.return_value = Mock()
        mock_font.return_value = mock_font_obj
        yield mock_font


@pytest.fixture
def mock_pygame_time():
    """Mock pygame time functions for controlled timing tests."""
    with patch('pygame.time.Clock') as mock_clock_class, \
         patch('pygame.time.delay') as mock_delay:
        
        mock_clock = Mock()
        mock_clock.tick.return_value = 16  # 60 FPS
        mock_clock.get_fps.return_value = 60.0
        mock_clock.get_time.return_value = 1000
        mock_clock_class.return_value = mock_clock
        
        yield {
            'Clock': mock_clock_class,
            'clock_instance': mock_clock,
            'delay': mock_delay
        }


@pytest.fixture
def headless_game(mock_pygame_display, mock_pygame_font, mock_pygame_image):
    """Create a game instance in headless mode for testing."""
    game = Game(headless=True)
    return game


@pytest.fixture
def test_player():
    """Create a player instance for testing."""
    return Player(80, 223, headless=True)


@pytest.fixture
def test_clock():
    """Create a clock instance for testing."""
    return Clock()


@pytest.fixture
def sample_maze_lines():
    """Create sample maze lines for testing."""
    lines = []
    # Create a simple test maze pattern
    for i in range(5):
        # Horizontal lines
        lines.append(Line((100 + i * 22, 100), (122 + i * 22, 100), i, i + 1))
        # Vertical lines
        lines.append(Line((100 + i * 22, 100), (100 + i * 22, 122), i + 5, i + 6))
    return lines


@pytest.fixture
def pygame_events():
    """Factory for creating pygame events for testing."""
    def create_event(event_type, **kwargs):
        return pygame.event.Event(event_type, **kwargs)
    return create_event


@pytest.fixture
def key_press_events(pygame_events):
    """Factory for creating key press events."""
    def create_key_event(key, pressed=True):
        event_type = pygame.KEYDOWN if pressed else pygame.KEYUP
        return pygame_events(event_type, key=key)
    return create_key_event


@pytest.fixture
def mock_game_state():
    """Create a mock game state for testing."""
    return {
        'score': 0,
        'pace': 0,
        'paused': False,
        'over': False,
        'shutdown': False,
        'player_x': 80,
        'player_y': 223,
        'maze_lines': [],
        'time_millis': 0
    }


@pytest.fixture
def collision_test_setup():
    """Setup for collision detection tests."""
    player = Player(100, 100, headless=True)
    
    # Create walls around the player
    walls = [
        Line((95, 95), (125, 95)),    # Top wall
        Line((95, 125), (125, 125)),  # Bottom wall
        Line((95, 95), (95, 125)),    # Left wall
        Line((125, 95), (125, 125))   # Right wall
    ]
    
    return {
        'player': player,
        'walls': walls,
        'center_position': (100, 100)
    }


@pytest.fixture
def performance_test_data():
    """Generate test data for performance testing."""
    return {
        'large_maze_size': (50, 100),
        'stress_test_duration': 1.0,  # seconds
        'target_fps': 60,
        'max_memory_mb': 100,
        'test_iterations': 1000
    }


# Test utilities
def assert_position_equal(actual: Tuple[int, int], expected: Tuple[int, int], tolerance: int = 1):
    """Assert that two positions are equal within tolerance."""
    assert abs(actual[0] - expected[0]) <= tolerance, f"X position {actual[0]} not close to {expected[0]}"
    assert abs(actual[1] - expected[1]) <= tolerance, f"Y position {actual[1]} not close to {expected[1]}"


def assert_score_change(initial_score: int, final_score: int, expected_change: int):
    """Assert that score changed by expected amount."""
    actual_change = final_score - initial_score
    assert actual_change == expected_change, f"Score changed by {actual_change}, expected {expected_change}"


def simulate_game_time(game: Game, milliseconds: int):
    """Simulate the passage of game time."""
    if hasattr(game, 'clock') and game.clock:
        original_millis = game.clock.getMillis()
        game.clock.millis = original_millis + milliseconds


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "functional: mark test as a functional test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "collision: mark test as collision detection test")
    config.addinivalue_line("markers", "input: mark test as input handling test")
    config.addinivalue_line("markers", "rendering: mark test as rendering test")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip slow-running tests"
    )
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="Run performance tests (skipped by default)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "functional" in str(item.fspath):
            item.add_marker(pytest.mark.functional)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
    
    # Skip slow tests if --skip-slow is specified
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="Skipping slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Skip performance tests unless explicitly requested
    if not config.getoption("--run-performance"):
        skip_performance = pytest.mark.skip(reason="Performance tests skipped by default. Use --run-performance to enable.")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_performance)
