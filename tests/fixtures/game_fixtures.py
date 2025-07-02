"""
Game-specific fixtures and test utilities for Infinite Maze testing.

This module provides fixtures specifically designed for testing
Infinite Maze game components and interactions.
"""

import pytest
import pygame
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Tuple

from infinite_maze.core.game import Game
from infinite_maze.entities.player import Player
from infinite_maze.entities.maze import Line
from infinite_maze.core.clock import Clock


class MockGameEngine:
    """Mock game engine for controlled testing."""
    
    def __init__(self):
        self.game = None
        self.player = None
        self.lines = []
        self.running = False
        self.events = []
        
    def setup_game(self, headless=True):
        """Setup a test game instance."""
        self.game = Game(headless=headless)
        self.player = Player(80, 223, headless=headless)
        self.lines = Line.generateMaze(self.game, 15, 20)
        
    def simulate_movement(self, direction: str, frames: int = 1):
        """Simulate player movement for specified frames."""
        movements = {
            'right': (1, 0),
            'left': (-1, 0),
            'up': (0, -1),
            'down': (0, 1)
        }
        
        if direction.lower() in movements:
            dx, dy = movements[direction.lower()]
            for _ in range(frames):
                if dx != 0:
                    self.player.moveX(dx)
                if dy != 0:
                    self.player.moveY(dy)
    
    def get_collision_state(self) -> Dict[str, bool]:
        """Get current collision state in all directions."""
        current_pos = self.player.getPosition()
        
        # Test movement in each direction
        collisions = {}
        directions = {
            'right': (self.player.getSpeed(), 0),
            'left': (-self.player.getSpeed(), 0),
            'up': (0, -self.player.getSpeed()),
            'down': (0, self.player.getSpeed())
        }
        
        for direction, (dx, dy) in directions.items():
            # Temporarily move player
            new_x = current_pos[0] + dx
            new_y = current_pos[1] + dy
            
            # Check for collisions
            collision = self._check_collision_at_position(new_x, new_y)
            collisions[direction] = collision
            
        return collisions
    
    def _check_collision_at_position(self, x: int, y: int) -> bool:
        """Check if position would cause collision with maze walls."""
        player_width = self.player.getWidth()
        player_height = self.player.getHeight()
        
        for line in self.lines:
            if line.getIsHorizontal():
                # Check horizontal line collision
                if (y <= line.getYStart() <= y + player_height and
                    x < line.getXEnd() and x + player_width > line.getXStart()):
                    return True
            else:
                # Check vertical line collision
                if (x <= line.getXStart() <= x + player_width and
                    y < line.getYEnd() and y + player_height > line.getYStart()):
                    return True
        
        return False


@pytest.fixture
def mock_engine():
    """Create a mock game engine for testing."""
    return MockGameEngine()


@pytest.fixture
def game_with_player(mock_pygame_display, mock_pygame_font, mock_pygame_image):
    """Create a complete game setup with player and maze."""
    game = Game(headless=True)
    player = Player(80, 223, headless=True)
    lines = Line.generateMaze(game, 15, 20)
    
    return {
        'game': game,
        'player': player,
        'lines': lines
    }


@pytest.fixture
def minimal_maze():
    """Create a minimal maze for basic testing."""
    lines = [
        # Simple box maze
        Line((100, 100), (200, 100)),  # Top
        Line((100, 200), (200, 200)),  # Bottom
        Line((100, 100), (100, 200)),  # Left
        Line((200, 100), (200, 200)),  # Right
        # Entry gap
        Line((150, 100), (150, 120)),  # Entry barrier
    ]
    return lines


@pytest.fixture
def complex_maze():
    """Create a complex maze for advanced testing."""
    lines = []
    
    # Generate a more complex maze pattern
    width, height = 10, 8
    for x in range(width):
        for y in range(height):
            base_x = 100 + x * 25
            base_y = 100 + y * 25
            
            # Add some walls pseudo-randomly
            if (x + y) % 3 == 0:  # Horizontal walls
                lines.append(Line((base_x, base_y), (base_x + 25, base_y)))
            if (x + y) % 4 == 0:  # Vertical walls
                lines.append(Line((base_x, base_y), (base_x, base_y + 25)))
    
    return lines


@pytest.fixture
def score_test_scenarios():
    """Provide test scenarios for score calculation."""
    return [
        {
            'name': 'right_movement',
            'movements': ['right'] * 5,
            'expected_score_change': 5
        },
        {
            'name': 'left_movement',
            'movements': ['left'] * 3,
            'expected_score_change': -3
        },
        {
            'name': 'vertical_movement',
            'movements': ['up', 'down', 'up', 'down'],
            'expected_score_change': 0
        },
        {
            'name': 'mixed_movement',
            'movements': ['right', 'right', 'left', 'up', 'down', 'right'],
            'expected_score_change': 2  # +2 -1 +0 +0 +1 = 2
        }
    ]


@pytest.fixture
def pace_test_scenarios():
    """Provide test scenarios for pace mechanics."""
    return [
        {
            'time_millis': 5000,   # 5 seconds
            'expected_pace': 0     # No pace yet
        },
        {
            'time_millis': 15000,  # 15 seconds
            'expected_pace': 0     # Still no pace
        },
        {
            'time_millis': 35000,  # 35 seconds
            'expected_pace': 1     # First pace increase
        },
        {
            'time_millis': 65000,  # 65 seconds
            'expected_pace': 2     # Second pace increase
        },
        {
            'time_millis': 125000, # 125 seconds
            'expected_pace': 4     # Multiple pace increases
        }
    ]


@pytest.fixture
def input_test_sequences():
    """Provide input sequences for testing."""
    return {
        'basic_movement': [
            pygame.K_RIGHT, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT
        ],
        'pause_sequence': [
            pygame.K_RIGHT, pygame.K_SPACE, pygame.K_RIGHT, pygame.K_SPACE
        ],
        'quit_sequence': [
            pygame.K_RIGHT, pygame.K_ESCAPE
        ],
        'alternative_keys': [
            pygame.K_d, pygame.K_a, pygame.K_w, pygame.K_s
        ]
    }


@pytest.fixture
def boundary_test_positions():
    """Provide boundary test positions for player movement."""
    return {
        'left_boundary': (79, 223),    # Just at left boundary
        'right_boundary': (321, 223),  # At right boundary
        'top_boundary': (160, 39),     # At top boundary
        'bottom_boundary': (160, 448), # At bottom boundary
        'safe_center': (160, 223),     # Safe center position
        'near_left_edge': (81, 223),   # Just inside left boundary
        'near_right_edge': (319, 223)  # Just inside right boundary
    }


@pytest.fixture
def collision_test_cases():
    """Provide collision test cases."""
    return [
        {
            'name': 'player_vs_horizontal_wall',
            'player_pos': (100, 95),
            'wall': Line((90, 100), (130, 100)),
            'movement': 'down',
            'should_collide': True
        },
        {
            'name': 'player_vs_vertical_wall',
            'player_pos': (95, 100),
            'wall': Line((100, 90), (100, 130)),
            'movement': 'right',
            'should_collide': True
        },
        {
            'name': 'player_clear_path',
            'player_pos': (100, 100),
            'wall': Line((150, 100), (180, 100)),
            'movement': 'right',
            'should_collide': False
        },
        {
            'name': 'player_wall_edge_case',
            'player_pos': (100, 100),
            'wall': Line((120, 100), (140, 100)),
            'movement': 'right',
            'should_collide': False  # Player width is 20, wall starts at 120
        }
    ]


@pytest.fixture
def performance_benchmarks():
    """Provide performance benchmarks for testing."""
    return {
        'frame_rate': {
            'target_fps': 60,
            'minimum_fps': 50,
            'test_duration': 2.0
        },
        'memory_usage': {
            'initial_mb': 50,
            'maximum_mb': 100,
            'growth_rate_mb_per_second': 1
        },
        'startup_time': {
            'maximum_seconds': 3.0,
            'target_seconds': 1.0
        },
        'input_latency': {
            'maximum_ms': 50,
            'target_ms': 16  # One frame at 60fps
        }
    }


class TestDataGenerator:
    """Generate test data for various testing scenarios."""
    
    @staticmethod
    def generate_random_maze(width: int, height: int, density: float = 0.3) -> List[Line]:
        """Generate a random maze with specified density."""
        import random
        lines = []
        
        for x in range(width):
            for y in range(height):
                if random.random() < density:
                    base_x = 100 + x * 22
                    base_y = 100 + y * 22
                    
                    if random.choice([True, False]):  # Horizontal or vertical
                        lines.append(Line((base_x, base_y), (base_x + 22, base_y)))
                    else:
                        lines.append(Line((base_x, base_y), (base_x, base_y + 22)))
        
        return lines
    
    @staticmethod
    def generate_movement_sequence(length: int, bias: str = 'none') -> List[str]:
        """Generate a sequence of movements with optional bias."""
        import random
        
        movements = ['right', 'left', 'up', 'down']
        
        if bias == 'right':
            weights = [0.4, 0.2, 0.2, 0.2]
        elif bias == 'forward':
            weights = [0.5, 0.1, 0.2, 0.2]
        else:
            weights = [0.25, 0.25, 0.25, 0.25]
        
        return random.choices(movements, weights=weights, k=length)
    
    @staticmethod
    def generate_stress_test_data(iterations: int) -> List[Dict[str, Any]]:
        """Generate data for stress testing."""
        import random
        
        test_data = []
        for i in range(iterations):
            test_data.append({
                'iteration': i,
                'player_start': (random.randint(80, 300), random.randint(40, 400)),
                'maze_size': (random.randint(10, 30), random.randint(10, 30)),
                'movement_sequence': TestDataGenerator.generate_movement_sequence(
                    random.randint(10, 100), 
                    random.choice(['none', 'right', 'forward'])
                ),
                'time_acceleration': random.uniform(1.0, 10.0)
            })
        
        return test_data


@pytest.fixture
def test_data_generator():
    """Provide the test data generator utility."""
    return TestDataGenerator
