"""
Common test helper functions for Infinite Maze testing.

This module provides utility functions and helpers that are commonly
used across different test modules.
"""

import time
import psutil
import os
import pygame
from typing import Dict, List, Tuple, Any, Callable, Optional
from contextlib import contextmanager
from unittest.mock import patch

from infinite_maze.core.game import Game
from infinite_maze.entities.player import Player
from infinite_maze.entities.maze import Line


def assert_within_tolerance(actual: float, expected: float, tolerance: float, 
                          message: str = ""):
    """Assert that actual value is within tolerance of expected value."""
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{message}Expected {expected} ± {tolerance}, got {actual}"
        )


def assert_position_equal(actual: Tuple[int, int], expected: Tuple[int, int], tolerance: int = 1):
    """Assert that two positions are equal within tolerance."""
    assert abs(actual[0] - expected[0]) <= tolerance, f"X position {actual[0]} not close to {expected[0]}"
    assert abs(actual[1] - expected[1]) <= tolerance, f"Y position {actual[1]} not close to {expected[1]}"


def assert_position_valid(position: Tuple[int, int], bounds: Dict[str, int]):
    """Assert that position is within valid game bounds."""
    x, y = position
    assert bounds['x_min'] <= x <= bounds['x_max'], f"X position {x} out of bounds"
    assert bounds['y_min'] <= y <= bounds['y_max'], f"Y position {y} out of bounds"


def assert_score_valid(score: int, minimum: int = 0):
    """Assert that score is valid (non-negative unless specified)."""
    assert score >= minimum, f"Score {score} below minimum {minimum}"


def calculate_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two positions."""
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def is_collision_detected(player: Player, line: Line) -> bool:
    """Check if player would collide with a specific line."""
    px, py = player.getPosition()
    pw, ph = player.getWidth(), player.getHeight()
    
    if line.getIsHorizontal():
        # Horizontal line collision
        line_y = line.getYStart()
        line_x1, line_x2 = line.getXStart(), line.getXEnd()
        
        # Check Y overlap
        if py <= line_y <= py + ph:
            # Check X overlap
            return not (px + pw <= line_x1 or px >= line_x2)
    else:
        # Vertical line collision
        line_x = line.getXStart()
        line_y1, line_y2 = line.getYStart(), line.getYEnd()
        
        # Check X overlap
        if px <= line_x <= px + pw:
            # Check Y overlap
            return not (py + ph <= line_y1 or py >= line_y2)
    
    return False


def get_valid_moves(player: Player, lines: List[Line], 
                   bounds: Dict[str, int]) -> Dict[str, bool]:
    """Get dictionary of valid moves for player in current position."""
    valid_moves = {}
    current_pos = player.getPosition()
    speed = player.getSpeed()
    
    # Test each direction
    test_positions = {
        'right': (current_pos[0] + speed, current_pos[1]),
        'left': (current_pos[0] - speed, current_pos[1]),
        'up': (current_pos[0], current_pos[1] - speed),
        'down': (current_pos[0], current_pos[1] + speed)
    }
    
    for direction, (test_x, test_y) in test_positions.items():
        # Check bounds
        in_bounds = (bounds['x_min'] <= test_x <= bounds['x_max'] and
                    bounds['y_min'] <= test_y <= bounds['y_max'])
        
        # Check collisions
        no_collision = True
        if in_bounds:
            # Temporarily move player to test position
            original_pos = player.getPosition()
            player.setX(test_x)
            player.setY(test_y)
            
            # Check for collisions
            for line in lines:
                if is_collision_detected(player, line):
                    no_collision = False
                    break
            
            # Restore original position
            player.setX(original_pos[0])
            player.setY(original_pos[1])
        
        valid_moves[direction] = in_bounds and no_collision
    
    return valid_moves


class PerformanceMonitor:
    """Monitor performance metrics during testing."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_samples = []
        self.fps_samples = []
        self.frame_times = []
        
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.memory_samples = [self._get_memory_usage()]
        
    def stop(self):
        """Stop monitoring."""
        self.end_time = time.time()
        self.memory_samples.append(self._get_memory_usage())
        
    def sample_frame(self, fps: float, frame_time_ms: float):
        """Record a frame sample."""
        self.fps_samples.append(fps)
        self.frame_times.append(frame_time_ms)
        self.memory_samples.append(self._get_memory_usage())
        
    def get_duration(self) -> float:
        """Get total monitoring duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
        
    def get_average_fps(self) -> float:
        """Get average FPS during monitoring."""
        return sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0.0
        
    def get_memory_usage_mb(self) -> Dict[str, float]:
        """Get memory usage statistics in MB."""
        if not self.memory_samples:
            return {'initial': 0, 'peak': 0, 'final': 0, 'average': 0}
            
        return {
            'initial': self.memory_samples[0],
            'peak': max(self.memory_samples),
            'final': self.memory_samples[-1],
            'average': sum(self.memory_samples) / len(self.memory_samples)
        }
        
    def get_frame_time_stats(self) -> Dict[str, float]:
        """Get frame time statistics."""
        if not self.frame_times:
            return {'min': 0, 'max': 0, 'average': 0, 'std_dev': 0}
            
        average = sum(self.frame_times) / len(self.frame_times)
        variance = sum((x - average) ** 2 for x in self.frame_times) / len(self.frame_times)
        std_dev = variance ** 0.5
        
        return {
            'min': min(self.frame_times),
            'max': max(self.frame_times),
            'average': average,
            'std_dev': std_dev
        }
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


class GameStateCapture:
    """Capture and restore game state for testing."""
    
    def __init__(self, game: Game, player: Player):
        self.game = game
        self.player = player
        self.captured_state = None
        
    def capture(self):
        """Capture current game state."""
        self.captured_state = {
            'game': {
                'score': self.game.getScore(),
                'pace': self.game.getPace(),
                'paused': self.game.isPaused(),
                'over': self.game.over if hasattr(self.game, 'over') else False,
                'shutdown': self.game.shutdown if hasattr(self.game, 'shutdown') else False
            },
            'player': {
                'position': self.player.getPosition(),
                'speed': self.player.getSpeed()
            },
            'clock': {
                'millis': self.game.getClock().getMillis() if self.game.getClock() else 0,
                'ticks': self.game.getClock().getTicks() if self.game.getClock() else 0
            }
        }
        
    def restore(self):
        """Restore previously captured game state."""
        if not self.captured_state:
            raise ValueError("No state has been captured")
            
        # Restore game state
        self.game.setScore(self.captured_state['game']['score'])
        self.game.setPace(self.captured_state['game']['pace'])
        if hasattr(self.game, 'paused'):
            self.game.paused = self.captured_state['game']['paused']
        if hasattr(self.game, 'over'):
            self.game.over = self.captured_state['game']['over']
        if hasattr(self.game, 'shutdown'):
            self.game.shutdown = self.captured_state['game']['shutdown']
            
        # Restore player state
        pos = self.captured_state['player']['position']
        self.player.setX(pos[0])
        self.player.setY(pos[1])
        
        # Note: Clock state restoration would require more complex logic
        
    def get_state_diff(self) -> Dict[str, Any]:
        """Get difference between captured state and current state."""
        if not self.captured_state:
            return {}
            
        current_state = {
            'game': {
                'score': self.game.getScore(),
                'pace': self.game.getPace(),
                'paused': self.game.isPaused()
            },
            'player': {
                'position': self.player.getPosition(),
                'speed': self.player.getSpeed()
            }
        }
        
        diff = {}
        for category in ['game', 'player']:
            diff[category] = {}
            for key in current_state[category]:
                old_val = self.captured_state[category][key]
                new_val = current_state[category][key]
                if old_val != new_val:
                    diff[category][key] = {'old': old_val, 'new': new_val}
                    
        return diff


@contextmanager
def temporary_game_state(game: Game, player: Player):
    """Context manager to temporarily modify game state."""
    capture = GameStateCapture(game, player)
    capture.capture()
    try:
        yield capture
    finally:
        capture.restore()


def run_game_simulation(game: Game, player: Player, lines: List[Line], 
                       steps: int, movement_function: Callable) -> Dict[str, Any]:
    """Run a simulation of the game for specified steps."""
    results = {
        'initial_state': {
            'score': game.getScore(),
            'position': player.getPosition(),
            'pace': game.getPace()
        },
        'movements': [],
        'collisions': [],
        'score_changes': [],
        'final_state': {}
    }
    
    for step in range(steps):
        initial_score = game.getScore()
        initial_pos = player.getPosition()
        
        # Get movement from function
        movement = movement_function(step, game, player, lines)
        results['movements'].append(movement)
        
        # Apply movement (this would need to be implemented based on game logic)
        # For now, just record the intended movement
        
        # Check for collisions
        collision = any(is_collision_detected(player, line) for line in lines)
        results['collisions'].append(collision)
        
        # Record score change
        final_score = game.getScore()
        score_change = final_score - initial_score
        results['score_changes'].append(score_change)
        
        # Update game state (basic simulation)
        if hasattr(game, 'clock') and game.clock:
            game.clock.tick()
    
    results['final_state'] = {
        'score': game.getScore(),
        'position': player.getPosition(),
        'pace': game.getPace()
    }
    
    return results


def create_test_maze_pattern(pattern_type: str, size: Tuple[int, int]) -> List[Line]:
    """Create predefined maze patterns for testing."""
    width, height = size
    lines = []
    
    if pattern_type == 'empty':
        # No walls - completely open
        pass
    elif pattern_type == 'corridor':
        # Simple horizontal corridor
        for x in range(width):
            base_x = 100 + x * 22
            # Top and bottom walls
            lines.append(Line((base_x, 100), (base_x + 22, 100)))
            lines.append(Line((base_x, 100 + height * 22), (base_x + 22, 100 + height * 22)))
    elif pattern_type == 'maze':
        # Basic maze pattern
        for x in range(width):
            for y in range(height):
                base_x = 100 + x * 22
                base_y = 100 + y * 22
                
                # Create maze pattern
                if (x + y) % 3 == 0:
                    lines.append(Line((base_x, base_y), (base_x + 22, base_y)))
                if (x + y) % 4 == 1:
                    lines.append(Line((base_x, base_y), (base_x, base_y + 22)))
    elif pattern_type == 'stress_test':
        # Dense pattern for stress testing
        for x in range(width):
            for y in range(height):
                base_x = 100 + x * 22
                base_y = 100 + y * 22
                
                # Dense wall pattern
                if (x + y) % 2 == 0:
                    lines.append(Line((base_x, base_y), (base_x + 22, base_y)))
                    lines.append(Line((base_x, base_y), (base_x, base_y + 22)))
    
    return lines


class TestMetrics:
    """Collect and analyze test metrics."""
    
    def __init__(self):
        self.test_results = []
        self.performance_data = []
        
    def record_test_result(self, test_name: str, duration: float, 
                          passed: bool, details: Dict[str, Any] = None):
        """Record the result of a test."""
        self.test_results.append({
            'name': test_name,
            'duration': duration,
            'passed': passed,
            'details': details or {}
        })
        
    def record_performance_data(self, test_name: str, metrics: Dict[str, float]):
        """Record performance metrics for a test."""
        self.performance_data.append({
            'test': test_name,
            'metrics': metrics
        })
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all test results."""
        if not self.test_results:
            return {'total': 0, 'passed': 0, 'failed': 0, 'success_rate': 0.0}
            
        total = len(self.test_results)
        passed = sum(1 for result in self.test_results if result['passed'])
        failed = total - passed
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'success_rate': passed / total,
            'average_duration': sum(r['duration'] for r in self.test_results) / total
        }
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.performance_data:
            return {}
            
        # Aggregate metrics across all tests
        all_metrics = {}
        for data in self.performance_data:
            for metric, value in data['metrics'].items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate statistics
        summary = {}
        for metric, values in all_metrics.items():
            summary[metric] = {
                'min': min(values),
                'max': max(values),
                'average': sum(values) / len(values),
                'count': len(values)
            }
            
        return summary
