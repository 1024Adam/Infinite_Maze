"""
Core game systems for Infinite Maze.

This package contains the fundamental game engine components:
- engine: Main game loop and coordination
- game: Game state management
- clock: Timing and frame rate control
"""

from .engine import maze
from .game import Game
from .clock import Clock

__all__ = ["maze", "Game", "Clock"]
