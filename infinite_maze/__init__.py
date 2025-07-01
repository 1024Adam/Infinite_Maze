"""
Infinite Maze - A Python maze game built with Pygame.

This package provides a complete maze game where players navigate through
an infinite procedurally generated maze, trying to progress as far as possible.

Modules:
    - infinite_maze: Main game logic and entry point
    - Game: Core game state management
    - Player: Player character implementation
    - Clock: Game timing and frame rate control
    - Line: Maze generation and line utilities
    - config: Game configuration and settings
    - logger: Centralized logging system

Usage:
    To run the game:
        python -m infinite_maze
    Or import and run programmatically:
        from infinite_maze import maze
        maze()
"""

__version__ = "0.1.0"
__author__ = "Adam Reid"
__email__ = "adamjreid10@gmail.com"

# Import main game function for convenience
from .core.engine import maze

# Import core game classes
from .core.game import Game
from .entities.player import Player
from .entities.maze import Line
from .core.clock import Clock

# Import configuration for external access
from .utils.config import config, GameConfig

# Import logger for external use
from .utils.logger import logger, GameLogger

__all__ = [
    "maze",
    "Game",
    "Player",
    "Line",
    "Clock",
    "config",
    "GameConfig",
    "logger",
    "GameLogger",
    "__version__",
    "__author__",
    "__email__",
]
