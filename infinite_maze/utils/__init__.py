"""
Utility modules for Infinite Maze.

This package contains supporting utilities:
- config: Game configuration and settings
- logger: Logging and debugging utilities
"""

from .config import config, GameConfig
from .logger import logger, GameLogger

__all__ = ["config", "GameConfig", "logger", "GameLogger"]
