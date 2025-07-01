"""
Game entities for Infinite Maze.

This package contains game objects and entities:
- player: Player character implementation
- maze: Maze generation and wall management
"""

from .player import Player
from .maze import Line

__all__ = ["Player", "Line"]
