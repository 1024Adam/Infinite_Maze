"""
Environment module for Infinite Maze AI.

This module contains the environments used for training and evaluating
reinforcement learning agents for the Infinite Maze game.
"""

# We can import this class directly as it doesn't create circular references
from infinite_maze.ai.environments.environment_phase1 import InfiniteMazeEnv

__all__ = ['InfiniteMazeEnv']
