"""
Compatibility layer for Infinite Maze AI environments.

This module re-exports the InfiniteMazeEnv class from phase1_env.py for backward
compatibility with existing code. New code should directly import from phase1_env
or phase2_env as appropriate.
"""

# Re-export the InfiniteMazeEnv class for backward compatibility
from infinite_maze.ai.phase1_env import InfiniteMazeEnv

# Also import the Phase2MazeEnv for convenience
from infinite_maze.ai.phase2_env import Phase2MazeEnv
    

