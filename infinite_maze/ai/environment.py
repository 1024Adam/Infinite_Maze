"""
Compatibility layer for Infinite Maze AI environments.

This module re-exports the InfiniteMazeEnv class from environments/environment_phase1.py for backward
compatibility with existing code. New code should directly import from environments/environment_phase1
or environments/environment_phase2 as appropriate.
"""

# Re-export the InfiniteMazeEnv class for backward compatibility
from infinite_maze.ai.environments.environment_phase1 import InfiniteMazeEnv

# NOTE: Phase2MazeEnv currently does not exist, uncomment when it is implemented
# from infinite_maze.ai.environments.environment_phase2 import Phase2MazeEnv

