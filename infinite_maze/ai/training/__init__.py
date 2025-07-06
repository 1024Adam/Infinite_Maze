"""
Training module for Infinite Maze AI.

This module contains training scripts for reinforcement learning agents
for the Infinite Maze game.
"""

# Instead of importing directly, we'll expose the function through a lazy import system
# This avoids circular imports when the module is run directly

__all__ = ['train_phase_1']

# Define the function in __init__ that will import the real function when needed
def train_phase_1(*args, **kwargs):
    """
    Train the agent for Phase 1 of the curriculum.
    This is a proxy function that imports the actual implementation when called.
    """
    from infinite_maze.ai.training.train_phase1 import train_phase_1 as _train_phase_1
    return _train_phase_1(*args, **kwargs)
