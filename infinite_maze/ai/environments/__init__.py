"""
Environment implementations for the Infinite Maze AI.

This package contains different environment implementations used for training 
and evaluating AI agents at different phases of the training curriculum.
"""

# Re-export the environment classes for easier access
from .environment_phase1 import InfiniteMazeEnv

# Phase2 environment will be added here in the future

# Export all relevant classes
__all__ = ['InfiniteMazeEnv']
