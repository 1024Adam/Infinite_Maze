"""
Reinforcement Learning Package for Infinite Maze Game

This package contains RL environment wrappers, training scripts, and utilities
for training RL agents to play the Infinite Maze game.
"""

from .environment import InfiniteMazeEnv, InfiniteMazeWrapper

__all__ = ['InfiniteMazeEnv', 'InfiniteMazeWrapper']
