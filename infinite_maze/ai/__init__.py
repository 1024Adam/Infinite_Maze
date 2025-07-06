"""
Infinite Maze AI module for training and using reinforcement learning models to navigate the maze.

This module is organized into several subpackages:
- environments: Contains different environment implementations for each training phase
- evaluation: Contains utilities for evaluating trained agents
- training: Contains training scripts for different phases
"""

# Re-export main components for backwards compatibility
from infinite_maze.ai.environments import InfiniteMazeEnv
from infinite_maze.ai.agent import RainbowDQNAgent
from infinite_maze.ai.evaluation import evaluate_agent, load_trained_agent
from infinite_maze.ai.training import train_phase_1
