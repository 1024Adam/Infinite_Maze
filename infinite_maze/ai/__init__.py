"""
Infinite Maze AI module for training and using reinforcement learning models to navigate the maze.

This package contains various components for reinforcement learning:
- environments: Environment implementations compatible with Gymnasium interface
- models: Neural network architecture definitions
- agents: Reinforcement learning agent implementations
- training: Training scripts and utilities
- evaluation: Evaluation and visualization utilities
"""

# Define the exports but use a lazy-loading approach to avoid circular imports
__all__ = [
    'InfiniteMazeEnv',
    'RainbowDQNAgent',
    'train_phase_1',
    'evaluate_agent',
    'load_trained_agent',
    'visualize_evaluation',
]

# Import the components lazily
# Note: This prevents circular imports when modules are executed directly
from infinite_maze.ai.environments import InfiniteMazeEnv
from infinite_maze.ai.agents import RainbowDQNAgent

# Create proxy functions for the training and evaluation modules
def train_phase_1(*args, **kwargs):
    """Proxy for train_phase_1 function"""
    from infinite_maze.ai.training import train_phase_1 as _train_phase_1
    return _train_phase_1(*args, **kwargs)

def evaluate_agent(*args, **kwargs):
    """Proxy for evaluate_agent function"""
    from infinite_maze.ai.evaluation import evaluate_agent as _evaluate_agent
    return _evaluate_agent(*args, **kwargs)

def load_trained_agent(*args, **kwargs):
    """Proxy for load_trained_agent function"""
    from infinite_maze.ai.evaluation import load_trained_agent as _load_trained_agent
    return _load_trained_agent(*args, **kwargs)

def visualize_evaluation(*args, **kwargs):
    """Proxy for visualize_evaluation function"""
    from infinite_maze.ai.evaluation import visualize_evaluation as _visualize_evaluation
    return _visualize_evaluation(*args, **kwargs)