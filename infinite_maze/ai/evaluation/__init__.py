"""
Evaluation module for Infinite Maze AI.

This module contains functions for evaluating and visualizing the performance
of reinforcement learning agents for the Infinite Maze game.
"""

__all__ = ['evaluate_agent', 'load_trained_agent', 'visualize_evaluation']

# Define proxy functions to avoid circular imports
def evaluate_agent(*args, **kwargs):
    """
    Comprehensive evaluation of agent performance.
    Proxy function that imports the real implementation when called.
    """
    from infinite_maze.ai.evaluation.evaluate_phase1 import evaluate_agent as _evaluate_agent
    return _evaluate_agent(*args, **kwargs)

def load_trained_agent(*args, **kwargs):
    """
    Load a trained agent from checkpoint.
    Proxy function that imports the real implementation when called.
    """
    from infinite_maze.ai.evaluation.evaluate_phase1 import load_trained_agent as _load_trained_agent
    return _load_trained_agent(*args, **kwargs)

def visualize_evaluation(*args, **kwargs):
    """
    Visualize the evaluation results.
    Proxy function that imports the real implementation when called.
    """
    from infinite_maze.ai.evaluation.evaluate_phase1 import visualize_evaluation as _visualize_evaluation
    return _visualize_evaluation(*args, **kwargs)
