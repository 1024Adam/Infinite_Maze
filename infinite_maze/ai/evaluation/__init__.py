"""
Evaluation utilities for the Infinite Maze AI.

This package contains modules for evaluating AI agents at different phases 
of the training curriculum.
"""

# Re-export the evaluation functions for easier access
from .evaluate_phase1 import (
    evaluate_agent,
    visualize_evaluation,
    load_trained_agent,
    compare_agents,
    visualize_comparison
)

# Define __all__ to explicitly specify what's exported
__all__ = [
    'evaluate_agent',
    'visualize_evaluation',
    'load_trained_agent',
    'compare_agents',
    'visualize_comparison'
]
