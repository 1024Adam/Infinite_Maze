"""
Compatibility layer for Infinite Maze AI evaluation utilities.

This module re-exports evaluation functions from evaluate_phase1.py for backward
compatibility with existing code. New code should directly import from 
evaluate_phase1 or evaluate_phase2 as appropriate.
"""

# Re-export the evaluation functions for backward compatibility
from infinite_maze.ai.evaluate_phase1 import (
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

# For direct script execution, delegate to evaluate_phase1.py
if __name__ == "__main__":
    import sys
    from infinite_maze.ai.evaluate_phase1 import main as phase1_main
    
    print("Note: Using evaluate.py directly is deprecated. Please use evaluate_phase1.py instead.")
    phase1_main()


    

