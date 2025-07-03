"""
Launcher script for Infinite Maze game.

This script allows running the game with: python run_game.py
"""

import sys
import os
import warnings

# Suppress pygame's pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    from infinite_maze.core.engine import maze
    maze()
