# Maze Simplicity Factor

## Overview

The Infinite Maze project implements a configurable "maze simplicity factor" that allows for creating mazes with varying levels of complexity. This feature is particularly important for the AI training curriculum.

## How It Works

The maze simplicity factor controls how many additional walls are removed after generating a "perfect" maze (a maze with exactly one path between any two points):

- **Simplicity Factor = 0.0**: A perfect maze with exactly one path between any two points (most difficult)
- **Simplicity Factor = 0.5**: A significantly simplified maze with many possible paths (easier for navigation)

Higher simplicity values result in more walls being removed, creating multiple possible paths between points and making the maze easier to navigate.

## Training Approach

Our AI training curriculum uses the maze simplicity factor in a structured way:

1. **Phase 1 (Basic Navigation)**
   - Uses simplified mazes (simplicity factor = 0.3)
   - Focus on learning basic movement and collision avoidance
   - Evaluation is done on the same simplified maze structure

2. **Phase 2 (Advanced Navigation)**
   - Begins with introducing the agent to perfect mazes (simplicity factor = 0.0)
   - Focus on efficient path finding in more complex environments
   - Introduces the pace line that pushes the player forward

This progressive approach allows the AI to learn foundational skills in a forgiving environment before tackling the challenges of navigating perfect mazes.

## Configuration

The maze simplicity factor can be configured in several ways:

1. **Global Configuration**: Set `MAZE_SIMPLICITY` in the config file
2. **Per-Maze Instance**: Override when creating a maze instance
3. **AI Environment**: Set when creating training environments

Example:
```python
# Create a maze with custom simplicity factor
maze = Maze(game, simplicity_factor=0.2)

# Create an environment with a specific maze complexity
env = InfiniteMazeEnv(maze_simplicity=0.3)
```

## Implementation

The maze generation algorithm works in two phases:
1. First, it creates a perfect maze using Kruskal's algorithm
2. Then, based on the simplicity factor, it removes additional random walls to create multiple paths

This allows for a smooth progression from simple to complex maze structures.
