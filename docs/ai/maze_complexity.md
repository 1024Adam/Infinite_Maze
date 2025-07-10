# Maze Complexity Control

The Infinite Maze AI training system now includes a configurable maze complexity feature that allows for progressive difficulty scaling during training. This document explains the implementation and usage of this feature.

## Overview

Maze complexity is controlled through a `maze_simplicity` parameter that determines how many wall lines are removed from the perfect maze. A perfect maze (simplicity = 0.0) has exactly one path between any two points, while higher simplicity values create additional paths by removing more walls.

## Implementation Details

### Maze Simplicity Factor

The maze simplicity factor is a floating-point value between 0.0 and 1.0:

- **0.0**: Perfect maze (standard gameplay)
- **0.1-0.2**: Slightly simplified maze
- **0.3-0.4**: Moderately simplified maze
- **0.5+**: Very simple maze with many alternative paths

### Technical Implementation

The maze generation algorithm has been enhanced to:

1. First generate a perfect maze using Kruskal's algorithm (ensuring all cells are connected)
2. After the perfect maze is generated, randomly remove additional walls based on the simplicity factor
3. The number of walls removed is proportional to the total number of walls and the simplicity factor

```python
# After generating a perfect maze:
if simplicity_factor > 0:
    # Calculate how many additional walls to remove
    remaining_walls = len(lines)
    walls_to_remove = int(remaining_walls * simplicity_factor)
    
    # Remove random walls (but not too many to keep the maze structure)
    for _ in range(walls_to_remove):
        if len(lines) > width + height:  # Keep a minimum number of walls
            line_num = randint(0, len(lines) - 1)
            del lines[line_num]
```

## Usage in AI Training

### Curriculum Learning

The simplicity factor enables curriculum learning by gradually increasing maze complexity during training:

1. Start with very simple mazes (high simplicity factor, e.g., 0.5)
2. Gradually decrease the simplicity factor as training progresses
3. End with perfect mazes (simplicity factor = 0.0)

### Configurable Training Stages

The training module implements progressive stages:

```
Curriculum Stages:
Stage 1: 0.5 simplicity - Very simple mazes with many paths
Stage 2: 0.3 simplicity - Moderately complex mazes
Stage 3: 0.15 simplicity - More complex mazes
Stage 4: 0.05 simplicity - Nearly perfect mazes
Stage 5: 0.0 simplicity - Perfect mazes with single paths
```

### Command-Line Usage

To enable curriculum learning with progressive maze complexity:

```bash
python -m infinite_maze.ai.training.train_phase1 --steps 100000 --curriculum
```

## Benefits for AI Training

1. **Easier Initial Learning**: Simpler mazes allow the AI to learn basic navigation before tackling complex mazes
2. **Progressive Challenge**: Gradually increasing difficulty helps the model generalize better
3. **Transfer Learning**: Knowledge gained from simpler mazes transfers to more complex ones
4. **Faster Training Convergence**: Starting with simpler problems accelerates initial learning

## Integration with Evaluation

While training can use simplified mazes, evaluation always uses standard perfect mazes (simplicity = 0.0) to ensure the agent is evaluated against the actual game difficulty.

## Configuration Settings

The default maze simplicity can be configured in `config.py`:

```python
# Maze generation settings
MAZE_SIMPLICITY: float = 0.0  # 0.0 = perfect maze (one path), higher values create simpler mazes
```

This allows game developers and AI researchers to experiment with different maze complexity levels for both gameplay and training purposes.
