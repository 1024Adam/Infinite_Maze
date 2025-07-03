# State Representation for Infinite Maze AI

This document details the state representation approach for the Infinite Maze AI model, focusing on efficient encoding of an infinite procedurally generated environment.

## Local View Approach

Despite the infinite nature of the maze, the AI only needs to make decisions based on its immediate surroundings. A local view representation provides several advantages:

- **Computational Efficiency**: Smaller input size means faster processing
- **Position Invariance**: Model learns general maze navigation patterns rather than absolute positions
- **Scalability**: Works regardless of how large the maze becomes
- **Consistency**: Provides uniform input regardless of player's absolute position

## Grid Representation Structure

### Core Components

1. **Fixed-size Grid (11×11)**
   - Centered on player position
   - Covers approximately 5-6 moves in any direction
   - Provides sufficient context for strategic decision-making

2. **Multi-Channel Design**
   - **Channel 1 - Walls**: Binary encoding (1 = wall, 0 = open space)
   - **Channel 2 - Player**: Binary encoding (1 = player position, always at center)
   - **Channel 3 - Pace Line**: Gradient values representing proximity
   - **Channel 4 - Visit History**: Binary encoding (1 = previously visited, 0 = unvisited)

3. **Example Grid Representation**

```
Wall Channel:
0 0 0 1 0 0 0 1 0 0 0
0 1 0 1 0 1 0 1 0 1 0
0 1 0 0 0 1 0 0 0 1 0
1 1 0 1 1 1 0 1 1 1 1
0 0 0 1 0 0 0 1 0 0 0
0 1 1 1 0 1 1 1 0 1 0
0 1 0 0 0 1 0 0 0 1 0
1 1 0 1 1 1 0 1 0 1 1
0 0 0 1 0 0 0 1 0 0 0
0 1 0 1 0 1 0 1 0 1 0
0 1 0 0 0 1 0 0 0 1 0

Player Channel:
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
```

### Additional Feature Vector

To complement the grid representation, a feature vector captures:

- **Distance to pace line**: Scalar value
- **Current pace level**: Integer value
- **Time until next pace increase**: Countdown value
- **Available directions**: Binary vector [UP, RIGHT, DOWN, LEFT]
- **Recent actions**: One-hot encoding of last 3 actions taken

## Implementation Considerations

### Extracting the Local View

```python
def extract_local_view(global_maze, player_pos, grid_size=11):
    """
    Extract a local view of the maze centered on the player.
    
    Args:
        global_maze: The complete maze state (can be infinite/procedural)
        player_pos: (x, y) position of the player
        grid_size: Size of the local view (must be odd)
    
    Returns:
        Dictionary with channels for walls, player, pace line, and visited cells
    """
    # Calculate grid boundaries
    half_size = grid_size // 2
    min_x, max_x = player_pos[0] - half_size, player_pos[0] + half_size
    min_y, max_y = player_pos[1] - half_size, player_pos[1] + half_size
    
    # Initialize channels
    wall_channel = np.zeros((grid_size, grid_size))
    player_channel = np.zeros((grid_size, grid_size))
    pace_channel = np.zeros((grid_size, grid_size))
    visit_channel = np.zeros((grid_size, grid_size))
    
    # Set player position (always at center)
    player_channel[half_size, half_size] = 1
    
    # Fill in wall information
    for i in range(grid_size):
        for j in range(grid_size):
            x, y = min_x + i, min_y + j
            
            # Get wall information from global maze
            if (x, y) in global_maze.walls:
                wall_channel[j, i] = 1
                
            # Set visited cells
            if (x, y) in global_maze.visited_cells:
                visit_channel[j, i] = 1
                
            # Calculate pace line proximity
            distance_to_pace = max(1, x - global_maze.pace_line_position)
            pace_channel[j, i] = 1.0 / distance_to_pace
    
    return {
        'walls': wall_channel,
        'player': player_channel,
        'pace': pace_channel,
        'visited': visit_channel
    }
```

### Handling Out-of-Bounds Areas

For areas that haven't been generated yet or are outside the maze bounds:

- **Ungenerated Areas**: Since the maze is procedurally generated, areas not yet visited can be represented as open space until generated
- **Beyond Map Boundaries**: Areas beyond vertical boundaries can be represented as walls

### Efficiency Considerations

- **Memory Efficiency**: The fixed-size representation requires constant memory regardless of maze size
- **Computational Efficiency**: Small input size (11×11×4 = 484 values) is manageable for neural networks
- **Update Efficiency**: Only needs to be recalculated when player moves or environment changes

## Advantages Over Alternative Approaches

### Compared to Global Representation

A global representation of the entire maze would:
- Require ever-increasing memory as the maze grows
- Lead to sparse inputs with mostly irrelevant information
- Force the model to learn position-dependent patterns

### Compared to 1D Sensor Arrays

Using distance sensors in specific directions would:
- Lose the spatial relationship information between walls
- Make pattern recognition of maze structures more difficult
- Provide less context for strategic planning

### Compared to Larger Local Views

A larger grid (e.g., 20×20 or 30×30) would:
- Increase computational requirements
- Provide diminishing returns on decision quality
- Include information too far away to be immediately relevant

## Conclusion

The 11×11 multi-channel grid representation provides an optimal balance of context and efficiency for the Infinite Maze AI. This approach captures the information needed for effective decision-making while remaining computationally efficient and scalable to the infinite nature of the maze environment.

This representation, combined with the numerical feature vector, provides the AI with sufficient information to navigate the maze, avoid the advancing pace line, and develop sophisticated navigation strategies without being overwhelmed by the infinite scope of the environment.
