# Infinite Maze AI Training Plan

## 1. Environment Analysis

### Game Mechanics Overview
- **Procedural Maze Generation**: The game creates an infinite, procedurally generated maze u### Exploration Strategy**: 
  - ε-greedy with decay from 1.0 to 0.05 over 2M steps
  - 10% random actions during evaluation for robustness
  - Direction-balanced exploration: Ensure exploration samples all movement directionsg a modified depth-first search algorithm
- **Player Movement**: Four-directional movement (UP, DOWN, LEFT, RIGHT) with collision detection
- **Pacing Mechanic**: An advancing boundary that pushes from the left, increasing in speed every 30 seconds
- **Scoring System**: +1 point for rightward movement, -1 point for leftward movement (minimum 0)
- **Game Over Condition**: When the pace line catches up to the player

### State Representation Analysis
- **Maze Structure**: Collection of horizontal and vertical lines forming walls
- **Player Position**: (x,y) coordinates within game boundaries
- **Pace Line**: Position and current speed of the advancing boundary
- **Game Boundaries**: Fixed vertical limits, infinite horizontal extent
- **Score**: Current player score based on movement history

### Core Challenges
- **Infinite Environment**: Unbounded, procedurally generated maze requiring adaptable strategies
- **Time Pressure**: Increasing pace requires progressively faster decision-making
- **Path Planning**: Finding efficient routes through unpredictable maze structures
- **Exploration-Exploitation Balance**: Weighing immediate rightward progress against longer-term survival
- **Varied Maze Patterns**: Adapting to different maze structures and densities

## 2. Model Architecture Design

### Primary Architecture: Deep Q-Network (DQN)

#### Input Layer
- **Visual State Representation**: 
  - Local grid representation of maze (11×11 centered on player)
  - Multiple channels for different entity types:
    - Channel 1: Walls (1 for wall, 0 for open)
    - Channel 2: Player position (1 at player position, 0 elsewhere)
    - Channel 3: Pace line proximity (values decreasing with distance)
    - Channel 4: Visited cells (1 for visited, 0 for unvisited)
- **Numerical Features**: 
  - Distance to pace line
  - Current pace value
  - Available directions (binary features)
  - Time since last pace increase

#### Hidden Layers
```
CNN Backbone (adapted for smaller input size):
- Conv2D: 32 filters, 3×3 kernel, stride 1, ReLU
- Conv2D: 64 filters, 3×3 kernel, stride 1, ReLU
- Conv2D: 64 filters, 3×3 kernel, stride 1, ReLU
- Flatten

Combined Processing:
- Dense: 256 neurons, ReLU (processing CNN output)
- Dense: 128 neurons, ReLU (combining with numerical features)
- Dense: 128 neurons, ReLU
```

#### Output Layer
- 5 neurons (one per action: UP, DOWN, LEFT, RIGHT, NO_ACTION)
- Linear activation function

#### Justification
1. **CNN Backbone**: Effective for processing spatial relationships in the maze
2. **Multiple Dense Layers**: Integration of spatial and numerical features
3. **Modest Size**: Balance between capacity and computational efficiency
4. **ReLU Activation**: Proven effectiveness for deep RL problems
5. **Separate Output Per Action**: Clear Q-value estimation for each possible move

### Specialized Components
- **Recurrent Memory**: LSTM layer (128 units) for tracking unexplored paths
- **Attention Mechanism**: For focusing on critical decision points in complex maze sections

## 3. Training Methodology

### Selected Algorithm: Rainbow DQN

Rainbow combines several DQN improvements:
- **Double Q-Learning**: Reduces overestimation bias
- **Prioritized Experience Replay**: Focuses on important transitions
- **Dueling Network**: Separate estimation of state value and action advantages
- **Multi-step Learning**: Faster propagation of rewards
- **Distributional RL**: Models distribution of returns
- **Noisy Networks**: Parameter-space exploration

#### Rationale for Selection
1. **Sample Efficiency**: Critical for complex environments with sparse rewards
2. **Stability**: Combined improvements address various DQN limitations
3. **Performance**: State-of-the-art results on challenging environments
4. **Implementation Availability**: Well-supported in major RL libraries

## 4. Curriculum Design

> **Important**: Each phase uses the best model checkpoint from the previous phase as its starting point. Do not proceed to the next phase until all success criteria for the current phase have been fully satisfied.

### Phase 1: Basic Navigation (500K steps)
- **Environment**: Training-specific environment with static mazes, no advancing pace line
- **Objective**: Master basic collision avoidance and strategic rightward progress
- **Reward Focus**: +1 for rightward movement, -0.8 for collisions
- **Success Criteria**: 
  - Consistent forward movement (>90% successful rightward attempts)
  - Minimal collisions (<5% of actions result in wall collisions) 
  - Average score of at least 200 points per episode
- **Anti-bias Strategy**: Start training immediately with maze structures present
- **Checkpoint**: Save best-performing model when success criteria are met

### Phase 2: Complex Navigation & Pace Introduction (1M steps)
- **Starting Point**: Best checkpoint from Phase 1
- **Environment**: Dynamic mazes with constant, slow pace
- **Objective**: Learn to maintain position ahead of advancing boundary
- **Reward Focus**: Survival time + rightward progress + path discovery reward
- **Success Criteria**: 
  - Maintaining safe distance from pace line (average distance >100 pixels)
  - Survival time of at least 2 minutes per episode
  - Vertical movement utilization in 20-40% of actions
  - Minimal oscillation behavior (<5% of navigation attempts)
- **Anti-bias Strategy**: Include scenarios with dense maze sections requiring up/down navigation
- **Gradual Pace Introduction**: Start with pace at 25% speed, gradually increase over 100K steps
- **Enhanced Vertical Movement Training**: Add specialized corridor scenarios
- **Model Architecture**: Activate dueling network architecture and add memory mechanism
- **Reward Function Enhancement**: Improved vertical movement rewards and balanced pace distance incentives
- **Oscillation Mitigation**: Implement detection and prevention of up-down oscillation patterns at vertical walls (see [oscillation_mitigation.md](oscillation_mitigation.md))
- **Checkpoint**: Save best-performing model when success criteria are met

### Phase 3: Varied Maze Structures (1.5M steps)
- **Starting Point**: Best checkpoint from Phase 2
- **Environment**: Multiple maze generation parameters
- **Objective**: Generalize navigation strategies across maze variations
- **Reward Focus**: Efficient path finding through complex sections
- **Success Criteria**: 
  - Consistent performance across maze types (≤15% variance in survival time)
  - Path efficiency ratio >0.7 (forward progress / total movement)
  - Successful navigation of narrow corridors and complex junctions (≥80% success rate)
- **Checkpoint**: Save best-performing model when success criteria are met

### Phase 4: Progressive Difficulty (2M steps)
- **Starting Point**: Best checkpoint from Phase 3
- **Environment**: Transition to actual game environment with full mechanics including pace acceleration
- **Objective**: Adapt to increasing speed and pressure while maintaining navigation skills
- **Reward Focus**: Long-term survival at escalating pace levels
- **Success Criteria**: 
  - Reaching survival time of ≥4 minutes consistently
  - Adapting to at least 3 pace increases
  - Maintaining navigation skills when transitioning from open area to maze (≤20% performance drop)
- **Transition Strategy**: Gradually introduce more complex maze patterns as in the actual game
- **Checkpoint**: Save best-performing model when success criteria are met

### Phase 5: Adversarial Training (1M steps)
- **Starting Point**: Best checkpoint from Phase 4
- **Environment**: Deliberately challenging maze configurations
- **Objective**: Build robustness against worst-case scenarios
- **Reward Focus**: Recovery from difficult situations
- **Success Criteria**: 
  - Successful navigation of adversarial mazes (≥70% escape rate from traps)
  - Recovery from near-pace-line collisions
  - Maintaining performance with reduced observation quality (≤15% degradation)
- **Final Model**: Select best-performing model for deployment

## 5. Training Implementation

### Environment Setup
- **OpenAI Gym Interface**: Custom environment implementing gym.Env
- **Observation Processing**:
  - Visual component: 11×11 multi-channel grid centered on player
  - Feature vector: Pace value, distances to nearest walls in 8 directions
  - Stacked frames: Last 4 frames to capture movement
  - Path quality indicators: Openness measures and dead-end detection
- **Action Space**: Discrete(5) for UP, DOWN, LEFT, RIGHT, NO_ACTION
- **Execution Modes**:
  - **Headless Mode**: Modified game engine for accelerated training without rendering
  - **Visual Mode**: Simple rendering option for direct observation and debugging
- **Training-specific Environment**:
  - Custom version separate from the actual game
  - Configurable parameters for training progression
  - Start with maze structures present with varying complexities
  - Gradual transition to match the actual game environment in later phases
- **Anti-bias Mechanisms**:
  - Include varied starting scenarios with different wall configurations
  - Curriculum progression from simple to complex navigation challenges
  - Track action distribution and penalize excessive rightward bias
- **Visualization for Inspection**:
  - Toggle between headless and visual modes with a simple flag
  - Option to slow down execution speed during visual inspection
  - Basic state information display (current reward, action taken, episode length)
  - Ability to save and load training checkpoints for visual inspection
- **Training Progress Tracking**:
  - Real-time display of elapsed training time (total and per phase)
  - Estimated time remaining for current phase based on steps/hour
  - Progress indicators showing percentage completion of current phase
  - Time-based performance metrics (e.g., steps/second, episodes/hour)
  - Periodic ETA updates adjusted based on actual training speed

### Reward Function Design
```python
def calculate_reward(self, state, action, new_state, done):
    reward = 0
    
    # Base movement rewards - modified to prevent "always go right" bias
    if rightward_movement and not collision:
        # Check if the rightward movement was beneficial (not into a wall soon)
        if path_is_open_ahead(new_state, 3):  # Look 3 steps ahead
            reward += 1.0
        else:
            reward += 0.3  # Reduced reward for suboptimal rightward movement
    elif leftward_movement:
        reward -= 1.0
    
    # Strategic vertical movement rewards
    if (upward_movement or downward_movement) and path_improves(state, new_state):
        reward += 0.5  # Reward for vertical movement that leads to better paths
    
    # Distance-based rewards
    pace_line_distance = new_state.player_x - new_state.pace_line_x
    reward += 0.1 * min(10, pace_line_distance / 50)  # Capped distance bonus
    
    # Path discovery rewards (prevent "always go right" bias)
    if discovered_better_path(state, new_state):
        reward += 0.7  # Significant reward for finding efficient paths
    
    # Survival incentive
    reward += 0.05  # Small reward for surviving each step
    
    # Penalties
    if collision:
        reward -= 0.8  # Increased collision penalty
    
    # Repeated action penalty (discourage mindless direction holding)
    if same_action_count > 5:
        reward -= 0.1 * (same_action_count - 5)  # Increasing penalty for repetition
    
    # Oscillation detection and penalty (Phase 2+)
    oscillation_detected = self._detect_oscillation(self.action_history, self.position_history)
    if oscillation_detected:
        # Apply significant penalty to discourage oscillation patterns
        reward -= 0.8
        # Escalate penalty for persistent oscillation
        consecutive_oscillations = self._count_consecutive_oscillations()
        if consecutive_oscillations > 3:
            reward -= 0.2 * (consecutive_oscillations - 3)
    
    # Terminal state
    if done:
        reward -= 10.0  # Strong penalty for game over
    
    return reward
```

### Training Configuration
- **Batch Size**: 128
- **Learning Rate**: 2.5e-4 with linear decay
- **Discount Factor (γ)**: 0.99
- **Target Network Update**: Every 8,000 steps
- **Replay Buffer Size**: 1,000,000 transitions
- **Exploration Strategy**: 
  - ε-greedy with decay from 1.0 to 0.05 over 2M steps
  - 10% random actions during evaluation for robustness
- **Logging and Monitoring**:
  - Training speed metrics (steps/second, episodes/hour)
  - Time-based checkpointing (every 2 hours)
  - Progress indicators for each phase (% complete)
  - Auto-generated ETA for phase completion
  - CSV log files with timestamped performance metrics

## 6. Feature Engineering

### Core Features
1. **Local View Maze Representation**:
   - Fixed-size 11×11 grid centered on player
   - Multiple channels for different features:
     - Wall presence (binary)
     - Visited cell markers
     - Pace line proximity gradient
   - Sufficient for local decision-making in infinite maze

2. **Distance Transforms**:
   - Distance to nearest walls in 8 directions
   - Distance to pace line
   - Distance to nearest dead end

3. **Path Analysis**:
   - Identification of open corridors vs. constricted areas
   - Dead-end detection within visible range
   - Potential path widths ahead of player

4. **Temporal Features**:
   - Recent movement history (last 4 actions)
   - Pace acceleration countdown
   - Survival time at current pace level

### Feature Importance Analysis
Regular feature importance evaluation will be conducted to refine the state representation:
- Permutation importance testing
- Ablation studies removing individual features
- Correlation analysis between features and performance

## 7. Data Collection and Management

### Experience Collection
- **Parallel Environments**: 16 simultaneous environments
- **Frames per Environment**: 250K
- **Total Frames**: 4M per training session
- **Training Steps**: 1M steps (4 frames per step)
- **Demonstration Data**: Optional expert human gameplay sessions

### Replay Buffer Design
- **Prioritized Experience Replay**:
  - Transitions stored with priority based on TD error
  - Importance sampling to correct for bias
  - Regular re-computation of priorities

- **Efficient Storage**:
  - Compression of visual observations
  - Circular buffer implementation
  - Segmented storage for efficient sampling

### Training Data Analysis
Regular analysis of collected experience:
- State distribution visualization
- Reward distribution monitoring
- Action frequency analysis
- Buffer memory utilization

---

This training plan provides a comprehensive approach to developing an AI agent for the Infinite Maze game, addressing the unique challenges of maze navigation under increasing time pressure. The plan is designed to be practical, focusing on established techniques with proven effectiveness, while incorporating specialized components for the specific dynamics of this game environment.

## Additional Documentation

- [Training Safeguards](training_safeguards.md) - Detailed strategies to prevent the "always go right" bias problem and other common training pitfalls
- [State Representation](state_representation.md) - Detailed explanation of the state representation approach, including the local view grid design and feature encoding
- [Phase Transition Checklist](phase_transition.md) - Comprehensive guidelines for moving between training phases, including required performance metrics and technical requirements
- [Oscillation Mitigation](oscillation_mitigation.md) - Specific approaches to address oscillatory behavior where the agent repeatedly alternates between up and down movements when encountering vertical walls

> **Implementation Note**: Each phase builds upon the previous one through transfer learning. Do not proceed to the next phase until all success criteria for the current phase have been fully satisfied. This sequential approach ensures the model develops a strong foundation before tackling more complex challenges.
