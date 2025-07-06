# Oscillation Mitigation Strategies

This document outlines specific approaches to address the oscillatory behavior observed in Phase 2 training, where the agent rapidly alternates between up and down movements when encountering vertical walls.

## Problem Definition

Oscillation occurs when the agent:
1. Encounters a vertical wall obstacle
2. Attempts to navigate around it but lacks a consistent strategy
3. Rapidly alternates between upward and downward movements
4. Makes little or no horizontal progress
5. Gets stuck in this repetitive pattern

This behavior is counterproductive as it:
- Wastes time and actions
- Prevents exploration of better paths
- May cause the agent to be caught by the pace line
- Indicates poor learning of appropriate navigation strategies

## Detection Methods

To address oscillation, we first need reliable detection methods:

### 1. Action History Analysis

```python
def _detect_oscillation_from_actions(self, recent_actions, window_size=8):
    """
    Detect oscillation patterns in recent actions.
    
    Args:
        recent_actions: List of recent actions taken by the agent
        window_size: Number of recent actions to analyze
    
    Returns:
        Boolean indicating whether oscillation is detected
    """
    # Extract only vertical actions (UP=0, DOWN=2) from recent history
    vertical_actions = [a for a in recent_actions[-window_size:] if a in [0, 2]]
    
    if len(vertical_actions) < 4:
        return False  # Need enough vertical actions to detect pattern
    
    # Check for alternating pattern (UP, DOWN, UP, DOWN or DOWN, UP, DOWN, UP)
    alternating = True
    for i in range(len(vertical_actions) - 1):
        if vertical_actions[i] == vertical_actions[i+1]:
            alternating = False
            break
    
    # Check for horizontal progress during this period
    horizontal_actions = [a for a in recent_actions[-window_size:] if a == 1]  # RIGHT=1
    made_horizontal_progress = len(horizontal_actions) > 0
    
    # It's oscillation if we have alternating vertical actions with minimal horizontal movement
    return alternating and not made_horizontal_progress
```

### 2. Position Tracking

```python
def _detect_oscillation_from_positions(self, position_history, window_size=10):
    """
    Detect oscillation by analyzing recent position changes.
    
    Args:
        position_history: List of recent (x, y) positions
        window_size: Number of recent positions to analyze
    
    Returns:
        Boolean indicating whether oscillation is detected
    """
    if len(position_history) < window_size:
        return False
    
    recent_positions = position_history[-window_size:]
    
    # Extract x and y coordinates
    x_positions = [pos[0] for pos in recent_positions]
    y_positions = [pos[1] for pos in recent_positions]
    
    # Calculate horizontal progress
    x_progress = x_positions[-1] - x_positions[0]
    
    # Calculate vertical movement
    y_changes = [abs(y_positions[i+1] - y_positions[i]) for i in range(len(y_positions)-1)]
    total_y_movement = sum(y_changes)
    
    # Check for significant vertical movement with minimal horizontal progress
    if total_y_movement > 3 * abs(x_progress) and x_progress < 3:
        # Look for alternating up/down pattern in y changes
        direction_changes = 0
        for i in range(1, len(y_changes)):
            if (y_positions[i] - y_positions[i-1]) * (y_positions[i+1] - y_positions[i]) < 0:
                direction_changes += 1
        
        # High number of direction changes indicates oscillation
        return direction_changes >= 3
    
    return False
```

### 3. Wall Proximity Analysis

```python
def _detect_oscillation_near_wall(self, state, action_history, position_history):
    """
    Detect oscillation specifically when near vertical walls.
    
    Args:
        state: Current state observation
        action_history: List of recent actions
        position_history: List of recent positions
    
    Returns:
        Boolean indicating whether oscillation near a wall is detected
    """
    # Check if agent is near a vertical wall
    if not self._is_near_vertical_wall(state):
        return False
    
    # Combine action and position analysis for more accurate detection
    action_oscillation = self._detect_oscillation_from_actions(action_history)
    position_oscillation = self._detect_oscillation_from_positions(position_history)
    
    return action_oscillation or position_oscillation
```

## Mitigation Strategies

### 1. Reward Function Engineering

```python
def calculate_reward(self, state, action, next_state, done):
    reward = 0
    
    # Base navigation rewards
    if self._hit_wall(next_state):
        reward -= 1.0
    else:
        reward += 0.1
    
    # Other standard rewards...
    
    # Anti-oscillation penalties
    oscillation_detected = self._detect_oscillation(state, self.action_history, self.position_history)
    if oscillation_detected:
        # Apply significant penalty to discourage oscillation
        reward -= 0.8
        
        # Add stronger penalty if this has been happening repeatedly
        consecutive_oscillations = self._count_consecutive_oscillations()
        if consecutive_oscillations > 3:
            # Escalating penalty for persistent oscillation
            reward -= 0.2 * (consecutive_oscillations - 3)
    
    # Directional momentum reward
    if len(self.action_history) >= 2:
        last_action = self.action_history[-1]
        prev_action = self.action_history[-2]
        
        # Reward for maintaining consistent direction (especially vertical)
        if last_action in [0, 2] and last_action == prev_action:
            # Same vertical direction
            reward += 0.15
    
    # Wall avoidance strategy reward
    if self._is_near_vertical_wall(state):
        if action == 1:  # RIGHT movement near wall
            # Reward horizontal progress when near walls
            reward += 0.2
            
        # Check if agent found a path around the wall
        if self._made_progress_around_wall(state, next_state):
            reward += 0.5  # Significant reward for successful wall navigation
    
    return reward
```

### 2. Specialized Training Scenarios

```python
def generate_vertical_wall_scenario(self):
    """
    Generate a specialized training scenario with vertical walls.
    """
    maze = np.zeros((20, 30))
    
    # Create a vertical wall that requires deliberate navigation
    for y in range(5, 15):
        maze[y, 10] = 1  # Vertical wall
    
    # Add gaps that require deliberate choice
    maze[7, 10] = 0
    maze[12, 10] = 0
    
    # Set player position
    start_x, start_y = 5, 10
    
    # Create environment with this specific maze layout
    env = MazeEnvironment(
        maze=maze, 
        start_position=(start_x, start_y),
        pace_enabled=True,
        pace_start_distance=15,
        pace_speed=0.5  # Reduced pace speed for learning
    )
    
    return env
```

### 3. Memory Mechanisms

```python
class EnhancedMazeAgent(nn.Module):
    def __init__(self, input_shape, action_space):
        super(EnhancedMazeAgent, self).__init__()
        
        # CNN backbone (unchanged)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate CNN output size
        conv_out_size = self._get_conv_output(input_shape)
        
        # LSTM for temporal memory - helps remember recent positions and detect oscillation
        self.lstm = nn.LSTM(conv_out_size, 512, batch_first=True)
        
        # Attention mechanism specifically for wall features
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 49),  # Attention map size
            nn.Softmax(dim=1)
        )
        
        # Value and Advantage streams (Dueling DQN architecture)
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_space)
        )
```

### 4. Training Process Modifications

```python
class EnvironmentWrapper(gym.Wrapper):
    def __init__(self, env):
        super(EnvironmentWrapper, self).__init__(env)
        self.position_history = deque(maxlen=20)
        self.action_history = deque(maxlen=20)
        self.oscillation_count = 0
    
    def step(self, action):
        # Store action
        self.action_history.append(action)
        
        # Execute action
        obs, reward, done, info = self.env.step(action)
        
        # Store position
        self.position_history.append((info['player_x'], info['player_y']))
        
        # Detect and track oscillation
        if len(self.position_history) >= 10:
            oscillation_detected = self._detect_oscillation(self.position_history, self.action_history)
            if oscillation_detected:
                self.oscillation_count += 1
                # Add to info for monitoring
                info['oscillation_detected'] = True
                info['oscillation_count'] = self.oscillation_count
        
        return obs, reward, done, info
```

### 5. Advanced Anti-oscillation Techniques

#### A. Pattern Breaking

```python
def _apply_pattern_breaking(self, state, action_probs):
    """
    Modify action probabilities to break oscillation patterns when detected.
    
    Args:
        state: Current state
        action_probs: Original action probabilities
    
    Returns:
        Modified action probabilities
    """
    if not self._detect_oscillation(self.state, self.action_history, self.position_history):
        return action_probs  # No oscillation, return original probs
    
    modified_probs = action_probs.copy()
    
    # Identify the main oscillation pattern
    up_down_pattern = self._identify_oscillation_pattern()
    
    if up_down_pattern == "UP_DOWN":
        # If oscillating UP-DOWN, temporarily boost RIGHT probability
        modified_probs[1] *= 1.5  # Boost RIGHT
        # And reduce the dominant oscillation directions
        modified_probs[0] *= 0.5  # Reduce UP
        modified_probs[2] *= 0.5  # Reduce DOWN
    
    # Renormalize probabilities
    modified_probs = modified_probs / np.sum(modified_probs)
    
    return modified_probs
```

#### B. Temporary Exploration Boost

```python
def select_action(self, state):
    """
    Select action with temporary exploration boost when oscillation is detected.
    """
    if np.random.random() < self.epsilon:
        # Regular epsilon-greedy exploration
        return random.randrange(self.action_space)
    
    # Check for oscillation
    oscillation_detected = self._detect_oscillation(state, self.action_history, self.position_history)
    
    if oscillation_detected:
        # Temporarily increase exploration when oscillating
        if np.random.random() < 0.4:  # 40% chance to explore when oscillating
            # Bias exploration toward RIGHT movement to break pattern
            exploration_probs = [0.2, 0.6, 0.2, 0.0]  # UP, RIGHT, DOWN, LEFT
            return np.random.choice(self.action_space, p=exploration_probs)
    
    # Normal DQN action selection
    with torch.no_grad():
        q_values = self.policy_net(state.unsqueeze(0))
        return q_values.max(1)[1].item()
```

## Implementation in Environment

```python
class Phase2MazeEnvironment(gym.Env):
    def __init__(self, maze_size=(20, 30), pace_enabled=True):
        super(Phase2MazeEnvironment, self).__init__()
        
        # Standard environment setup
        self.maze_size = maze_size
        self.maze = self._generate_maze(maze_size)
        self.player_position = self._get_valid_start_position()
        self.pace_enabled = pace_enabled
        self.pace_position = max(0, self.player_position[0] - 10)
        self.pace_speed = 0.1
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(4)  # UP, RIGHT, DOWN, LEFT
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(1, maze_size[0], maze_size[1]), 
            dtype=np.float32
        )
        
        # Anti-oscillation tracking
        self.position_history = []
        self.action_history = []
        self.oscillation_count = 0
    
    def step(self, action):
        # Store current position and action for oscillation detection
        self.position_history.append(self.player_position.copy())
        self.action_history.append(action)
        
        # Trim histories to prevent memory growth
        if len(self.position_history) > 20:
            self.position_history.pop(0)
        if len(self.action_history) > 20:
            self.action_history.pop(0)
        
        # Process action
        new_position = self._get_new_position(action)
        reward = self._calculate_reward(action, new_position)
        
        # Update player position if valid move
        if self._is_valid_position(new_position):
            self.player_position = new_position
        
        # Update pace line
        if self.pace_enabled:
            self.pace_position += self.pace_speed
        
        # Check termination conditions
        done = self._is_terminal_state()
        
        # Detect oscillation
        oscillation_detected = self._detect_oscillation()
        if oscillation_detected:
            self.oscillation_count += 1
            # Add oscillation penalty to reward
            reward -= 0.5 * min(5, self.oscillation_count * 0.5)  # Increasing penalty
        
        # Create observation
        observation = self._get_observation()
        
        # Info dictionary
        info = {
            'player_position': self.player_position,
            'pace_position': self.pace_position,
            'oscillation_detected': oscillation_detected,
            'oscillation_count': self.oscillation_count
        }
        
        return observation, reward, done, info
```

## Evaluation Metrics

To measure the effectiveness of oscillation mitigation:

1. **Oscillation Frequency**:
   - Count of oscillation incidents per episode
   - Average duration of oscillation patterns
   - Percentage of time spent in oscillation

2. **Navigation Efficiency**:
   - Horizontal progress rate near vertical walls
   - Time taken to navigate past vertical obstacles
   - Consistency of vertical direction choices

3. **Learning Progress**:
   - Reduction in oscillation incidents over training episodes
   - Improvement in reward when encountering vertical walls

## Conclusion

Oscillation at vertical walls is a common reinforcement learning pathology where agents get stuck in repetitive patterns. By implementing these detection and mitigation strategies, the agent should develop more intelligent navigation behavior, leading to improved maze exploration efficiency and better overall performance in Phase 2 training.
