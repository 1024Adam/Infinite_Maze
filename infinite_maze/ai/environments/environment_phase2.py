"""
Training environment for Phase 2 of the Infinite Maze AI.

This module creates a gymnasium-compatible environment for training the AI agent,
with specific modifications for the Phase 2 training as outlined in the training plan.
The key additions from Phase 1 include:
- Constant, slow pace line that advances from the left
- Enhanced maze complexity and variation
- Improved detection and prevention of oscillatory behavior
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import random
from typing import Dict, Tuple, Optional, List, Any
import math
from collections import deque

# Import core game components
from infinite_maze.core.game import Game
from infinite_maze.entities.maze import Maze, Line
from infinite_maze.entities.player import Player
from infinite_maze.utils.config import config

class Phase2MazeEnv(gym.Env):
    """
    OpenAI Gym environment for the Infinite Maze game - Phase 2.
    
    This environment builds upon the Phase 1 environment with the following changes:
    - Introduction of a constant, slow pace line that advances from the left
    - Enhanced oscillation detection and prevention
    - Increased maze complexity and variation
    - More challenging starting positions
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 pace_enabled: bool = True,
                 pace_speed: float = 0.2,  # 20% of normal speed, as specified in training plan
                 pace_acceleration: bool = False,
                 render_mode: Optional[str] = None,
                 grid_size: int = 11,
                 max_steps: int = 10000,
                 maze_density: float = 1.0,
                 start_position_difficulty: float = 0.0,  # 0.0=easy, 1.0=hard
                 oscillation_penalty: float = 0.7):
        """
        Initialize the Infinite Maze Phase 2 training environment.
        
        Args:
            pace_enabled: Whether the advancing pace line is enabled
            pace_speed: Speed multiplier for the pace line (0.2 = 20% of normal speed)
            pace_acceleration: Whether the pace line accelerates over time
            render_mode: Mode for visualization ('human', 'rgb_array', or None)
            grid_size: Size of the observation grid (must be odd)
            max_steps: Maximum steps per episode
            maze_density: Density multiplier for maze walls (1.0 = normal)
            start_position_difficulty: Difficulty of starting positions (0-1)
            oscillation_penalty: Penalty strength for oscillatory behavior
        """
        super().__init__()
        
        # Store configuration
        self.pace_enabled = pace_enabled
        self.pace_speed = pace_speed
        self.pace_acceleration = pace_acceleration
        self.render_mode = render_mode
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.maze_density = maze_density
        self.start_position_difficulty = start_position_difficulty
        self.oscillation_penalty = oscillation_penalty
        
        # Initialize game components (will be properly set in reset())
        self.config = config  # Use the global config instance
        self.game = None
        self.lines = None
        self.maze = None
        self.player = None
        self.pace_line_x = 0  # Will be set properly in reset()
        self.pace_level = 1   # Tracks pace increases for reward calculation
        self.time_to_next_pace_increase = 30  # Seconds until pace increases
        
        # Track episode stats
        self.steps = 0
        self.steps_since_pace_increase = 0
        self.total_reward = 0
        self.action_history = deque(maxlen=20)  # Track recent actions
        self.position_history = deque(maxlen=20)  # Track recent positions
        self.reward_history = deque(maxlen=100)  # Track recent rewards
        
        # Track oscillation detection
        self.consecutive_oscillations = 0
        self.oscillation_count = 0
        self.last_oscillation_step = 0
        
        # Define action space: UP, RIGHT, DOWN, LEFT, NO_ACTION
        self.action_space = spaces.Discrete(5)
        
        # Define observation space based on grid representation
        # 4 channels: walls, player, pace line, visited
        self.channels = 4
        
        # Observation includes both the grid and numerical features
        grid_space = spaces.Box(
            low=0, high=1, 
            shape=(self.grid_size, self.grid_size, self.channels),
            dtype=np.float32
        )
        
        # Numerical features:
        # - Distance to pace line
        # - Current pace level
        # - Time until next pace increase
        # - 4 binary features for available directions
        # - 3 one-hot encoded previous actions
        numerical_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([10000, 10, 30, 1, 1, 1, 1, 1, 1, 1, 1]),
            shape=(11,),
            dtype=np.float32
        )
        
        # Combined observation space
        self.observation_space = spaces.Dict({
            'grid': grid_space,
            'numerical': numerical_space
        })

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Returns:
            Tuple of (observation, info)
        """
        # Initialize or reset game components
        headless = (self.render_mode is None)
        self.game = Game(headless=headless)
        
        # Generate the maze lines with enhanced density variation
        maze_rows = int(config.MAZE_ROWS * self.maze_density)
        maze_cols = int(config.MAZE_COLS * self.maze_density)
        self.lines = Line.generateMaze(self.game, maze_rows, maze_cols)
        
        # Create our Maze wrapper to handle AI interactions
        self.maze = Maze(self.game, self.lines)
        
        # Create the player at a valid starting position based on difficulty
        start_x_base = config.PLAYER_START_X + 20
        start_y_base = config.PLAYER_START_Y
        
        # Phase 2: More challenging starting positions
        start_positions = []
        
        # Easy positions (used when start_position_difficulty is low)
        easy_positions = [
            (start_x_base, start_y_base),  # Default
            (start_x_base + 15, start_y_base - 20),  # Slightly up-right
            (start_x_base + 10, start_y_base + 25),  # Slightly down-right
            (start_x_base + 30, start_y_base),       # Further right
        ]
        
        # Medium difficulty positions
        medium_positions = [
            (start_x_base + 5, start_y_base - 45),   # Far up, close to start
            (start_x_base + 5, start_y_base + 45),   # Far down, close to start
            (start_x_base + 40, start_y_base - 35),  # Further up-right
            (start_x_base + 40, start_y_base + 35),  # Further down-right
        ]
        
        # Hard positions (near walls, in tight spots)
        hard_positions = [
            (start_x_base + 15, start_y_base - 55),  # Very far up
            (start_x_base + 15, start_y_base + 55),  # Very far down
            (start_x_base + 70, start_y_base),       # Far right
            (start_x_base + 60, start_y_base - 40),  # Far up-right
            (start_x_base + 60, start_y_base + 40),  # Far down-right
        ]
        
        # Add positions based on difficulty
        if self.start_position_difficulty < 0.33:
            start_positions.extend(easy_positions)
            start_positions.extend(medium_positions[:2])  # Add a couple medium ones
        elif self.start_position_difficulty < 0.66:
            start_positions.extend(easy_positions[:2])    # Add a couple easy ones
            start_positions.extend(medium_positions)
            start_positions.extend(hard_positions[:1])    # Add one hard one
        else:
            start_positions.extend(medium_positions[:2])  # Add a couple medium ones
            start_positions.extend(hard_positions)
        
        # Select a position and verify it's not inside a wall
        max_attempts = 15
        for _ in range(max_attempts):
            start_x, start_y = random.choice(start_positions)
            # Ensure starting position doesn't collide with walls
            if not self.maze.is_wall(start_x, start_y):
                break
                
        # Create the player
        self.player = Player(start_x, start_y, headless=headless)
        
        # Initialize pace line
        if self.pace_enabled:
            # Phase 2: Pace line starts at a safe distance
            self.pace_line_x = max(0, self.player.getX() - 100)
        else:
            self.pace_line_x = -1000  # Effectively disabled
        
        # Reset episode tracking
        self.steps = 0
        self.steps_since_pace_increase = 0
        self.total_reward = 0
        self.action_history.clear()
        self.position_history.clear()
        self.reward_history.clear()
        self.collision_count = 0
        self.oscillation_count = 0
        self.consecutive_oscillations = 0
        self.last_oscillation_step = 0
        self.game_over = False
        self.pace_level = 1
        self.time_to_next_pace_increase = 30
        
        # Mark initial position as visited
        self.maze.mark_visited(self.player.getX(), self.player.getY())
        
        # Get initial observation
        observation = self._get_observation()
        
        # Initial render to make sure we have a visible game screen
        if self.render_mode == 'human':
            # Process any pending pygame events to prevent freezing
            import pygame
            pygame.event.pump()
            
            # Force initial render
            self.render()
            
            # Wait a short moment to ensure window is visible and responsive
            import time
            time.sleep(0.2)
                
        # Return initial observation and empty info
        return observation, {}
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment using the given action.
        
        Args:
            action: Action to take (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=NO_ACTION)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Track episode steps
        self.steps += 1
        self.steps_since_pace_increase += 1
        
        # Store the state before the action for reward calculation
        old_state = self._get_state_snapshot()
        
        # Map the action to the player movement
        blocked = False
        current_x, current_y = self.player.getX(), self.player.getY()
        speed = self.player.getSpeed()
        
        if action == 0:  # UP
            # Check for collision along the entire movement path
            if not self.maze.check_collision(current_x, current_y, 0, -speed):
                self.player.moveY(-1)  # Move up
            else:
                blocked = True
                
        elif action == 1:  # RIGHT
            # Check for collision along the entire movement path
            if not self.maze.check_collision(current_x, current_y, speed, 0):
                self.player.moveX(1)  # Move right
                self.game.incrementScore()
            else:
                blocked = True
                
        elif action == 2:  # DOWN
            # Check for collision along the entire movement path
            if not self.maze.check_collision(current_x, current_y, 0, speed):
                self.player.moveY(1)  # Move down
            else:
                blocked = True
                
        elif action == 3:  # LEFT
            # Check for collision along the entire movement path
            if not self.maze.check_collision(current_x, current_y, -speed, 0):
                self.player.moveX(-1)  # Move left
                self.game.decrementScore()
            else:
                blocked = True
                
        # action 4 is NO_ACTION, so do nothing
        
        # Double-check that we're not inside a wall and correct if needed
        if self.maze.is_wall(self.player.getX(), self.player.getY()):
            # Emergency correction - reset to previous position
            self.player.setX(current_x)
            self.player.setY(current_y)
            blocked = True
        
        # Track for collision counting
        if blocked:
            self.collision_count += 1
        
        # Store action and position history
        self.action_history.append(action)
        self.position_history.append((self.player.getX(), self.player.getY()))
        
        # Update pace line
        if self.pace_enabled:
            # Phase 2: Constant pace movement
            self.pace_line_x += self.pace_speed
            
            # Phase 2: Optional pace acceleration
            if self.pace_acceleration and self.steps_since_pace_increase >= 300:  # Every ~30 seconds (10 fps)
                self.pace_speed *= 1.2  # Increase pace by 20%
                self.pace_level += 1
                self.steps_since_pace_increase = 0
                self.time_to_next_pace_increase = 30  # Reset countdown
            
            # Time to next pace increase (for observation)
            if self.pace_acceleration:
                self.time_to_next_pace_increase = max(0, 30 - (self.steps_since_pace_increase // 10))
            
            # Check if player is caught by pace line
            if self.player.getX() <= self.pace_line_x:
                self.game_over = True
        
        # Position boundary checks
        if self.player.getX() < config.PLAYER_START_X:
            self.game_over = True
        
        # Mark current position as visited
        self.maze.mark_visited(self.player.getX(), self.player.getY())
        
        # Get the new state after action
        new_state = self._get_state_snapshot()
        
        # Calculate the reward - enhanced for Phase 2
        reward = self._calculate_reward(old_state, action, new_state, self.game_over, blocked)
        self.total_reward += reward
        self.reward_history.append(reward)
        
        # Check if episode is done (game over or max steps reached)
        done = self.game_over or self.steps >= self.max_steps
        
        # Get the observation after taking the action
        observation = self._get_observation()
        
        # Additional info for monitoring
        info = {
            'score': self.game.getScore(),
            'steps': self.steps,
            'collision': blocked,
            'episode_reward': self.total_reward,
            'pace_line_x': self.pace_line_x,
            'pace_speed': self.pace_speed,
            'pace_level': self.pace_level,
            'oscillation_count': self.oscillation_count,
            'distance_to_pace': max(0, self.player.getX() - self.pace_line_x)
        }
        
        # In gymnasium API, we need to return 5 values:
        # observation, reward, terminated, truncated, info
        terminated = self.game_over
        truncated = self.steps >= self.max_steps
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Create the observation based on the current game state.
        
        Returns:
            Dict with 'grid' and 'numerical' features
        """
        # Get player position and pace line position
        player_pos = (self.player.getX(), self.player.getY())
        pace_line_pos = self.pace_line_x
        
        # Extract the local view grid
        grid = self._extract_local_view(player_pos, pace_line_pos)
        
        # Create numerical features
        numerical = np.zeros(11, dtype=np.float32)
        
        # 1. Distance to pace line (normalized)
        distance_to_pace = max(1, player_pos[0] - pace_line_pos)
        numerical[0] = min(distance_to_pace / 500, 1.0)  # Normalize to [0,1]
        
        # 2. Current pace level
        numerical[1] = self.pace_level
        
        # 3. Time until next pace increase (normalized)
        numerical[2] = self.time_to_next_pace_increase / 30.0
        
        # 4-7. Available directions (binary features)
        # Check which directions are valid moves (not into walls)
        dirs = ["UP", "RIGHT", "DOWN", "LEFT"]
        for i, direction in enumerate(dirs):
            numerical[3 + i] = 1.0 if self._is_valid_move(direction) else 0.0
        
        # 7-10. One-hot encoding of the most recent action
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            # Make sure index is within bounds (actions 0-4 map to indices 7-11)
            if 0 <= last_action < 5 and 7 + last_action < len(numerical):
                numerical[7 + last_action] = 1.0
        
        return {
            'grid': grid.astype(np.float32),
            'numerical': numerical
        }
    
    def _extract_local_view(self, player_pos: Tuple[float, float], pace_line_pos: float) -> np.ndarray:
        """
        Extract a local view grid representation of the maze centered on the player.
        
        Args:
            player_pos: (x, y) position of the player
            pace_line_pos: x position of the pace line
            
        Returns:
            Multi-channel grid representation as numpy array
        """
        # Initialize grid with 4 channels
        # Channels: [walls, player, pace_line, visited]
        grid = np.zeros((self.grid_size, self.grid_size, self.channels))
        
        # Calculate grid boundaries
        half_size = self.grid_size // 2
        min_x = int(player_pos[0]) - half_size
        min_y = int(player_pos[1]) - half_size
        
        # Place player (always at center)
        grid[half_size, half_size, 1] = 1.0
        
        # Fill in wall information
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = min_x + i
                y = min_y + j
                
                # Check if position has a wall
                if self.maze.is_wall(x, y):
                    grid[i, j, 0] = 1.0
                
                # Mark visited cells
                grid_x = int(x // config.MAZE_CELL_SIZE)
                grid_y = int(y // config.MAZE_CELL_SIZE)
                if (grid_x, grid_y) in self.maze.visited:
                    grid[i, j, 3] = 1.0
                
                # Pace line proximity - Enhanced for Phase 2 with gradient
                if self.pace_enabled:
                    # Calculate distance and normalize to [0,1] with exponential decay
                    distance_to_pace = x - pace_line_pos
                    if distance_to_pace < 0:
                        # Already past the pace line
                        grid[i, j, 2] = 1.0  # Maximum danger
                    else:
                        # Exponential proximity gradient - stronger signal when close
                        proximity = math.exp(-distance_to_pace / 50)  # Tune the divisor for gradient steepness
                        grid[i, j, 2] = min(1.0, proximity)
        
        return grid
    
    def _get_state_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current game state for reward calculation.
        
        Returns:
            Dict with state information
        """
        return {
            'player_x': self.player.getX(),
            'player_y': self.player.getY(),
            'score': self.game.getScore(),
            'pace_line_x': self.pace_line_x,
            'game_over': self.game_over,
            'pace_level': self.pace_level
        }
    
    def _is_valid_move(self, direction: str) -> bool:
        """
        Check if a move in the given direction would be valid (not into a wall).
        
        Args:
            direction: Direction to check ("UP", "RIGHT", "DOWN", "LEFT")
            
        Returns:
            True if the move is valid, False otherwise
        """
        # Get the player's current position
        x, y = self.player.getX(), self.player.getY()
        speed = self.player.getSpeed()
        
        # Calculate movement vector based on direction
        dx, dy = 0, 0
        if direction == "UP":
            dy = -speed
        elif direction == "RIGHT":
            dx = speed
        elif direction == "DOWN":
            dy = speed
        elif direction == "LEFT":
            dx = -speed
        
        # Use the improved collision detection that checks the entire path
        return not self.maze.check_collision(x, y, dx, dy)
    
    def _detect_oscillation(self) -> bool:
        """
        Enhanced oscillation detection for Phase 2.
        
        This combines both action-based and position-based detection methods
        to more accurately identify oscillatory behavior.
        
        Returns:
            True if oscillation is detected, False otherwise
        """
        # Need enough history to detect patterns
        if len(self.action_history) < 8 or len(self.position_history) < 8:
            return False
            
        # Method 1: Check for alternating vertical actions (UP/DOWN pattern)
        recent_actions = list(self.action_history)
        vertical_actions = [a for a in recent_actions[-6:] if a in [0, 2]]  # UP(0) and DOWN(2)
        
        action_oscillation = False
        if len(vertical_actions) >= 4:
            # Check for alternating pattern
            alternating = True
            for i in range(len(vertical_actions)-1):
                if vertical_actions[i] == vertical_actions[i+1]:
                    alternating = False
                    break
            
            if alternating:
                # Check if horizontal progress is minimal
                recent_positions = list(self.position_history)
                start_x = recent_positions[-6][0] if len(recent_positions) >= 6 else recent_positions[0][0]
                end_x = recent_positions[-1][0]
                horizontal_progress = abs(end_x - start_x)
                
                if horizontal_progress < self.player.getSpeed() * 2:
                    action_oscillation = True
                    
        # Method 2: Position-based oscillation detection
        recent_positions = list(self.position_history)
        position_oscillation = False
        
        if len(recent_positions) >= 8:
            # Extract y positions
            y_positions = [pos[1] for pos in recent_positions[-8:]]
            x_positions = [pos[0] for pos in recent_positions[-8:]]
            
            # Calculate changes in y direction
            y_changes = []
            for i in range(len(y_positions) - 1):
                y_changes.append(y_positions[i+1] - y_positions[i])
                
            # Count direction changes
            direction_changes = 0
            for i in range(len(y_changes) - 1):
                if y_changes[i] * y_changes[i+1] < 0:  # Different signs = direction change
                    direction_changes += 1
            
            # Many direction changes with minimal x progress indicates oscillation
            if direction_changes >= 3:
                x_progress = abs(x_positions[-1] - x_positions[0])
                if x_progress < self.player.getSpeed() * 3:
                    position_oscillation = True
        
        # Combined detection
        oscillation_detected = action_oscillation or position_oscillation
        
        # Update tracking for consecutive oscillations
        if oscillation_detected:
            self.consecutive_oscillations += 1
            # Only count as a new oscillation incident if it's been a while
            if self.steps - self.last_oscillation_step > 10:
                self.oscillation_count += 1
                self.last_oscillation_step = self.steps
        else:
            self.consecutive_oscillations = 0
            
        return oscillation_detected
    
    def _is_near_vertical_wall(self) -> bool:
        """
        Check if the player is near a vertical wall.
        
        Returns:
            True if near vertical wall
        """
        x, y = self.player.getX(), self.player.getY()
        speed = self.player.getSpeed()
        
        # Check for walls to the right
        for i in range(1, 3):  # Check 2 steps ahead
            if self.maze.is_wall(x + i * speed, y):
                return True
        
        # Also check slightly above and below to detect corners
        for i in range(1, 3):
            if self.maze.is_wall(x + i * speed, y - speed) or self.maze.is_wall(x + i * speed, y + speed):
                return True
                
        return False
    
    def _made_progress_around_wall(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> bool:
        """
        Check if the agent successfully navigated around a wall obstacle.
        
        Args:
            old_state: State before action
            new_state: State after action
            
        Returns:
            True if progress was made around a wall
        """
        old_x, old_y = old_state['player_x'], old_state['player_y']
        new_x, new_y = new_state['player_x'], new_state['player_y']
        
        # First, check if we've moved around an obstacle
        # This requires:
        # 1. Horizontal progress was made
        # 2. We moved vertically
        # 3. There was a wall blocking direct rightward movement before
        
        horizontal_progress = new_x > old_x
        vertical_movement = abs(new_y - old_y) > 0
        
        if horizontal_progress and vertical_movement:
            # Check if direct rightward movement was blocked before
            speed = self.player.getSpeed()
            was_blocked = self.maze.is_wall(old_x + speed, old_y)
            
            # Now check if we can move rightward from new position
            can_move_right_now = not self.maze.is_wall(new_x + speed, new_y)
            
            return was_blocked and can_move_right_now
            
        return False
    
    def _path_is_open_ahead(self, state: Dict[str, Any], steps: int = 3) -> bool:
        """
        Check if the path ahead (rightward) is open for n steps.
        
        Args:
            state: Current state
            steps: Number of steps to check ahead
            
        Returns:
            True if path is clear, False if obstacles detected
        """
        x, y = state['player_x'], state['player_y']
        speed = self.player.getSpeed()
        
        # Check directly ahead
        for i in range(1, steps + 1):
            if self.maze.is_wall(x + i * speed, y):
                return False
        
        # Also check slightly above and below for a more complete picture
        # This helps detect narrow passages that might be problematic
        has_path_above = True
        has_path_below = True
        
        for i in range(1, steps + 1):
            if self.maze.is_wall(x + i * speed, y - speed):
                has_path_above = False
            if self.maze.is_wall(x + i * speed, y + speed):
                has_path_below = False
                
        # Return true if we have a direct path or at least one clear path above/below
        return True or has_path_above or has_path_below
    
    def _path_improves(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> bool:
        """
        Determine if the move led to a better path opportunity.
        
        Args:
            old_state: State before action
            new_state: State after action
            
        Returns:
            True if new position offers better rightward pathing options
        """
        # Check if the new position has better rightward movement options
        old_x, old_y = old_state['player_x'], old_state['player_y']
        new_x, new_y = new_state['player_x'], new_state['player_y']
        
        # If we can now move right more freely, this is an improvement
        old_right_clear = self._path_is_open_ahead({'player_x': old_x, 'player_y': old_y}, 2)
        new_right_clear = self._path_is_open_ahead({'player_x': new_x, 'player_y': new_y}, 2)
        
        if not old_right_clear and new_right_clear:
            return True
            
        # Check if we can see more open space to the right from new position
        open_spaces_old = self._count_open_spaces_right(old_x, old_y, 5)
        open_spaces_new = self._count_open_spaces_right(new_x, new_y, 5)
        
        return open_spaces_new > open_spaces_old
    
    def _count_open_spaces_right(self, x: float, y: float, steps: int) -> int:
        """
        Count open spaces to the right of the given position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            steps: Number of steps to check
            
        Returns:
            Number of open spaces
        """
        count = 0
        speed = self.player.getSpeed()
        
        for i in range(1, steps + 1):
            # Check not just directly right but also slightly up and down
            if not self.maze.is_wall(x + i * speed, y):
                count += 1
            if not self.maze.is_wall(x + i * speed, y - speed):
                count += 1
            if not self.maze.is_wall(x + i * speed, y + speed):
                count += 1
                
        return count
    
    def _calculate_reward(self, old_state: Dict[str, Any], action: int, 
                         new_state: Dict[str, Any], done: bool, collision: bool) -> float:
        """
        Calculate the reward based on the action and resulting state change.
        
        Enhanced for Phase 2 with:
        - Pace line distance rewards
        - Better path discovery incentives
        - Stronger oscillation penalties
        - Enhanced vertical movement rewards
        
        Args:
            old_state: State before action
            action: Action taken
            new_state: State after action
            done: Whether the episode is done
            collision: Whether a collision occurred
            
        Returns:
            Calculated reward value
        """
        reward = 0.0
        
        # Extract state information
        old_x, old_y = old_state['player_x'], old_state['player_y']
        new_x, new_y = new_state['player_x'], new_state['player_y']
        pace_x = new_state['pace_line_x']
        
        # Check movement direction
        moved_right = new_x > old_x
        moved_left = new_x < old_x
        moved_up = new_y < old_y
        moved_down = new_y > old_y
        
        # Base movement rewards
        if moved_right and not collision:
            # Phase 2: More nuanced rightward movement rewards
            if self._path_is_open_ahead(new_state, 3):
                reward += 1.0  # Good rightward movement
            else:
                reward += 0.5  # Suboptimal but still rightward
        elif moved_left:
            # Phase 2: Context-sensitive leftward penalty
            # Reduced penalty when near pace line (might be necessary to escape)
            distance_to_pace = old_x - pace_x
            if distance_to_pace < 50:  # Close to pace line
                reward -= 0.3  # Smaller penalty when close to pace
            else:
                reward -= 0.6  # Larger penalty when safe
        
        # Vertical movement rewards - enhanced for Phase 2
        if moved_up or moved_down:
            # Check for beneficial vertical movement
            if self._path_improves(old_state, new_state):
                reward += 0.8  # Significantly increased reward for good vertical movement
            elif self._made_progress_around_wall(old_state, new_state):
                reward += 1.0  # Strong reward for successfully navigating around obstacles
            elif not collision:
                reward += 0.2  # Small reward for non-colliding vertical movement
        
        # Phase 2: Pace line distance rewards
        distance_to_pace = max(1, new_x - pace_x)
        
        # Progressive reward based on distance maintained
        if distance_to_pace >= 150:
            reward += 0.3  # Excellent distance
        elif distance_to_pace >= 100:
            reward += 0.2  # Good distance
        elif distance_to_pace >= 50:
            reward += 0.1  # Adequate distance
        elif distance_to_pace < 30:
            reward -= 0.2  # Dangerously close
        
        # Reward for increasing distance from pace line
        old_distance = max(1, old_x - old_state['pace_line_x'])
        if distance_to_pace > old_distance:
            reward += 0.15  # Increased distance
        
        # Phase 2: Oscillation penalties
        if self._detect_oscillation():
            # Progressive penalty based on consecutive oscillations
            oscillation_penalty = self.oscillation_penalty * min(2.0, 1.0 + (self.consecutive_oscillations / 5))
            reward -= oscillation_penalty
        
        # Collision penalties
        if collision:
            reward -= 0.7  # Slightly reduced from Phase 1
            
            # Phase 2: Context-aware collision penalty
            # Reduce penalty for collisions when close to pace line (emphasizes survival)
            if distance_to_pace < 50:
                reward += 0.2  # Partial penalty reduction
        
        # Survival incentive - Phase 2: Increased to emphasize staying alive
        reward += 0.08  # Up from 0.05 in Phase 1
        
        # Terminal state
        if done:
            # Strong penalty for game over, modulated by survival time
            base_penalty = -10.0
            
            # Reduce penalty slightly for longer survival
            survival_factor = min(1.0, self.steps / 500)  # Caps at 500 steps
            adjusted_penalty = base_penalty * (1.0 - survival_factor * 0.3)  # Up to 30% reduction
            
            reward += adjusted_penalty
        
        return reward
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment for visualization.
        
        Returns:
            RGB array if mode is 'rgb_array', otherwise None
        """
        if self.render_mode is None or self.game is None:
            return None
        
        # Update the screen with current game state
        if self.game.screen is not None:
            # Clear the screen
            self.game.screen.fill(self.game.BG_COLOR)
            
            # Draw the player
            if self.player.getCursor() is not None:
                self.game.screen.blit(self.player.getCursor(), self.player.getPosition())
            
            # Draw the maze lines
            for line in self.lines:
                pygame.draw.line(
                    self.game.getScreen(), 
                    self.game.FG_COLOR, 
                    line.getStart(), 
                    line.getEnd(), 
                    1
                )
            
            # Draw pace line if enabled
            if self.pace_enabled and self.pace_line_x > -1000:
                pygame.draw.line(
                    self.game.getScreen(),
                    pygame.Color(255, 0, 0),  # Red color
                    (self.pace_line_x, self.game.Y_MIN),
                    (self.pace_line_x, self.game.Y_MAX),
                    2
                )
            
            # Display game info
            if self.game.font:
                # Score
                score_text = self.game.font.render(
                    f"Score: {self.game.getScore()}", 1, self.game.FG_COLOR
                )
                self.game.screen.blit(score_text, (config.TEXT_MARGIN, config.TEXT_MARGIN))
                
                # Steps and pace info
                steps_text = self.game.font.render(
                    f"Steps: {self.steps} | Pace Level: {self.pace_level}", 1, self.game.FG_COLOR
                )
                self.game.screen.blit(steps_text, (config.TEXT_MARGIN, config.TEXT_MARGIN + 25))
                
                # Distance to pace
                distance_text = self.game.font.render(
                    f"Distance to Pace: {max(0, self.player.getX() - self.pace_line_x):.1f}", 
                    1, self.game.FG_COLOR
                )
                self.game.screen.blit(distance_text, (config.TEXT_MARGIN, config.TEXT_MARGIN + 50))
                
                # Pace speed
                pace_text = self.game.font.render(
                    f"Pace Speed: {self.pace_speed:.2f}", 1, self.game.FG_COLOR
                )
                self.game.screen.blit(pace_text, (config.TEXT_MARGIN, config.TEXT_MARGIN + 75))
                
                # Oscillation info
                osc_text = self.game.font.render(
                    f"Oscillations: {self.oscillation_count}", 1, self.game.FG_COLOR
                )
                self.game.screen.blit(osc_text, (config.TEXT_MARGIN, config.TEXT_MARGIN + 100))
            
            # Update the display
            pygame.display.flip()
            
            # Process pygame events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    
            if self.render_mode == 'rgb_array':
                # Convert the Pygame surface to a numpy array
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.game.screen)), 
                    axes=(1, 0, 2)
                )
        
        return None
    
    def close(self) -> None:
        """
        Clean up resources when environment is no longer needed.
        """
        if self.game:
            self.game.cleanup()
