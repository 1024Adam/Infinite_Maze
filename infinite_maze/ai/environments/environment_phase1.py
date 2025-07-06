"""
Enhanced environment for Phase 1 of the Infinite Maze AI training.

This module creates a gymnasium-compatible environment with specific enhancements 
to address the observed issues in training:
1. Rightward bias ("always go right" problem)
2. Low vertical movement utilization
3. Poor score accumulation

Key improvements:
- Strategic maze configurations that require vertical movement
- Stronger rewards for path discovery and vertical navigation
- Comprehensive metrics tracking for bias detection
- Anti-oscillation mechanics
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import random
from typing import Dict, Tuple, Optional, List, Any
from collections import deque

# We'll need to import core game components to leverage the existing maze generation
from infinite_maze.core.game import Game
from infinite_maze.entities.maze import Maze, Line
from infinite_maze.entities.player import Player
from infinite_maze.utils.config import config

class InfiniteMazeEnv(gym.Env):
    """
    Enhanced OpenAI Gym environment for the Infinite Maze game.
    
    This version includes specific improvements to prevent the rightward bias problem
    and promote proper vertical movement and score accumulation.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 training_phase: int = 1,
                 use_maze_from_start: bool = True,
                 pace_enabled: bool = False,
                 pace_speed: float = 1.0,
                 render_mode: Optional[str] = None,
                 grid_size: int = 11,
                 max_steps: int = 10000,
                 vertical_corridor_frequency: float = 0.6,  # NEW: Control frequency of vertical corridors
                 force_strategic_vertical_movement: bool = True):  # NEW: Force scenarios requiring vertical movement
        """
        Initialize the enhanced Infinite Maze training environment.
        
        Args:
            training_phase: The current training phase (1-5)
            use_maze_from_start: Whether to start with maze structures (True for training)
            pace_enabled: Whether the advancing pace line is enabled
            pace_speed: Speed multiplier for the pace line
            render_mode: Mode for visualization ('human', 'rgb_array', or None)
            grid_size: Size of the observation grid (must be odd)
            max_steps: Maximum steps per episode
            vertical_corridor_frequency: Frequency of vertical corridors in maze (0.0-1.0)
            force_strategic_vertical_movement: Whether to create scenarios requiring vertical movement
        """
        super().__init__()
        
        # Store configuration
        self.training_phase = training_phase
        self.use_maze_from_start = use_maze_from_start
        self.pace_enabled = pace_enabled
        self.pace_speed = pace_speed
        self.render_mode = render_mode
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        # Store enhanced configuration
        self.vertical_corridor_frequency = vertical_corridor_frequency
        self.force_strategic_vertical_movement = force_strategic_vertical_movement
        
        # Initialize game components (will be properly set in reset())
        self.config = config  # Use the global config instance
        self.game = None
        self.lines = None
        self.maze = None
        self.player = None
        self.pace_line_x = -1000  # Default value for no pace line
        
        # Track episode stats
        self.steps = 0
        self.total_reward = 0
        self.action_history = deque(maxlen=20)  # Track actions for repetition detection
        self.position_history = deque(maxlen=20)  # Track positions for oscillation detection
        
        # NEW: Track vertical movement statistics
        self.vertical_movements = 0
        self.total_movements = 0
        self.action_counts = [0, 0, 0, 0, 0]  # UP, RIGHT, DOWN, LEFT, NO_ACTION
        
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
        
        # Generate the maze lines with enhanced vertical corridor frequency
        if self.force_strategic_vertical_movement:
            # Create a maze with strategic vertical corridors
            self.lines = self._generate_strategic_maze()
        else:
            # Use standard maze generation with adjusted parameters
            self.lines = Line.generateMaze(self.game, config.MAZE_ROWS, config.MAZE_COLS)
        
        # Create our Maze wrapper to handle AI interactions
        self.maze = Maze(self.game, self.lines)
        
        # Create the player at a valid starting position
        # Enhanced: Use varied starting positions to improve generalization
        start_x_base = config.PLAYER_START_X + 20  # Move slightly to the right of starting line
        start_y_base = config.PLAYER_START_Y
        
        # Enhanced: More varied starting positions for better generalization
        start_positions = [
            (start_x_base, start_y_base),  # Default
            (start_x_base + 15, start_y_base - 20),  # Slightly up-right
            (start_x_base + 10, start_y_base + 25),  # Slightly down-right
            (start_x_base + 30, start_y_base),       # Further right
            (start_x_base + 25, start_y_base - 35),  # Further up-right
            (start_x_base + 25, start_y_base + 35),  # Further down-right
            (start_x_base + 50, start_y_base),       # Much further right
            (start_x_base + 5, start_y_base - 15),   # Slightly up-right, close to start
        ]
        
        # Select a position and verify it's not inside a wall
        max_attempts = 10
        for _ in range(max_attempts):
            start_x, start_y = random.choice(start_positions)
            # Ensure starting position doesn't collide with walls
            if not self.maze.is_wall(start_x, start_y):
                break
                
        # Create the player
        self.player = Player(start_x, start_y, headless=headless)
        
        # Reset episode tracking
        self.steps = 0
        self.total_reward = 0
        self.action_history.clear()
        self.position_history.clear()
        self.collision_count = 0
        self.oscillation_count = 0
        self.consecutive_oscillations = 0
        
        # Reset movement statistics
        self.vertical_movements = 0
        self.total_movements = 0
        self.action_counts = [0, 0, 0, 0, 0]
        
        self.game_over = False
        self.pace_line_x = -1000  # Default position for pace line (disabled in Phase 1)
        
        # Mark initial position as visited
        self.maze.mark_visited(self.player.getX(), self.player.getY())
        
        # Get initial observation
        observation = self._get_observation()
        
        # Initial render to make sure we have a visible game screen
        if self.render_mode == 'human':
            # Process any pending pygame events to prevent freezing
            pygame.event.pump()
            self.render()
            import time
            time.sleep(0.2)
                
        # Return initial observation and empty info
        return observation, {}
    
    def _generate_strategic_maze(self) -> List[Line]:
        """
        Generate a maze with strategic vertical corridors that force vertical movement.
        
        Returns:
            List of Line objects representing the maze
        """
        # Start with standard maze generation
        lines = Line.generateMaze(self.game, config.MAZE_ROWS, config.MAZE_COLS)
        
        # Now add strategic vertical walls to force vertical movement
        x_start = config.PLAYER_START_X + 100  # Start placing strategic walls a bit ahead
        y_mid = config.PLAYER_START_Y
        
        # Create some vertical walls with gaps at specific positions to force navigation
        for i in range(4):  # Create several strategic sections
            x_pos = x_start + i * 150  # Space them out horizontally
            
            # Create a vertical wall with specific gaps to force navigation
            wall_top = Line((x_pos, config.Y_MIN), (x_pos, y_mid - 50))
            wall_bottom = Line((x_pos, y_mid + 50), (x_pos, config.Y_MAX))
            
            # Add these strategic walls
            lines.append(wall_top)
            lines.append(wall_bottom)
            
            # Create horizontal obstacles near these gaps to force more complex navigation
            if random.random() < self.vertical_corridor_frequency:
                # Add a horizontal line near the gap to create a corridor
                corridor_y = y_mid + random.choice([-30, 30])
                corridor_line = Line((x_pos - 50, corridor_y), (x_pos + 50, corridor_y))
                lines.append(corridor_line)
        
        # Create a wall section that requires deliberate up/down navigation
        if random.random() < self.vertical_corridor_frequency:
            x_special = x_start + 75
            # Create a zigzag pattern that forces up then down movement
            zigzag_lines = [
                Line((x_special, y_mid - 100), (x_special + 30, y_mid - 100)),
                Line((x_special + 30, y_mid - 100), (x_special + 30, y_mid)),
                Line((x_special + 30, y_mid), (x_special + 60, y_mid)),
                Line((x_special + 60, y_mid), (x_special + 60, y_mid + 100)),
                Line((x_special + 60, y_mid + 100), (x_special + 90, y_mid + 100))
            ]
            lines.extend(zigzag_lines)
            
        return lines
    
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
        
        # Update action counts
        self.action_counts[action] += 1
        
        # Store the state before the action for reward calculation
        old_state = self._get_state_snapshot()
        old_position = (self.player.getX(), self.player.getY())
        
        # Map the action to the player movement
        blocked = False
        current_x, current_y = self.player.getX(), self.player.getY()
        speed = self.player.getSpeed()
        
        # Store action in history
        self.action_history.append(action)
        
        if action == 0:  # UP
            # Check for collision along the entire movement path
            if not self.maze.check_collision(current_x, current_y, 0, -speed):
                self.player.moveY(-1)  # Move up
                self.vertical_movements += 1
                self.total_movements += 1
            else:
                blocked = True
                
        elif action == 1:  # RIGHT
            # Check for collision along the entire movement path
            if not self.maze.check_collision(current_x, current_y, speed, 0):
                self.player.moveX(1)  # Move right
                self.game.incrementScore()
                self.total_movements += 1
            else:
                blocked = True
                
        elif action == 2:  # DOWN
            # Check for collision along the entire movement path
            if not self.maze.check_collision(current_x, current_y, 0, speed):
                self.player.moveY(1)  # Move down
                self.vertical_movements += 1
                self.total_movements += 1
            else:
                blocked = True
                
        elif action == 3:  # LEFT
            # Check for collision along the entire movement path
            if not self.maze.check_collision(current_x, current_y, -speed, 0):
                self.player.moveX(-1)  # Move left
                self.game.decrementScore()
                self.total_movements += 1
            else:
                blocked = True
        
        # Double-check that we're not inside a wall and correct if needed
        if self.maze.is_wall(self.player.getX(), self.player.getY()):
            # Emergency correction - reset to previous position
            self.player.setX(current_x)
            self.player.setY(current_y)
            blocked = True
        
        # Update game state (handle pace line if enabled)
        if self.training_phase > 1 and self.pace_enabled:
            # Simple pace line simulation
            if self.steps % 10 == 0:
                self.pace_line_x += self.pace_speed
                
                # Check if player is caught by pace line
                if self.player.getX() < self.pace_line_x:
                    self.game_over = True
        
        # Position boundary checks
        if self.player.getX() < config.PLAYER_START_X:
            self.game_over = True
            
        # Mark current position as visited
        self.maze.mark_visited(self.player.getX(), self.player.getY())
        
        # Store position in history
        self.position_history.append((self.player.getX(), self.player.getY()))
        
        # Get the new state after action
        new_state = self._get_state_snapshot()
        
        # Check if episode is done (game over or max steps reached)
        done = self.game_over or self.steps >= self.max_steps
        
        # Analyze path improvement
        path_improved = self._path_improves(old_state, new_state)
        path_clearance = self._calculate_path_clearance(new_state)
        
        # Detect oscillation
        oscillation_detected = self._detect_oscillation()
        if oscillation_detected:
            self.oscillation_count += 1
            self.consecutive_oscillations += 1
        else:
            self.consecutive_oscillations = 0
            
        # Calculate the reward
        reward = self._calculate_enhanced_reward(old_state, action, new_state, done, blocked)
        self.total_reward += reward
            
        # Get the observation after taking the action
        observation = self._get_observation()
        
        # Calculate vertical movement rate
        vertical_movement_rate = self.vertical_movements / max(1, self.total_movements)
        
        # Calculate forward success rate
        right_attempts = self.action_counts[1]
        right_successes = self.game.getScore()  # Score increases by 1 for each successful right move
        forward_success_rate = right_successes / max(1, right_attempts)
        
        # Calculate collision rate
        collision_rate = self.collision_count / max(1, sum(self.action_counts))
        
        # Additional info for monitoring
        info = {
            'score': self.game.getScore(),
            'steps': self.steps,
            'collision': blocked,
            'episode_reward': self.total_reward,
            'vertical_movement_rate': vertical_movement_rate,
            'forward_success_rate': forward_success_rate,
            'collision_rate': collision_rate,
            'action_counts': self.action_counts.copy(),
            'oscillation_detected': oscillation_detected,
            'oscillation_count': self.oscillation_count,
            'path_improved': path_improved,
            'path_clearance': path_clearance
        }
        
        if blocked:
            self.collision_count += 1
        
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
        
        # 2. Current pace level (using pace line speed as a proxy)
        numerical[1] = self.pace_speed if self.pace_enabled else 0
        
        # 3. Time until next pace increase (placeholder for now)
        numerical[2] = 1.0  # Placeholder
        
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
                
                # Calculate pace line proximity
                if self.pace_enabled:
                    distance_to_pace = max(1, x - pace_line_pos)
                    proximity = 1.0 / min(distance_to_pace, 100)
                    grid[i, j, 2] = proximity
        
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
            'game_over': self.game_over
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
        for i in range(1, steps + 1):
            if self.maze.is_wall(x + i * speed, y):
                return False
        return True
    
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
        old_right_clear = not self.maze.is_wall(old_x + self.player.getSpeed(), old_y)
        new_right_clear = not self.maze.is_wall(new_x + self.player.getSpeed(), new_y)
        
        if not old_right_clear and new_right_clear:
            return True
            
        # Check if we can see more open space to the right from new position
        open_spaces_old = self._count_open_spaces_right(old_x, old_y, 5)
        open_spaces_new = self._count_open_spaces_right(new_x, new_y, 5)
        
        return open_spaces_new > open_spaces_old
    
    def _calculate_path_clearance(self, state: Dict[str, Any]) -> float:
        """
        Calculate a metric for the clearance of paths ahead.
        
        Args:
            state: Current state
            
        Returns:
            Path clearance metric (higher is better)
        """
        x, y = state['player_x'], state['player_y']
        speed = self.player.getSpeed()
        clearance = 0.0
        
        # Check forward clearance
        for i in range(1, 6):  # Check 5 steps ahead
            if not self.maze.is_wall(x + i * speed, y):
                clearance += 1.0
                
        # Check diagonal clearance (up-right and down-right)
        for i in range(1, 4):  # Check 3 steps in diagonal directions
            if not self.maze.is_wall(x + i * speed, y - i * speed):  # Up-right
                clearance += 0.5
            if not self.maze.is_wall(x + i * speed, y + i * speed):  # Down-right
                clearance += 0.5
                
        return clearance
    
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
    
    def _results_in_clear_path(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> bool:
        """
        Check if the move resulted in a clearer path forward.
        
        Args:
            old_state: State before action
            new_state: State after action
            
        Returns:
            True if move led to clearer path
        """
        # Simple check: is there now a clear path rightward?
        new_x, new_y = new_state['player_x'], new_state['player_y']
        
        # Check if we have at least 3 steps clear ahead
        steps_clear = 0
        for i in range(1, 6):  # Check 5 steps ahead
            if self.maze.is_wall(new_x + i * self.player.getSpeed(), new_y):
                break
            steps_clear += 1
            
        return steps_clear >= 3
    
    def _calculate_nearest_path_distance(self, state: Dict[str, Any]) -> float:
        """
        Calculate distance to nearest open rightward path.
        
        Args:
            state: Current state
            
        Returns:
            Distance to nearest open path (lower is better)
        """
        x, y = state['player_x'], state['player_y']
        speed = self.player.getSpeed()
        
        # Check vertical distances (up and down) to find open rightward path
        for dist in range(1, 11):  # Check up to 10 cells up and down
            # Check upward
            if not self.maze.is_wall(x + speed, y - dist * speed) and \
               not self.maze.is_wall(x + 2 * speed, y - dist * speed):
                return dist
                
            # Check downward
            if not self.maze.is_wall(x + speed, y + dist * speed) and \
               not self.maze.is_wall(x + 2 * speed, y + dist * speed):
                return dist
                
        return 10.0  # Default high value if no path found
    
    def _not_previously_visited(self, state: Dict[str, Any]) -> bool:
        """
        Check if the position has not been visited before.
        
        Args:
            state: Current state
            
        Returns:
            True if position hasn't been visited
        """
        x, y = state['player_x'], state['player_y']
        grid_x = int(x // self.config.MAZE_CELL_SIZE)
        grid_y = int(y // self.config.MAZE_CELL_SIZE)
        
        return (grid_x, grid_y) not in self.maze.visited
    
    def _discovered_better_path(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> bool:
        """
        Check if the agent discovered a more efficient path.
        
        Args:
            old_state: State before action
            new_state: State after action
            
        Returns:
            True if new viable paths were discovered
        """
        old_x, old_y = old_state['player_x'], old_state['player_y']
        new_x, new_y = new_state['player_x'], new_state['player_y']
        
        # Check if we've moved to a position that has more rightward path options
        old_path_options = self._count_rightward_paths(old_x, old_y)
        new_path_options = self._count_rightward_paths(new_x, new_y)
        
        return new_path_options > old_path_options
    
    def _count_rightward_paths(self, x: float, y: float) -> int:
        """
        Count possible rightward paths from current position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Number of possible rightward paths
        """
        count = 0
        speed = self.player.getSpeed()
        
        # Check for paths at different vertical offsets
        for offset in range(-3, 4):
            if not self.maze.is_wall(x + speed, y + offset * speed) and \
               not self.maze.is_wall(x + 2 * speed, y + offset * speed):
                count += 1
                
        return count
    
    def _detect_oscillation(self) -> bool:
        """
        Detect oscillation patterns in recent actions and positions.
        
        Returns:
            Boolean indicating whether oscillation is detected
        """
        # Need enough history to detect patterns
        if len(self.action_history) < 6 or len(self.position_history) < 6:
            return False
            
        # Method 1: Check for alternating vertical actions (UP/DOWN pattern)
        recent_actions = list(self.action_history)
        vertical_actions = [a for a in recent_actions[-6:] if a in [0, 2]]  # UP(0) and DOWN(2)
        
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
                if len(recent_positions) >= 6:
                    horizontal_progress = abs(recent_positions[-1][0] - recent_positions[-6][0])
                    if horizontal_progress < self.player.getSpeed() * 2:
                        return True
                    
        # Method 2: Position-based oscillation detection
        recent_positions = list(self.position_history)
        if len(recent_positions) >= 8:
            # Extract y positions
            y_positions = [pos[1] for pos in recent_positions[-8:]]
            
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
                x_progress = abs(recent_positions[-1][0] - recent_positions[-8][0])
                if x_progress < self.player.getSpeed() * 3:
                    return True
        
        return False
    
    def _is_near_vertical_wall(self, state: Dict[str, Any]) -> bool:
        """
        Check if the player is near a vertical wall.
        
        Args:
            state: Current state
            
        Returns:
            True if near vertical wall
        """
        x, y = state['player_x'], state['player_y']
        speed = self.player.getSpeed()
        
        # Check for walls to the right
        for i in range(1, 3):  # Check 2 steps ahead
            if self.maze.is_wall(x + i * speed, y):
                return True
                
        return False
    
    def _calculate_enhanced_reward(self, old_state: Dict[str, Any], action: int, 
                                 new_state: Dict[str, Any], done: bool, collision: bool) -> float:
        """
        Calculate enhanced reward structure to address rightward bias and improve vertical movement.
        
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
        
        # Check if movement occurred and in which direction
        moved_right = new_state['player_x'] > old_state['player_x']
        moved_left = new_state['player_x'] < old_state['player_x']
        moved_up = new_state['player_y'] < old_state['player_y']
        moved_down = new_state['player_y'] > old_state['player_y']
        
        # Get current action distribution to detect and correct bias
        total_actions = sum(self.action_counts)
        if total_actions > 0:
            right_action_percentage = self.action_counts[1] / total_actions
            vertical_action_percentage = (self.action_counts[0] + self.action_counts[2]) / total_actions
        else:
            right_action_percentage = 0
            vertical_action_percentage = 0
        
        # REWARD STRUCTURE ENHANCEMENT #1: Balanced rightward movement rewards
        if moved_right and not collision:
            # Base reward for rightward movement
            base_right_reward = 1.5
            
            # Check if rightward movement is strategic (has open path ahead)
            if self._path_is_open_ahead(new_state, 3):
                path_bonus = 1.0
            else:
                path_bonus = 0.3
                
            # Apply anti-bias reduction if too much rightward movement
            if right_action_percentage > 0.6:
                # Apply progressive reduction based on bias level
                bias_reduction = min(0.8, (right_action_percentage - 0.6) * 2)
                reward += base_right_reward * (1 - bias_reduction) + path_bonus
            else:
                # Normal reward for balanced behavior
                reward += base_right_reward + path_bonus
                
        # Left movement penalty (slightly reduced to allow strategic left moves)
        elif moved_left:
            reward -= 0.7  # Less penalty than before (was -1.0)
            
            # Check if leftward movement is strategic (enables better path)
            if self._path_improves(old_state, new_state):
                # Reduce the penalty if it's strategic
                reward += 0.4
        
        # REWARD STRUCTURE ENHANCEMENT #2: Much stronger vertical movement rewards
        if moved_up or moved_down:
            # Base reward for any vertical movement (was 0.2)
            vertical_base = 0.3
            
            # Calculate strategic value of vertical movement
            if self._path_improves(old_state, new_state):
                # Strongly reward beneficial vertical movement
                strategic_value = 2.0  # Doubled from previous value
            elif self._results_in_clear_path(old_state, new_state):
                strategic_value = 1.5  # Increased substantially
            else:
                strategic_value = 0.5  # Increased base value for any vertical movement
            
            # Apply anti-bias boost if we need more vertical movement
            if vertical_action_percentage < 0.15:  # Below target range
                # Boost increases as we get further from target
                vertical_boost = 1.0 + (0.15 - vertical_action_percentage) * 5
                reward += vertical_base + strategic_value * vertical_boost
            else:
                reward += vertical_base + strategic_value
        
        # REWARD STRUCTURE ENHANCEMENT #3: Pathfinding rewards
        # Moving closer to open paths
        nearest_path_distance_old = self._calculate_nearest_path_distance(old_state)
        nearest_path_distance_new = self._calculate_nearest_path_distance(new_state)
        if nearest_path_distance_new < nearest_path_distance_old:
            # Moving closer to open paths is good (increased from 0.4)
            reward += 0.6
        
        # Exploration bonus for finding new areas (useful early in training)
        if self.steps < 1500 and self._not_previously_visited(new_state):  # Extended window
            reward += 0.3  # Increased from 0.2
        
        # REWARD STRUCTURE ENHANCEMENT #4: Score-based rewards
        # Survival incentive (slightly increased)
        reward += 0.08  # Was 0.05
        
        # Score-based bonus to emphasize score accumulation
        score_diff = new_state['score'] - old_state['score']
        if score_diff > 0:
            # Much stronger reward for score increases
            reward += 3.0 * score_diff
            
            # Milestone bonuses
            current_score = new_state['score']
            if current_score > 0 and current_score % 20 == 0:  # More frequent milestones
                reward += 5.0  # Significant milestone bonus
        
        # Collision penalty (balancing strictness with learning capability)
        if collision:
            # Base collision penalty
            collision_penalty = 1.0  # Decreased from 1.5
            
            # Reduce penalty for early exploratory collisions
            if self.steps < 500 and self.collision_count < 15:
                collision_penalty *= 0.7
                
            reward -= collision_penalty
        
        # REWARD STRUCTURE ENHANCEMENT #5: Anti-oscillation penalty
        if self._detect_oscillation():
            # Apply significant penalty to discourage oscillation
            oscillation_penalty = 0.8
            
            # Escalate penalty for persistent oscillation
            if hasattr(self, 'consecutive_oscillations') and self.consecutive_oscillations > 3:
                oscillation_penalty += 0.3 * (self.consecutive_oscillations - 3)
                
            reward -= oscillation_penalty
        
        # REWARD STRUCTURE ENHANCEMENT #6: Path discovery and strategic movement
        # Path discovery bonus (increased)
        if self._discovered_better_path(old_state, new_state):
            reward += 1.2  # Increased from 0.7
        
        # Check if near a vertical wall that requires navigation
        if self._is_near_vertical_wall(old_state):
            # Reward vertical movement near walls
            if (moved_up or moved_down) and self._path_improves(old_state, new_state):
                reward += 1.5  # Strong incentive for strategic navigation around walls
        
        # REWARD STRUCTURE ENHANCEMENT #7: Significant terminal and milestone rewards
        # Terminal state penalty
        if done and new_state['game_over']:
            reward -= 10.0  # Slightly reduced from 15.0
            
        # Periodic milestone bonus based on survival and progress
        if self.steps % 50 == 0:  # More frequent milestone bonuses
            reward += 3.0  # Smaller but more frequent milestone rewards
            
        # Larger milestone every 200 steps
        if self.steps % 200 == 0:
            reward += 8.0
        
        # Major milestone for achieving target score ranges
        if self.game.getScore() >= 100 and self.steps % 100 == 0:
            reward += 10.0
            
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
            
            # Display score and step info
            if self.game.font:
                score_text = self.game.font.render(
                    f"Score: {self.game.getScore()}", 1, self.game.FG_COLOR
                )
                self.game.screen.blit(score_text, (config.TEXT_MARGIN, config.TEXT_MARGIN))
                
                steps_text = self.game.font.render(
                    f"Steps: {self.steps}", 1, self.game.FG_COLOR
                )
                self.game.screen.blit(steps_text, (config.TEXT_MARGIN, config.TEXT_MARGIN + 25))
                
                # Add debug info about player position
                pos_text = self.game.font.render(
                    f"Pos: ({self.player.getX():.1f}, {self.player.getY():.1f})", 1, self.game.FG_COLOR
                )
                self.game.screen.blit(pos_text, (config.TEXT_MARGIN, config.TEXT_MARGIN + 50))
                
                # Show vertical movement rate
                vert_rate = self.vertical_movements / max(1, self.total_movements)
                vert_text = self.game.font.render(
                    f"Vertical: {vert_rate:.1%} (Target: 15-25%)", 1, self.game.FG_COLOR
                )
                self.game.screen.blit(vert_text, (config.TEXT_MARGIN, config.TEXT_MARGIN + 75))
                
                # Show last action
                action_names = ["UP", "RIGHT", "DOWN", "LEFT", "NONE"]
                action_text = "Last: " + (action_names[self.action_history[-1]] if self.action_history else "None")
                action_display = self.game.font.render(action_text, 1, self.game.FG_COLOR)
                self.game.screen.blit(action_display, (config.TEXT_MARGIN, config.TEXT_MARGIN + 100))
            
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
