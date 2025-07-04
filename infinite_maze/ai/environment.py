"""
Training environment for the Infinite Maze AI.

This module creates a gymnasium-compatible environment for training the AI agent,
with specific modifications for the training phases as outlined in the training plan.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import random
from typing import Dict, Tuple, Optional, List, Any

# We'll need to import core game components to leverage the existing maze generation
from infinite_maze.core.game import Game
from infinite_maze.entities.maze import Maze, Line
from infinite_maze.entities.player import Player
from infinite_maze.utils.config import config

class InfiniteMazeEnv(gym.Env):
    """
    OpenAI Gym environment for the Infinite Maze game.
    
    This environment is specifically designed for training the AI agent through
    curriculum learning, with modifications for each training phase.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 training_phase: int = 1,
                 use_maze_from_start: bool = True,
                 pace_enabled: bool = False,
                 pace_speed: float = 1.0,
                 render_mode: Optional[str] = None,
                 grid_size: int = 11,
                 max_steps: int = 10000):
        """
        Initialize the Infinite Maze training environment.
        
        Args:
            training_phase: The current training phase (1-5)
            use_maze_from_start: Whether to start with maze structures (True for training)
            pace_enabled: Whether the advancing pace line is enabled
            pace_speed: Speed multiplier for the pace line
            render_mode: Mode for visualization ('human', 'rgb_array', or None)
            grid_size: Size of the observation grid (must be odd)
            max_steps: Maximum steps per episode
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
        self.action_history = []  # Track actions for repetition detection
        
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
        # For training, we modify the standard game to include maze structures from start
        
        # Phase 1 specific modifications: start with maze structures
        # but disable pace line for initial training
        headless = (self.render_mode is None)
        self.game = Game(headless=headless)
        
        # Create the player at a valid starting position
        start_x = config.PLAYER_START_X + 20  # Move slightly to the right of starting line
        start_y = config.PLAYER_START_Y
        
        # Create the player
        self.player = Player(start_x, start_y, headless=headless)
        
        # Generate the maze lines
        self.lines = Line.generateMaze(self.game, config.MAZE_ROWS, config.MAZE_COLS)
        
        # Create our Maze wrapper to handle AI interactions
        self.maze = Maze(self.game, self.lines)
        
        # Reset episode tracking
        self.steps = 0
        self.total_reward = 0
        self.action_history = []
        self.game_over = False
        self.pace_line_x = -1000  # Default position for pace line (disabled in Phase 1)
        
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
        
        # Get the new state after action
        new_state = self._get_state_snapshot()
        
        # Check if episode is done (game over or max steps reached)
        done = self.game_over or self.steps >= self.max_steps
        
        # Calculate the reward
        reward = self._calculate_reward(old_state, action, new_state, done, blocked)
        self.total_reward += reward
        
        # Keep track of action history for repetition detection
        self.action_history.append(action)
        if len(self.action_history) > 10:  # Keep only recent history
            self.action_history.pop(0)
            
        # Get the observation after taking the action
        observation = self._get_observation()
        
        # Additional info for monitoring
        info = {
            'score': self.game.getScore(),
            'steps': self.steps,
            'collision': blocked,
            'episode_reward': self.total_reward
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
    
    def _detected_collision(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> bool:
        """
        Detect if a collision occurred between old and new state.
        
        Args:
            old_state: State before action
            new_state: State after action
            
        Returns:
            True if a collision was detected
        """
        # If position didn't change after a move, likely a collision
        return (old_state['player_x'] == new_state['player_x'] and 
                old_state['player_y'] == new_state['player_y'])
    
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
    
    def _calculate_reward(self, old_state: Dict[str, Any], action: int, 
                         new_state: Dict[str, Any], done: bool, collision: bool) -> float:
        """
        Calculate the reward based on the action and resulting state change.
        
        Uses a balanced reward function with anti-bias measures as described in the training plan.
        
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
        
        # Base movement rewards - modified to prevent "always go right" bias
        if moved_right and not collision:
            # Check if the rightward movement was beneficial (not into a wall soon)
            if self._path_is_open_ahead(new_state, 3):
                reward += 1.0
            else:
                reward += 0.3  # Reduced reward for suboptimal rightward movement
        elif moved_left:
            reward -= 1.0
        
        # Strategic vertical movement rewards
        if (moved_up or moved_down):
            # Simple path improvement check: reward vertical movement that doesn't lead to collision
            if not collision:
                reward += 0.5
        
        # Small reward for surviving each step
        reward += 0.05
        
        # Penalties
        if collision:
            reward -= 0.8  # Significant collision penalty
        
        # Repeated action penalty (discourage mindless direction holding)
        if len(self.action_history) >= 5:
            same_action_count = 0
            last_action = self.action_history[-1]
            for a in reversed(self.action_history):
                if a == last_action:
                    same_action_count += 1
                else:
                    break
            
            if same_action_count > 5:
                reward -= 0.1 * (same_action_count - 5)
        
        # Terminal state penalty
        if done and new_state['game_over']:
            reward -= 10.0
        
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
                
                # Add debug info about training status
                action_text = "Last action: " + str(self.action_history[-1] if self.action_history else "None")
                status_text = self.game.font.render(action_text, 1, self.game.FG_COLOR)
                self.game.screen.blit(status_text, (config.TEXT_MARGIN, config.TEXT_MARGIN + 75))
            
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
