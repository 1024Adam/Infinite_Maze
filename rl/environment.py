"""
Reinforcement Learning Environment for Infinite Maze Game

This module provides a Gym-compatible environment wrapper for the Infinite Maze game,
allowing RL agents to interact with the game through standard RL interfaces.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pygame.locals import *
import time
import sys
import os
from typing import Dict, Any, Tuple, Optional

# Add the parent directory to sys.path to import infinite_maze modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infinite_maze.Game import Game
from infinite_maze.Player import Player
from infinite_maze.Line import Line

# Action constants
DO_NOTHING = 0
RIGHT = 1
LEFT = 2
UP = 3
DOWN = 4

class InfiniteMazeEnv(gym.Env):
    """
    Custom Gym environment for the Infinite Maze game.
    
    State Space:
    - Player position (x, y)
    - Player velocity
    - Game pace
    - Current score
    - Local maze structure around player
    - Distance to nearest walls in each direction
    - Time survived
    
    Action Space:
    - 0: DO_NOTHING
    - 1: RIGHT
    - 2: LEFT  
    - 3: UP
    - 4: DOWN
    
    Reward Function:
    - +1 for moving right (score increase)
    - -1 for moving left (score decrease)
    - -100 for collision/death
    - +0.1 for surviving each step
    - Bonus for staying ahead of pace
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(self, render_mode: Optional[str] = None, headless: bool = False):
        super().__init__()
        
        self.render_mode = render_mode
        self.headless = headless
        
        # Initialize pygame if not headless
        if not self.headless:
            pygame.init()
            
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        # Observation space
        # [player_x, player_y, pace, score, time_survived, 
        #  wall_distances (4 directions), local_maze_structure (grid)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1000, 0, 0, 0, 0, 0] + [0]*100),  # 100 for local maze grid
            high=np.array([1000, 500, 100, 10000, 10000, 1000, 1000, 1000, 1000] + [1]*100),
            dtype=np.float32
        )
        
        # Game components
        self.game = None
        self.player = None
        self.lines = None
        
        # Episode tracking
        self.episode_length = 0
        self.max_episode_length = 10000
        
        # State tracking
        self.last_score = 0
        self.survival_time = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize game components
        self.game = Game(headless=self.headless)
        self.player = Player(80, 223, headless=self.headless)
        self.lines = Line.generateMaze(self.game, 15, 20)
        
        # Reset tracking variables
        self.episode_length = 0
        self.last_score = 0
        self.survival_time = 0
        
        # Reset navigation tracking
        self._consecutive_failed_right_moves = 0
        self._last_rightward_progress = 0
        self._position_history = []
        
        # Reset game clock
        self.game.getClock().reset()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.episode_length += 1
        self.survival_time += 1
        
        # Store previous state for reward calculation
        prev_score = self.game.getScore()
        prev_x = self.player.getX()
        
        # Execute action
        reward = self._execute_action(action)
        
        # Update game mechanics
        self._update_game_mechanics()
        
        # Check if game is over
        terminated = not self.game.isActive()
        truncated = self.episode_length >= self.max_episode_length
        
        # Calculate additional rewards
        if not terminated:
            # Survival bonus
            reward += 0.1
            
            # Pace management reward
            distance_from_left = self.player.getX() - self.game.X_MIN
            if distance_from_left > 50:  # Safe distance from pace
                reward += 0.05
            elif distance_from_left < 20:  # Dangerous proximity
                reward -= 0.1
            
            # Navigation intelligence reward
            nav_reward = self._calculate_navigation_reward()
            reward += nav_reward
                
        else:
            # Death penalty
            reward -= 100
            
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> float:
        """Execute the given action and return immediate reward."""
        reward = 0.0
        blocked = False
        
        # Store previous position for progress calculation
        prev_x = self.player.getX()
        prev_y = self.player.getY()
        
        if action == RIGHT:
            blocked = self._check_collision_right()
            if not blocked:
                self.player.moveX(self.player.getSpeed())
                self.game.incrementScore()
                # Reduced base reward - progress bonus will be added later
                reward += 0.2
                # Reset failed moves counter
                self._consecutive_failed_right_moves = 0
            else:
                # Penalty for trying to move into a wall
                reward -= 0.15
                # Track consecutive failed right moves
                if not hasattr(self, '_consecutive_failed_right_moves'):
                    self._consecutive_failed_right_moves = 0
                self._consecutive_failed_right_moves += 1
                
        elif action == LEFT:
            blocked = self._check_collision_left()
            if not blocked:
                self.player.moveX(-self.player.getSpeed())
                self.game.decrementScore()
                # Higher penalty for leftward movement to discourage oscillation
                reward -= 0.8
            else:
                reward -= 0.15
                
        elif action == DOWN:
            blocked = self._check_collision_down()
            if not blocked:
                self.player.moveY(self.player.getSpeed())
                # Small reward for successful vertical movement
                reward += 0.1
            else:
                reward -= 0.1
                
        elif action == UP:
            blocked = self._check_collision_up()
            if not blocked:
                self.player.moveY(-self.player.getSpeed())
                # Small reward for successful vertical movement
                reward += 0.1
            else:
                reward -= 0.1
                
        # DO_NOTHING gets small penalty to encourage action
        elif action == DO_NOTHING:
            reward -= 0.05
        
        # Enhanced progress tracking
        current_x = self.player.getX()
        current_y = self.player.getY()
        
        # Reward for net rightward progress
        if current_x > prev_x:
            progress = (current_x - prev_x) / self.player.getSpeed()
            reward += 1.2 * progress  # Increased progress bonus
        
        # Track overall position for oscillation detection
        if not hasattr(self, '_position_history'):
            self._position_history = []
        
        self._position_history.append((current_x, current_y))
        
        # Keep only recent history
        if len(self._position_history) > 20:
            self._position_history.pop(0)
        
        # Penalty for oscillating behavior
        if len(self._position_history) >= 10:
            oscillation_penalty = self._detect_oscillation()
            reward -= oscillation_penalty
        
        # Penalty for being stuck against a wall trying to go right
        if action == RIGHT and blocked:
            if not hasattr(self, '_last_rightward_progress'):
                self._last_rightward_progress = 0
            
            self._last_rightward_progress += 1
            if self._last_rightward_progress > 8:  # Reduced threshold
                # Stronger incentive to try vertical movement
                reward -= 0.8
        else:
            if hasattr(self, '_last_rightward_progress'):
                self._last_rightward_progress = 0
        
        return reward
    
    def _detect_oscillation(self) -> float:
        """Detect oscillating behavior and return penalty."""
        if len(self._position_history) < 10:
            return 0.0
        
        # Check for repeated back-and-forth movement
        recent_x_positions = [pos[0] for pos in self._position_history[-10:]]
        
        # Count direction changes
        direction_changes = 0
        for i in range(1, len(recent_x_positions)):
            if i < len(recent_x_positions) - 1:
                # Check if direction changed
                prev_diff = recent_x_positions[i] - recent_x_positions[i-1]
                next_diff = recent_x_positions[i+1] - recent_x_positions[i]
                
                if (prev_diff > 0 and next_diff < 0) or (prev_diff < 0 and next_diff > 0):
                    direction_changes += 1
        
        # Penalty for excessive direction changes (oscillation)
        if direction_changes > 4:  # More than 4 direction changes in 10 steps
            return 0.5 * (direction_changes - 4)
        
        # Check for minimal net progress
        net_progress = recent_x_positions[-1] - recent_x_positions[0]
        if abs(net_progress) < 5:  # Very little net movement
            return 0.3
        
        return 0.0
    
    def _calculate_navigation_reward(self) -> float:
        """Calculate reward for intelligent navigation behavior."""
        nav_reward = 0.0
        
        # Check if rightward movement is blocked
        right_blocked = self._check_collision_right()
        
        # Check if there are clear vertical paths
        up_blocked = self._check_collision_up()
        down_blocked = self._check_collision_down()
        
        # If right is blocked but vertical movement is available, 
        # encourage exploration of vertical movement
        if right_blocked and (not up_blocked or not down_blocked):
            # Look ahead to see if vertical movement could lead to rightward progress
            if self._can_vertical_movement_help():
                nav_reward += 0.4  # Increased reward for considering navigation around obstacles
        
        # Reward finding alternative paths when stuck
        if hasattr(self, '_consecutive_failed_right_moves'):
            if self._consecutive_failed_right_moves > 3:  # Reduced threshold
                # If we've been stuck going right, reward vertical exploration
                if not up_blocked:
                    nav_reward += 0.3
                if not down_blocked:
                    nav_reward += 0.3
        
        # Bonus for making forward progress consistently
        if hasattr(self, '_position_history') and len(self._position_history) >= 5:
            recent_x = [pos[0] for pos in self._position_history[-5:]]
            if len(recent_x) >= 2:
                net_progress = recent_x[-1] - recent_x[0]
                if net_progress > 10:  # Good forward progress
                    nav_reward += 0.2
        
        return nav_reward
    
    def _can_vertical_movement_help(self) -> bool:
        """Check if vertical movement could eventually lead to rightward progress."""
        # Simple heuristic: look a few steps ahead in vertical directions
        player_x = self.player.getX()
        player_y = self.player.getY()
        speed = self.player.getSpeed()
        
        # Check up direction
        for steps in range(1, 5):  # Look up to 4 steps ahead
            test_y = player_y - (speed * steps)
            # Check if we can move right from this vertical position
            temp_y = self.player.getY()
            self.player.setY(test_y)
            can_move_right = not self._check_collision_right()
            self.player.setY(temp_y)  # Restore original position
            
            if can_move_right:
                return True
        
        # Check down direction
        for steps in range(1, 5):
            test_y = player_y + (speed * steps)
            temp_y = self.player.getY()
            self.player.setY(test_y)
            can_move_right = not self._check_collision_right()
            self.player.setY(temp_y)
            
            if can_move_right:
                return True
        
        return False
    
    def _check_collision_right(self) -> bool:
        """Check if moving right would cause a collision."""
        for line in self.lines:
            if line.getIsHorizontal():
                if (self.player.getY() <= line.getYStart() and
                    self.player.getY() + self.player.getHeight() >= line.getYStart() and
                    self.player.getX() + self.player.getWidth() + self.player.getSpeed() >= line.getXStart() and
                    self.player.getX() + self.player.getWidth() <= line.getXStart()):
                    return True
            else:  # vertical line
                if (self.player.getX() + self.player.getWidth() <= line.getXStart() and
                    self.player.getX() + self.player.getWidth() + self.player.getSpeed() >= line.getXStart() and
                    ((self.player.getY() >= line.getYStart() and self.player.getY() <= line.getYEnd()) or
                     (self.player.getY() + self.player.getHeight() >= line.getYStart() and 
                      self.player.getY() + self.player.getHeight() <= line.getYEnd()))):
                    return True
        return False
    
    def _check_collision_left(self) -> bool:
        """Check if moving left would cause a collision."""
        for line in self.lines:
            if line.getIsHorizontal():
                if (self.player.getY() <= line.getYStart() and
                    self.player.getY() + self.player.getHeight() >= line.getYStart() and
                    self.player.getX() - self.player.getSpeed() <= line.getXEnd() and
                    self.player.getX() >= line.getXEnd()):
                    return True
            else:  # vertical line
                if (self.player.getX() >= line.getXEnd() and
                    self.player.getX() - self.player.getSpeed() <= line.getXEnd() and
                    ((self.player.getY() >= line.getYStart() and self.player.getY() <= line.getYEnd()) or
                     (self.player.getY() + self.player.getHeight() >= line.getYStart() and 
                      self.player.getY() + self.player.getHeight() <= line.getYEnd()))):
                    return True
        return False
    
    def _check_collision_down(self) -> bool:
        """Check if moving down would cause a collision."""
        for line in self.lines:
            if line.getIsHorizontal():
                if (self.player.getY() + self.player.getHeight() <= line.getYStart() and
                    self.player.getY() + self.player.getHeight() + self.player.getSpeed() >= line.getYStart() and
                    ((self.player.getX() >= line.getXStart() and self.player.getX() <= line.getXEnd()) or
                     (self.player.getX() + self.player.getWidth() >= line.getXStart() and 
                      self.player.getX() + self.player.getWidth() <= line.getXEnd()))):
                    return True
            else:  # vertical line
                if (self.player.getX() <= line.getXStart() and
                    self.player.getX() + self.player.getWidth() >= line.getXStart() and
                    self.player.getY() + self.player.getHeight() + self.player.getSpeed() >= line.getYStart() and
                    self.player.getY() + self.player.getHeight() <= line.getYStart()):
                    return True
        return False
    
    def _check_collision_up(self) -> bool:
        """Check if moving up would cause a collision."""
        for line in self.lines:
            if line.getIsHorizontal():
                if (self.player.getY() >= line.getYStart() and
                    self.player.getY() - self.player.getSpeed() <= line.getYStart() and
                    ((self.player.getX() >= line.getXStart() and self.player.getX() <= line.getXEnd()) or
                     (self.player.getX() + self.player.getWidth() >= line.getXStart() and 
                      self.player.getX() + self.player.getWidth() <= line.getXEnd()))):
                    return True
            else:  # vertical line
                if (self.player.getX() <= line.getXStart() and
                    self.player.getX() + self.player.getWidth() >= line.getXStart() and
                    self.player.getY() - self.player.getSpeed() <= line.getYEnd() and
                    self.player.getY() >= line.getYEnd()):
                    return True
        return False
    
    def _update_game_mechanics(self):
        """Update game mechanics like pace and line positioning."""
        # Process game pace adjustments
        if self.game.getClock().getTicks() % 10 == 0:
            self.player.setX(self.player.getX() - self.game.getPace())
            for line in self.lines:
                line.setXStart(line.getXStart() - self.game.getPace())
                line.setXEnd(line.getXEnd() - self.game.getPace())
        
        # Update pace every 30 seconds (simplified)
        if self.survival_time % 1800 == 0 and self.survival_time > 0:  # 30 seconds at 60 FPS
            self.game.setPace(self.game.getPace() + 1)
        
        # Position adjustments
        if self.player.getX() < self.game.X_MIN:
            self.game.end()
            
        if self.player.getX() > self.game.X_MAX:
            self.player.setX(self.game.X_MAX)
            for line in self.lines:
                line.setXStart(line.getXStart() - self.player.getSpeed())
                line.setXEnd(line.getXEnd() - self.player.getSpeed())
                
        # Constrain player within vertical bounds
        self.player.setY(max(self.player.getY(), self.game.Y_MIN))
        self.player.setY(min(self.player.getY(), self.game.Y_MAX))
        
        # Reposition lines that have been passed
        xMax = Line.getXMax(self.lines)
        for line in self.lines:
            start = line.getXStart()
            end = line.getXEnd()
            if start < 80:
                line.setXStart(xMax)
                if start == end:
                    line.setXEnd(xMax)
                else:
                    line.setXEnd(xMax + 22)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        # Basic game state
        player_x = self.player.getX()
        player_y = self.player.getY()
        pace = self.game.getPace()
        score = self.game.getScore()
        time_survived = self.survival_time
        
        # Calculate wall distances in each direction
        wall_distances = self._get_wall_distances()
        
        # Get local maze structure around player
        local_maze = self._get_local_maze_structure()
        
        # Combine all features
        observation = np.array([
            player_x / 1000.0,  # Normalize positions
            player_y / 500.0,
            pace / 100.0,
            score / 1000.0,  # Normalize score
            time_survived / 10000.0,  # Normalize time
            wall_distances[0] / 1000.0,  # right
            wall_distances[1] / 1000.0,  # left
            wall_distances[2] / 1000.0,  # down
            wall_distances[3] / 1000.0,  # up
            *local_maze
        ], dtype=np.float32)
        
        return observation
    
    def _get_wall_distances(self) -> np.ndarray:
        """Calculate distance to nearest wall in each direction."""
        distances = [1000.0, 1000.0, 1000.0, 1000.0]  # right, left, down, up
        
        player_x = self.player.getX()
        player_y = self.player.getY()
        player_w = self.player.getWidth()
        player_h = self.player.getHeight()
        
        for line in self.lines:
            if line.getIsHorizontal():
                # Check vertical distances (up/down)
                if (player_x < line.getXEnd() and player_x + player_w > line.getXStart()):
                    if line.getYStart() > player_y + player_h:  # Wall below
                        distances[2] = min(distances[2], line.getYStart() - (player_y + player_h))
                    elif line.getYStart() < player_y:  # Wall above
                        distances[3] = min(distances[3], player_y - line.getYStart())
            else:
                # Check horizontal distances (left/right)
                if (player_y < line.getYEnd() and player_y + player_h > line.getYStart()):
                    if line.getXStart() > player_x + player_w:  # Wall to the right
                        distances[0] = min(distances[0], line.getXStart() - (player_x + player_w))
                    elif line.getXStart() < player_x:  # Wall to the left
                        distances[1] = min(distances[1], player_x - line.getXStart())
        
        return np.array(distances)
    
    def _get_local_maze_structure(self) -> np.ndarray:
        """Get a local grid representation of the maze around the player."""
        grid_size = 10  # 10x10 grid around player
        grid = np.zeros(grid_size * grid_size)
        
        player_x = self.player.getX()
        player_y = self.player.getY()
        
        # Define grid bounds
        cell_size = 22  # Based on maze generation
        start_x = player_x - (grid_size // 2) * cell_size
        start_y = player_y - (grid_size // 2) * cell_size
        
        # Check each grid cell for walls
        for i in range(grid_size):
            for j in range(grid_size):
                cell_x = start_x + i * cell_size
                cell_y = start_y + j * cell_size
                
                # Check if there's a wall in this cell
                has_wall = False
                for line in self.lines:
                    if (line.getXStart() >= cell_x and line.getXStart() <= cell_x + cell_size and
                        line.getYStart() >= cell_y and line.getYStart() <= cell_y + cell_size):
                        has_wall = True
                        break
                
                grid[i * grid_size + j] = 1.0 if has_wall else 0.0
        
        return grid
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state."""
        return {
            'score': self.game.getScore(),
            'pace': self.game.getPace(),
            'survival_time': self.survival_time,
            'player_position': (self.player.getX(), self.player.getY()),
            'episode_length': self.episode_length
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human' and not self.headless:
            if self.game:
                self.game.updateScreen(self.player, self.lines)
                pygame.display.flip()
        elif self.render_mode == 'rgb_array':
            # Return RGB array for recording
            if not self.headless and self.game:
                surface = self.game.getScreen()
                return pygame.surfarray.array3d(surface)
    
    def close(self):
        """Clean up the environment."""
        if self.game:
            self.game.cleanup()
        if not self.headless:
            pygame.quit()


class InfiniteMazeWrapper:
    """Legacy wrapper for compatibility with the existing controlled_run function."""
    
    def __init__(self, model=None):
        self.model = model
        self.env = InfiniteMazeEnv(headless=True)
        self.observation = None
        
    def control(self, values: Dict[str, Any]) -> int:
        """Called by controlled_run to get next action."""
        if self.model is None:
            # Random baseline
            return np.random.randint(0, 5)
        
        # Use the RL model to predict action
        if self.observation is not None:
            action, _ = self.model.predict(self.observation, deterministic=True)
            return int(action)
        else:
            return DO_NOTHING
    
    def gameover(self, final_score: int):
        """Called when game ends."""
        print(f"Game over! Final score: {final_score}")
        
    def update_observation(self, observation):
        """Update the current observation."""
        self.observation = observation
