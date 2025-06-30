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
        
        if action == RIGHT:
            blocked = self._check_collision_right()
            if not blocked:
                self.player.moveX(self.player.getSpeed())
                self.game.incrementScore()
                reward += 1.0  # Reward for rightward movement
                
        elif action == LEFT:
            blocked = self._check_collision_left()
            if not blocked:
                self.player.moveX(-self.player.getSpeed())
                self.game.decrementScore()
                reward -= 1.0  # Penalty for leftward movement
                
        elif action == DOWN:
            blocked = self._check_collision_down()
            if not blocked:
                self.player.moveY(self.player.getSpeed())
                
        elif action == UP:
            blocked = self._check_collision_up()
            if not blocked:
                self.player.moveY(-self.player.getSpeed())
                
        # DO_NOTHING (action == 0) requires no movement
        
        return reward
    
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
