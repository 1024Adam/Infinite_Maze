"""
Neural network architecture for the Infinite Maze AI.

This module implements the model architecture specified in the training plan,
with a CNN backbone for processing the grid representation and dense layers
for integrating numerical features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any

class MazeNavModel(nn.Module):
    """
    Deep Q-Network model for Infinite Maze navigation.
    
    The architecture follows the training plan design:
    - CNN backbone for processing the spatial grid
    - Dense layers for integrating numerical features
    - Output layer with one neuron per action
    """
    
    def __init__(self, grid_size: int = 11, channels: int = 4, num_actions: int = 5):
        """
        Initialize the model architecture.
        
        Args:
            grid_size: Size of the grid observation (default: 11)
            channels: Number of channels in the grid observation (default: 4)
            num_actions: Number of possible actions (default: 5)
        """
        super(MazeNavModel, self).__init__()
        
        # CNN Backbone for processing spatial information
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate the flattened size after convolutions
        self.conv_output_size = 64 * grid_size * grid_size
        
        # Dense layers for CNN output processing
        self.fc1 = nn.Linear(self.conv_output_size, 256)
        
        # Dense layer for numerical features processing
        self.numerical_fc = nn.Linear(11, 64)  # 11 numerical features
        
        # Combined processing of CNN and numerical features
        self.combined_fc1 = nn.Linear(256 + 64, 128)
        self.combined_fc2 = nn.Linear(128, 128)
        
        # Output layer - one neuron per action
        self.output = nn.Linear(128, num_actions)
        
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Dictionary with 'grid' and 'numerical' tensors
                - grid: [batch_size, channels, grid_size, grid_size]
                - numerical: [batch_size, 11]
                
        Returns:
            Q-values for each action [batch_size, num_actions]
        """
        # Process grid through CNN
        grid = x['grid']  # [batch_size, grid_size, grid_size, channels]
        grid = grid.permute(0, 3, 1, 2)  # [batch_size, channels, grid_size, grid_size]
        
        # Apply convolutions
        grid = F.relu(self.conv1(grid))
        grid = F.relu(self.conv2(grid))
        grid = F.relu(self.conv3(grid))
        
        # Flatten the CNN output
        grid_flat = grid.reshape(grid.size(0), -1)
        
        # Process through dense layer
        grid_features = F.relu(self.fc1(grid_flat))
        
        # Process numerical features
        numerical = x['numerical']
        numerical_features = F.relu(self.numerical_fc(numerical))
        
        # Combine features
        combined = torch.cat([grid_features, numerical_features], dim=1)
        
        # Process through combined layers
        combined = F.relu(self.combined_fc1(combined))
        combined = F.relu(self.combined_fc2(combined))
        
        # Output Q-values
        q_values = self.output(combined)
        
        return q_values

# For future implementation: Add Dueling Network Architecture
class DuelingMazeNavModel(nn.Module):
    """
    Dueling DQN architecture for Infinite Maze navigation.
    
    This architecture separates state value and action advantage estimation,
    which can lead to better performance in many RL tasks.
    
    This is prepared for future training phases.
    """
    
    def __init__(self, grid_size: int = 11, channels: int = 4, num_actions: int = 5):
        """
        Initialize the dueling network architecture.
        
        Args:
            grid_size: Size of the grid observation (default: 11)
            channels: Number of channels in the grid observation (default: 4)
            num_actions: Number of possible actions (default: 5)
        """
        super(DuelingMazeNavModel, self).__init__()
        
        # CNN Backbone for processing spatial information
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate the flattened size after convolutions
        self.conv_output_size = 64 * grid_size * grid_size
        
        # Dense layers for CNN output processing
        self.fc1 = nn.Linear(self.conv_output_size, 256)
        
        # Dense layer for numerical features processing
        self.numerical_fc = nn.Linear(11, 64)  # 11 numerical features
        
        # Combined processing of CNN and numerical features
        self.combined_fc = nn.Linear(256 + 64, 128)
        
        # Dueling Network streams
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single value representing state value
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)  # One advantage value per action
        )
        
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the dueling network.
        
        Args:
            x: Dictionary with 'grid' and 'numerical' tensors
                
        Returns:
            Q-values for each action [batch_size, num_actions]
        """
        # Process grid through CNN
        grid = x['grid']
        grid = grid.permute(0, 3, 1, 2)
        
        grid = F.relu(self.conv1(grid))
        grid = F.relu(self.conv2(grid))
        grid = F.relu(self.conv3(grid))
        
        # Flatten the CNN output
        grid_flat = grid.reshape(grid.size(0), -1)
        
        # Process through dense layer
        grid_features = F.relu(self.fc1(grid_flat))
        
        # Process numerical features
        numerical = x['numerical']
        numerical_features = F.relu(self.numerical_fc(numerical))
        
        # Combine features
        combined = torch.cat([grid_features, numerical_features], dim=1)
        combined = F.relu(self.combined_fc(combined))
        
        # Split into value and advantage streams
        value = self.value_stream(combined)
        advantages = self.advantage_stream(combined)
        
        # Combine value and advantages to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
