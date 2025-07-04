"""
Reinforcement learning agent for the Infinite Maze game.

This module implements the Rainbow DQN agent as specified in the training plan,
with all the components needed for Phase 1 of training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, Tuple, List, Any, Optional
import os
import time
import json

from .models import MazeNavModel, DuelingMazeNavModel

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', 
                       ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    Implements prioritized experience replay for more efficient learning.
    """
    
    def __init__(self, capacity: int = 1_000_000, alpha: float = 0.6):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform sampling, 1 = fully prioritized)
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        
    def add(self, state: Dict[str, np.ndarray], action: int, reward: float, 
            next_state: Dict[str, np.ndarray], done: bool) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
            
        # Set max priority for new experiences
        max_priority = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.priorities = np.append(self.priorities, max_priority)
        else:
            self.priorities[self.position] = max_priority
            
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences based on priorities.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            
        Returns:
            Tuple of (experiences, indices, weights)
        """
        if len(self.memory) < batch_size:
            return random.sample(self.memory, len(self.memory)), None, None
            
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.memory)]
        # Handle potential zero priorities
        priorities = np.clip(priorities, 1e-8, None)  # Ensure no zeros
        probabilities = priorities ** self.alpha
        prob_sum = probabilities.sum()
        if prob_sum > 0:
            probabilities /= prob_sum
        else:
            # Fallback to uniform if all priorities are 0
            probabilities = np.ones_like(probabilities) / len(probabilities)
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probabilities[indices]) ** -beta
        weights /= weights.max()  # Normalize weights
        
        experiences = [self.memory[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update the priorities of experiences.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            Number of experiences in buffer
        """
        return len(self.memory)

class RainbowDQNAgent:
    """
    Rainbow DQN agent for the Infinite Maze game.
    
    Implements key features of Rainbow DQN:
    - Double Q-Learning
    - Prioritized Experience Replay
    - Dueling Networks (optional)
    - Multi-step learning
    """
    
    def __init__(self, 
                 state_shape: Dict[str, tuple],
                 num_actions: int = 5,
                 learning_rate: float = 2.5e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 1000000,
                 target_update: int = 8000,
                 batch_size: int = 128,
                 replay_capacity: int = 1000000,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 n_steps: int = 1,  # Multi-step learning (for future phases)
                 use_dueling: bool = False,  # Use dueling architecture
                 alpha: float = 0.6,  # PER exponent
                 beta_start: float = 0.4,  # IS correction exponent
                 beta_frames: float = 1000000):  # Frames over which to anneal beta
        """
        Initialize the Rainbow DQN agent.
        
        Args:
            state_shape: Shape of the state observation
            num_actions: Number of possible actions
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Steps over which to decay epsilon
            target_update: Frequency of target network updates
            batch_size: Size of training batches
            replay_capacity: Capacity of the replay buffer
            device: Device to use for computation
            n_steps: Number of steps for multi-step learning
            use_dueling: Whether to use dueling network architecture
            alpha: Priority exponent for replay buffer
            beta_start: Initial importance sampling correction exponent
            beta_frames: Frames over which to anneal beta
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.device = device
        self.n_steps = n_steps
        self.use_dueling = use_dueling
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.steps_done = 0
        
        # Initialize networks
        if use_dueling:
            self.policy_net = DuelingMazeNavModel(
                grid_size=state_shape['grid'][0], 
                channels=state_shape['grid'][2],
                num_actions=num_actions
            ).to(device)
            
            self.target_net = DuelingMazeNavModel(
                grid_size=state_shape['grid'][0],
                channels=state_shape['grid'][2],
                num_actions=num_actions
            ).to(device)
        else:
            self.policy_net = MazeNavModel(
                grid_size=state_shape['grid'][0],
                channels=state_shape['grid'][2],
                num_actions=num_actions
            ).to(device)
            
            self.target_net = MazeNavModel(
                grid_size=state_shape['grid'][0],
                channels=state_shape['grid'][2],
                num_actions=num_actions
            ).to(device)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer with prioritized experience replay
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity, alpha=alpha)
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_loss': [],
            'exploration_rate': []
        }
        
    def select_action(self, state: Dict[str, np.ndarray], evaluate: bool = False) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            evaluate: Whether to use evaluation mode (reduced exploration)
            
        Returns:
            Selected action
        """
        # Convert numpy arrays to tensors and add batch dimension
        state_tensor = {
            'grid': torch.FloatTensor(state['grid']).unsqueeze(0).to(self.device),
            'numerical': torch.FloatTensor(state['numerical']).unsqueeze(0).to(self.device)
        }
        
        # During evaluation, use less exploration
        eval_epsilon = 0.01 if evaluate else self.epsilon
        
        # Epsilon-greedy action selection
        if random.random() > eval_epsilon:
            # Exploit: select best action
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            # Explore: select random action
            return random.randrange(self.num_actions)
    
    def update_epsilon(self) -> None:
        """
        Update the exploration rate based on the current step.
        """
        # Linear annealing from epsilon_start to epsilon_end over epsilon_decay steps
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon_start - (self.steps_done / self.epsilon_decay) * (self.epsilon_start - self.epsilon_end)
        )
        
    def calculate_beta(self) -> float:
        """
        Calculate the current beta value for importance sampling.
        
        Returns:
            Current beta value
        """
        # Linear annealing of beta from beta_start to 1.0
        fraction = min(self.steps_done / self.beta_frames, 1.0)
        return self.beta_start + fraction * (1.0 - self.beta_start)
    
    def store_experience(self, state: Dict[str, np.ndarray], action: int, reward: float, 
                        next_state: Dict[str, np.ndarray], done: bool) -> None:
        """
        Store an experience in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        
    def optimize_model(self) -> Optional[float]:
        """
        Perform one step of optimization using a batch of experiences.
        
        Returns:
            Loss value if optimization was performed, None otherwise
        """
        # Skip if not enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        # Sample a batch of experiences
        beta = self.calculate_beta()
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size, beta)
        
        # Extract batch components
        batch_states = {
            'grid': torch.FloatTensor(np.array([exp.state['grid'] for exp in experiences])).to(self.device),
            'numerical': torch.FloatTensor(np.array([exp.state['numerical'] for exp in experiences])).to(self.device)
        }
        
        batch_actions = torch.LongTensor(np.array([exp.action for exp in experiences])).unsqueeze(1).to(self.device)
        batch_rewards = torch.FloatTensor(np.array([exp.reward for exp in experiences])).unsqueeze(1).to(self.device)
        
        non_final_mask = torch.BoolTensor(np.array([not exp.done for exp in experiences])).to(self.device)
        
        non_final_next_states = {
            'grid': torch.FloatTensor(np.array([exp.next_state['grid'] for exp in experiences if not exp.done])).to(self.device),
            'numerical': torch.FloatTensor(np.array([exp.next_state['numerical'] for exp in experiences if not exp.done])).to(self.device)
        }
        
        importance_weights = torch.FloatTensor(weights if weights is not None else np.ones(self.batch_size)).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(batch_states).gather(1, batch_actions)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, 1, device=self.device)
        
        if non_final_mask.sum() > 0:
            # Double Q-learning: 
            # 1. Select action using policy network
            next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            
            # 2. Evaluate action using target network
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions)
        
        # Compute the expected Q values
        expected_state_action_values = batch_rewards + (self.gamma ** self.n_steps) * next_state_values
        
        # Compute loss (Huber loss)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
        
        # Apply importance sampling weights
        weighted_loss = (loss * importance_weights.unsqueeze(1)).mean()
        
        # Calculate priorities for replay buffer update
        with torch.no_grad():
            td_errors = loss.detach().cpu().numpy()
        
        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update priorities in the replay buffer
        if indices is not None:
            new_priorities = td_errors + 1e-6  # Small constant to ensure non-zero priorities
            self.replay_buffer.update_priorities(indices, new_priorities)
        
        return weighted_loss.item()
    
    def update_target_network(self) -> None:
        """
        Update the target network with the policy network weights.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def train_step(self, state: Dict[str, np.ndarray]) -> int:
        """
        Perform a single training step.
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action
        """
        # Select action
        action = self.select_action(state)
        
        # Update exploration rate
        self.update_epsilon()
        
        # Update step counter
        self.steps_done += 1
        
        # Update target network if needed
        if self.steps_done % self.target_update == 0:
            self.update_target_network()
            
        return action
    
    def train_episode(self, env: Any, max_steps: int = 10000, verbose: bool = False) -> Dict[str, float]:
        """
        Train for a complete episode.
        
        Args:
            env: Training environment
            max_steps: Maximum steps per episode
            verbose: Whether to print detailed debug information
            
        Returns:
            Dictionary with episode statistics
        """
        state, _ = env.reset()  # Gymnasium returns (obs, info)
        total_reward = 0
        losses = []
        
        # Add timeout tracking
        import time
        start_time = time.time()
        timeout = 60  # 60 seconds timeout
        
        for step in range(max_steps):
            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                if verbose:
                    print(f"WARNING: Episode timed out after {elapsed:.1f} seconds")
                break
                
            # Select action
            action = self.train_step(state)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Check if episode is done (terminated or truncated)
            done = terminated or truncated
            
            # Store experience
            self.store_experience(state, action, reward, next_state, done)
            
            # Optimize model
            loss = self.optimize_model()
            if loss is not None:
                losses.append(loss)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Force render every few steps if render is enabled
            if hasattr(env, 'render') and step % 5 == 0 and env.render_mode is not None:
                env.render()
                
            # Break if done
            if done and verbose:
                print(f"Episode ended after {step} steps with reward {total_reward:.2f}")
                break
                
        # Update metrics
        episode_length = step + 1
        self.training_metrics['episode_rewards'].append(total_reward)
        self.training_metrics['episode_lengths'].append(episode_length)
        self.training_metrics['exploration_rate'].append(self.epsilon)
        
        if losses:
            self.training_metrics['avg_loss'].append(np.mean(losses))
        else:
            self.training_metrics['avg_loss'].append(0)
            
        # Calculate statistics
        episode_stats = {
            'reward': total_reward,
            'length': episode_length,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(losses) if losses else 0
        }
        
        return episode_stats
    
    def evaluate(self, env: Any, num_episodes: int = 10, verbose: bool = False) -> Dict[str, Any]:
        """
        Evaluate the agent's performance.
        
        Args:
            env: Evaluation environment
            num_episodes: Number of episodes to evaluate
            verbose: Whether to print detailed debug information
            
        Returns:
            Dictionary with evaluation statistics
        """
        rewards = []
        lengths = []
        
        # Add timeout tracking
        max_eval_time = 60  # seconds
        
        for episode in range(num_episodes):
            start_time = time.time()
            state = env.reset()[0]  # Gymnasium returns (obs, info)
            total_reward = 0
            done = False
            step = 0
            
            # Set a reasonable maximum number of steps for evaluation
            max_steps = 500  # Even more reasonable for evaluation during short training runs
            
            while not done and step < max_steps:
                # Check for timeout
                if time.time() - start_time > max_eval_time:
                    print(f"Evaluation episode {episode+1} timed out after {max_eval_time} seconds")
                    break
                    
                action = self.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Only print debug info if verbose flag is set (we'll add this parameter)
                # Removed verbose debug output for cleaner training logs
                
                # Check if episode is done
                done = terminated or truncated
                
                state = next_state
                total_reward += reward
                step += 1
                
            rewards.append(total_reward)
            lengths.append(step)
            
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'rewards': rewards,
            'lengths': lengths
        }
    
    def save(self, path: str) -> None:
        """
        Save the model and training state.
        
        Args:
            path: Directory path to save to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model weights
        torch.save(self.policy_net.state_dict(), os.path.join(path, 'policy_net.pt'))
        torch.save(self.target_net.state_dict(), os.path.join(path, 'target_net.pt'))
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
        
        # Save training state
        training_state = {
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'metrics': self.training_metrics
        }
        
        with open(os.path.join(path, 'training_state.json'), 'w') as f:
            json.dump(training_state, f)
            
    def load(self, path: str) -> None:
        """
        Load the model and training state.
        
        Args:
            path: Directory path to load from
        """
        # Load model weights
        self.policy_net.load_state_dict(torch.load(os.path.join(path, 'policy_net.pt')))
        self.target_net.load_state_dict(torch.load(os.path.join(path, 'target_net.pt')))
        
        # Load optimizer state
        self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pt')))
        
        # Load training state
        with open(os.path.join(path, 'training_state.json'), 'r') as f:
            training_state = json.load(f)
            
        self.steps_done = training_state['steps_done']
        self.epsilon = training_state['epsilon']
        self.training_metrics = training_state['metrics']
