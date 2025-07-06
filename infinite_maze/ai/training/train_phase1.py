"""
Training script for Phase 1 of the Infinite Maze AI.

This script implements the Phase 1 training as described in the training plan,
with enhancements to address the observed issues:
1. Rightward bias ("always go right" problem)
2. Low vertical movement utilization
3. Poor score accumulation

Key improvements:
- Enhanced reward structure to better incentivize vertical movement
- Anti-bias mechanisms to prevent over-reliance on rightward movement
- Improved maze configurations with scenarios requiring strategic vertical navigation
- More comprehensive evaluation metrics
"""

import os
import numpy as np
import torch
import time
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import deque

from infinite_maze.ai.environments.environment_phase1 import InfiniteMazeEnv
from infinite_maze.ai.agent import RainbowDQNAgent

def smooth_curve(points: List[float], factor: float = 0.8) -> np.ndarray:
    """
    Smooth a list of values using exponential moving average.
    
    Args:
        points: List of values to smooth
        factor: Smoothing factor (0 = no smoothing, 1 = max smoothing)
        
    Returns:
        Smoothed values as numpy array
    """
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return np.array(smoothed)

def plot_training_metrics(metrics: Dict[str, List], save_path: str) -> None:
    """
    Plot and save training metrics.
    
    Args:
        metrics: Dictionary of training metrics
        save_path: Directory to save the plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Plot episode rewards
    plt.figure(figsize=(10, 5))
    rewards = np.array(metrics['episode_rewards'])
    plt.plot(rewards, alpha=0.3, label='Raw')
    plt.plot(smooth_curve(rewards), label='Smoothed')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'rewards.png'))
    plt.close()
    
    # Plot episode lengths
    plt.figure(figsize=(10, 5))
    lengths = np.array(metrics['episode_lengths'])
    plt.plot(lengths, alpha=0.3, label='Raw')
    plt.plot(smooth_curve(lengths), label='Smoothed')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Lengths')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'lengths.png'))
    plt.close()
    
    # Plot average loss
    if metrics['avg_loss']:
        plt.figure(figsize=(10, 5))
        losses = np.array(metrics['avg_loss'])
        plt.plot(losses, alpha=0.3, label='Raw')
        plt.plot(smooth_curve(losses), label='Smoothed')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Average Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'loss.png'))
        plt.close()
    
    # Plot exploration rate
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['exploration_rate'])
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'epsilon.png'))
    plt.close()
    
    # Plot action distribution (new)
    if 'action_distribution' in metrics:
        plt.figure(figsize=(10, 5))
        action_labels = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'NO_ACTION']
        action_data = np.array(metrics['action_distribution'])
        # Handle case where data doesn't include the 5th action (backward compatibility)
        if action_data.shape[1] >= len(action_labels):
            for i, action in enumerate(action_labels):
                plt.plot(action_data[:, i], label=action)
        else:
            # Only plot the available actions
            for i in range(action_data.shape[1]):
                plt.plot(action_data[:, i], label=action_labels[i])
        plt.xlabel('Evaluation Check')
        plt.ylabel('Action Percentage')
        plt.title('Action Distribution Over Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'action_distribution.png'))
        plt.close()
    
    # Plot score progress (new)
    if 'avg_score_progress' in metrics:
        plt.figure(figsize=(10, 5))
        scores = np.array(metrics['avg_score_progress'])
        plt.plot(scores, alpha=0.7)
        plt.xlabel('Evaluation Check')
        plt.ylabel('Average Score')
        plt.title('Score Progress')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'score_progress.png'))
        plt.close()
    
    # Plot vertical movement rate (new)
    if 'vertical_movement_rate' in metrics:
        plt.figure(figsize=(10, 5))
        v_rate = np.array(metrics['vertical_movement_rate'])
        plt.plot(v_rate, alpha=0.7)
        plt.axhline(y=0.15, color='r', linestyle='--', label='Min Target (15%)')
        plt.axhline(y=0.25, color='g', linestyle='--', label='Max Target (25%)')
        plt.xlabel('Evaluation Check')
        plt.ylabel('Vertical Movement %')
        plt.title('Vertical Movement Utilization')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'vertical_movement.png'))
        plt.close()

def check_phase_completion(eval_stats: Dict[str, float], success_criteria: Dict[str, float]) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if the Phase 1 success criteria have been met.
    
    Args:
        eval_stats: Evaluation statistics
        success_criteria: Success criteria thresholds
        
    Returns:
        Tuple of (success_flag, details)
    """
    # Phase 1 criteria from the updated training plan:
    # - Consistent forward movement (>85% successful rightward attempts)
    # - Minimal collisions (<8% of actions result in wall collisions)
    # - Average score of at least 180 points per episode
    # - Appropriate vertical movement utilization (15-25% of actions)
    
    results = {}
    
    # Check each criterion
    for criterion, threshold_info in success_criteria.items():
        if criterion in eval_stats:
            # Handle different comparison types (greater than or less than)
            threshold = threshold_info['value']
            comparison = threshold_info['comparison']
            
            if comparison == 'greater':
                passed = eval_stats[criterion] >= threshold
            elif comparison == 'less':
                passed = eval_stats[criterion] <= threshold
            elif comparison == 'range':
                min_val, max_val = threshold
                passed = min_val <= eval_stats[criterion] <= max_val
            else:
                passed = False  # Unknown comparison type
                
            results[criterion] = {
                'value': float(eval_stats[criterion]),
                'threshold': threshold,
                'comparison': comparison,
                'passed': bool(passed)
            }
    
    # Overall success if all criteria are met
    success = all(result['passed'] for result in results.values())
    
    return success, results

def detect_oscillation(position_history, action_history, window_size=8):
    """
    Detect oscillation patterns in agent behavior.
    
    Args:
        position_history: List of recent positions
        action_history: List of recent actions
        window_size: Number of steps to analyze
    
    Returns:
        Boolean indicating whether oscillation is detected
    """
    if len(position_history) < window_size or len(action_history) < window_size:
        return False
    
    # Get most recent actions and positions
    recent_actions = action_history[-window_size:]
    
    # Extract vertical actions (UP=0, DOWN=2)
    vertical_actions = [a for a in recent_actions if a in [0, 2]]
    
    # Not enough vertical actions to consider oscillation
    if len(vertical_actions) < 4:
        return False
    
    # Check for alternating pattern (UP, DOWN, UP, DOWN or DOWN, UP, DOWN, UP)
    alternating = True
    for i in range(len(vertical_actions) - 1):
        if vertical_actions[i] == vertical_actions[i+1]:
            alternating = False
            break
    
    # Check for horizontal progress during this period
    horizontal_actions = [a for a in recent_actions if a == 1]  # RIGHT=1
    made_horizontal_progress = len(horizontal_actions) > 0
    
    # It's oscillation if we have alternating vertical actions with minimal horizontal movement
    return alternating and not made_horizontal_progress

def train_phase_1(steps: int = 600000,
                  log_interval: int = 100,
                  eval_interval: int = 10000,
                  save_interval: int = 50000,
                  checkpoint_dir: str = 'ai/checkpoints/phase1',
                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                  render_mode: str = None,
                  use_enhanced_rewards: bool = True,  # New parameter to toggle enhanced rewards
                  anti_bias_checks: bool = True) -> None:  # New parameter to toggle anti-bias mechanisms
    """
    Train the agent for Phase 1 of the curriculum with improved reward structure and anti-bias mechanisms.
    
    Args:
        steps: Total training steps (600K per updated plan)
        log_interval: Episodes between logging
        eval_interval: Steps between evaluations
        save_interval: Steps between saving checkpoints
        checkpoint_dir: Directory to save checkpoints
        device: Device to use for training ('cuda' or 'cpu')
        render_mode: Rendering mode (None, 'human', 'rgb_array')
        use_enhanced_rewards: Whether to use the enhanced reward structure
        anti_bias_checks: Whether to enable anti-bias mechanisms
    """
    # Adjust evaluation and save intervals for short training sessions
    if steps < eval_interval:
        eval_interval = max(500, steps // 2)  # Ensure at least one evaluation
    if steps < save_interval:
        save_interval = max(1000, steps // 2)  # Ensure at least one save checkpoint
    print(f"Starting Phase 1 training on device: {device}")
    print(f"Enhanced rewards: {'Enabled' if use_enhanced_rewards else 'Disabled'}")
    print(f"Anti-bias mechanisms: {'Enabled' if anti_bias_checks else 'Disabled'}")
    
    start_time = time.time()
    
    # Create timestamp-based run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(checkpoint_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save training configuration
    config = {
        'steps': steps,
        'log_interval': log_interval,
        'eval_interval': eval_interval,
        'save_interval': save_interval,
        'device': device,
        'render_mode': render_mode,
        'timestamp': timestamp,
        'use_enhanced_rewards': use_enhanced_rewards,
        'anti_bias_checks': anti_bias_checks
    }
    
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize the training environment (Phase 1 specific with enhanced parameters)
    train_env = InfiniteMazeEnv(
        training_phase=1,
        use_maze_from_start=True,  # Start with maze structures for training
        pace_enabled=False,        # No pace line in Phase 1
        render_mode=render_mode,
        grid_size=11,             # 11x11 grid as specified in training plan
        max_steps=10000,          # Allow longer episodes for better learning
        vertical_corridor_frequency=0.6,  # NEW: Increase frequency of vertical corridors
        force_strategic_vertical_movement=True  # NEW: Force scenarios requiring vertical movement
    )
    
    # Initialize a separate evaluation environment with same enhanced settings
    eval_env = InfiniteMazeEnv(
        training_phase=1,
        use_maze_from_start=True,
        pace_enabled=False,
        render_mode=None,  # No rendering for evaluation
        grid_size=11,
        vertical_corridor_frequency=0.6,
        force_strategic_vertical_movement=True
    )
    
    # Get the observation shape from the environment
    observation, _ = train_env.reset()  # Gymnasium returns (obs, info)
    state_shape = {
        'grid': observation['grid'].shape,
        'numerical': observation['numerical'].shape
    }
    
    # Initialize the agent with further enhanced parameters
    agent = RainbowDQNAgent(
        state_shape=state_shape,
        num_actions=train_env.action_space.n,
        learning_rate=3.5e-4,  # Further increased for faster learning
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=300000,  # Further increased for more exploration
        epsilon_warmup=75000,  # Increased warmup for better exploration
        target_update=4000,    # More frequent updates
        batch_size=256,
        replay_capacity=1000000,
        device=device,
        use_dueling=True,
        n_steps=3,
        alpha=0.7,
        beta_start=0.4
    )
    
    # Define Phase 1 success criteria from updated training plan
    success_criteria = {
        'forward_success_rate': {
            'value': 0.85,         # >85% successful rightward attempts
            'comparison': 'greater'
        },
        'collision_rate': {
            'value': 0.08,         # <8% of actions result in wall collisions
            'comparison': 'less'
        },
        'avg_score': {
            'value': 180,          # Average score of at least 180 points
            'comparison': 'greater'
        },
        'vertical_movement_rate': {
            'value': (0.15, 0.25), # 15-25% vertical movement utilization
            'comparison': 'range'
        }
    }
    
    # Training loop
    episode = 0
    steps_done = 0
    best_eval_reward = float('-inf')  # Will be updated after first evaluation
    training_complete = False
    
    # For tracking progress and timing
    progress_interval = min(5000, steps // 20)  # Show progress ~20 times during training
    last_progress_time = start_time
    last_progress_step = 0
    
    # New tracking metrics for anti-bias mechanisms
    action_distributions = []
    avg_scores = []
    vertical_movement_rates = []
    oscillation_detections = []
    
    # For detecting right-movement bias
    right_action_bias_threshold = 0.60  # Alert if RIGHT actions exceed 60% of total
    
    # Action history for oscillation detection
    position_history = deque(maxlen=20)
    action_history = deque(maxlen=20)
    
    print("Starting training loop with improved anti-bias mechanisms...")
    print(f"Target: {steps} steps | Evaluations every {eval_interval} steps | Checkpoints every {save_interval} steps")
    
    while steps_done < steps and not training_complete:
        # Train for one episode
        episode_start_time = time.time()
        
        # Reset environment and get initial observation
        observation, _ = train_env.reset()
        done = False
        
        # Episode variables
        episode_reward = 0
        episode_length = 0
        episode_actions = [0, 0, 0, 0, 0]  # UP, RIGHT, DOWN, LEFT, NO_ACTION counts
        oscillation_count = 0
        
        # Run one episode
        while not done:
            # Select action
            action = agent.select_action(observation)
            
            # Update action counts
            episode_actions[action] += 1
            
            # Update action history
            action_history.append(action)
            
            # Execute action in environment
            next_observation, reward, done, truncated, info = train_env.step(action)
            
            # Update position history (if available in info)
            if 'player_position' in info:
                position_history.append(info['player_position'])
            
            # Detect oscillation
            if len(position_history) >= 8 and len(action_history) >= 8:
                if detect_oscillation(position_history, action_history):
                    oscillation_count += 1
                    # Apply oscillation penalty if enhanced rewards are enabled
                    if use_enhanced_rewards:
                        reward -= 0.7  # Significant penalty for oscillation behavior
            
            # Apply enhanced reward structure if enabled
            if use_enhanced_rewards:
                # Enhanced rewards for strategic vertical movement
                if action in [0, 2]:  # UP or DOWN
                    # Check if this vertical movement is strategic
                    if 'path_improved' in info and info['path_improved']:
                        reward += 1.0  # Strong reward for beneficial vertical movement
                    elif 'path_clearance' in info and info['path_clearance'] > 0:
                        reward += 0.6  # Medium reward for movement that leads to clearer areas
                
                # Balance rightward movement rewards
                if action == 1:  # RIGHT
                    # Calculate right action percentage so far
                    total_actions = sum(episode_actions)
                    if total_actions > 0:
                        right_action_percentage = episode_actions[1] / total_actions
                        
                        # If showing right bias, reduce the reward for going right
                        if right_action_percentage > right_action_bias_threshold and anti_bias_checks:
                            reward *= 0.6  # Reduce rightward movement reward if bias detected
                
                # Score-based rewards
                if 'score' in info:
                    score = info['score']
                    # Add small bonus for score milestones
                    if score > 0 and score % 50 == 0:
                        reward += 2.0  # Bonus for reaching score milestones
            
            # Store transition in memory
            agent.store_experience(observation, action, reward, next_observation, done)
            
            # Update the model
            loss = agent.optimize_model()
            
            # Move to the next state
            observation = next_observation
            episode_reward += reward
            episode_length += 1
            
            # Update steps count
            steps_done = agent.steps_done
            
            # Break if we've reached the step limit
            if steps_done >= steps:
                break
        
        # End of episode
        episode += 1
        
        # Update agent's tracking metrics
        if not hasattr(agent, 'training_metrics'):
            agent.training_metrics = {
                'episode_rewards': [],
                'episode_lengths': [],
                'avg_loss': [],
                'exploration_rate': [],
                'action_distribution': [],  # New tracking metric
                'avg_score_progress': [],   # New tracking metric
                'vertical_movement_rate': [],  # New tracking metric
                'oscillation_detections': []   # New tracking metric
            }
        
        agent.training_metrics['episode_rewards'].append(episode_reward)
        agent.training_metrics['episode_lengths'].append(episode_length)
        agent.training_metrics['exploration_rate'].append(agent.epsilon)
        
        # Calculate action distribution for this episode
        if sum(episode_actions) > 0:
            action_dist = [count / sum(episode_actions) for count in episode_actions]
            vertical_movement = action_dist[0] + action_dist[2]  # UP + DOWN percentage
        else:
            action_dist = [0, 0, 0, 0, 0]  # Make sure we have 5 values for 5 actions
            vertical_movement = 0
        
        # Track vertical movement rate
        vertical_movement_rates.append(vertical_movement)
        
        # Track oscillation detections
        oscillation_detections.append(oscillation_count)
        
        # Anti-bias check - if RIGHT bias is extreme, force exploration for a few episodes
        if anti_bias_checks and action_dist[1] > 0.75 and vertical_movement < 0.15:
            print("\n⚠️ WARNING: Excessive rightward bias detected! Increasing exploration temporarily.")
            # Force more exploration for a few episodes
            agent.epsilon = min(0.7, agent.epsilon * 2)
            # Create temporary save point before intervention
            agent.save(os.path.join(run_dir, 'pre_intervention_checkpoint'))
        
        # Progress reporting with estimated time
        if steps_done >= last_progress_step + progress_interval:
            current_timestamp = datetime.now()
            current_time = time.time()
            elapsed = current_time - start_time
            progress_percent = (steps_done / steps) * 100
            
            # Calculate time per step and estimate remaining time
            if steps_done > 0:
                time_per_step = elapsed / steps_done
                steps_remaining = steps - steps_done
                estimated_time_remaining = time_per_step * steps_remaining
                
                # Calculate elapsed and remaining time in hours/minutes/seconds
                elapsed_hrs, remainder = divmod(elapsed, 3600)
                elapsed_mins, elapsed_secs = divmod(remainder, 60)
                
                remaining_hrs, remainder = divmod(estimated_time_remaining, 3600)
                remaining_mins, remaining_secs = divmod(remainder, 60)
                
                progress_speed = (steps_done - last_progress_step) / (current_time - last_progress_time)

                print(f"\nTimestamp: {current_timestamp}")
                print(f"[Progress: {progress_percent:.1f}%] Steps: {steps_done}/{steps}")
                print(f"Time elapsed: {int(elapsed_hrs)}h {int(elapsed_mins)}m {int(elapsed_secs)}s | " 
                     f"Estimated remaining: {int(remaining_hrs)}h {int(remaining_mins)}m {int(remaining_secs)}s")
                print(f"Training speed: {progress_speed:.1f} steps/sec | Episodes completed: {episode}")
                print(f"Current reward: {episode_reward:.2f} | Exploration rate: {agent.epsilon:.3f}")
                
                # Print action distribution
                print(f"Action distribution: UP={action_dist[0]:.2f}, RIGHT={action_dist[1]:.2f}, "
                      f"DOWN={action_dist[2]:.2f}, LEFT={action_dist[3]:.2f}, "
                      f"NO_ACTION={action_dist[4] if len(action_dist) > 4 else 0:.2f}")
                print(f"Vertical movement: {vertical_movement:.2%} (Target: 15-25%)")
                
                if oscillation_count > 0:
                    print(f"⚠️ Oscillation detected {oscillation_count} times in this episode")
                
                if best_eval_reward != float('-inf'):
                    print(f"Best evaluation reward so far: {best_eval_reward:.2f}")
                print("-" * 80)
            
            last_progress_step = steps_done
            last_progress_time = current_time
        
        # Detailed logging (less frequent)
        elif episode % log_interval == 0:
            print(f"Episode {episode} | Steps: {steps_done}/{steps} | "
                 f"Reward: {episode_reward:.2f} | Eps: {agent.epsilon:.3f}")
        
        # Evaluation - check if we've crossed an evaluation threshold
        next_eval_step = (steps_done // eval_interval) * eval_interval
        if next_eval_step > 0 and steps_done >= next_eval_step and (last_eval_step := getattr(agent, 'last_eval_step', -eval_interval)) < next_eval_step:
            print("\n=== Running evaluation... ===")
            eval_start_time = time.time()
            eval_stats = agent.evaluate(eval_env, num_episodes=10)
            eval_duration = time.time() - eval_start_time
            
            # Store the last evaluation step to avoid repeated evaluations
            agent.last_eval_step = next_eval_step
            
            # Track action distribution for this evaluation
            if hasattr(eval_stats, 'action_counts') and sum(eval_stats['action_counts']) > 0:
                action_distribution = [count / sum(eval_stats['action_counts']) for count in eval_stats['action_counts']]
                action_distributions.append(action_distribution)
                agent.training_metrics['action_distribution'].append(action_distribution)
                
                # Track vertical movement rate
                v_movement = action_distribution[0] + action_distribution[2]  # UP + DOWN
                vertical_movement_rates.append(v_movement)
                agent.training_metrics['vertical_movement_rate'].append(v_movement)
            
            # Track average score
            if 'avg_score' in eval_stats:
                avg_scores.append(eval_stats['avg_score'])
                agent.training_metrics['avg_score_progress'].append(eval_stats['avg_score'])
            
            progress_percent = (steps_done / steps) * 100
            print(f"[Progress: {progress_percent:.1f}%] Evaluation Results:")
            print(f"Avg Reward: {eval_stats['avg_reward']:.2f} ± {eval_stats['std_reward']:.2f} | "
                 f"Avg Length: {eval_stats['avg_length']:.2f} | "
                 f"Time: {eval_duration:.2f}s")
            
            # Save if best model so far
            if eval_stats['avg_reward'] > best_eval_reward:
                best_eval_reward = eval_stats['avg_reward']
                agent.save(os.path.join(run_dir, 'best_model'))
                print(f"New best model saved with average reward: {best_eval_reward:.2f}")
            
            # Create a dictionary with available metrics and safely handle missing keys
            eval_stats_extended = {
                'avg_reward': eval_stats.get('avg_reward', 0.0),
                'avg_score': eval_stats.get('avg_score', 0.0),
                'forward_success_rate': eval_stats.get('forward_success_rate', 0.0),
                'collision_rate': eval_stats.get('collision_rate', 1.0),  # Default to worst case
                'vertical_movement_rate': eval_stats.get('vertical_movement_rate', 0.0)
            }
            
            # Print available metrics for debugging
            print(f"Forward success: {eval_stats_extended['forward_success_rate']:.2%}, "
                  f"Collision rate: {eval_stats_extended['collision_rate']:.2%}, "
                  f"Score: {eval_stats_extended['avg_score']:.1f}, "
                  f"Vertical movement: {eval_stats_extended['vertical_movement_rate']:.2%}")
            
            # Check for extreme rightward bias
            if eval_stats.get('action_counts') and sum(eval_stats['action_counts']) > 0:
                action_dist = [count / sum(eval_stats['action_counts']) for count in eval_stats['action_counts']]
                
                if action_dist[1] > right_action_bias_threshold and anti_bias_checks:
                    print("\n⚠️ WARNING: RIGHT movement bias detected in evaluation!")
                    print(f"RIGHT movement: {action_dist[1]:.2%} (threshold: {right_action_bias_threshold:.2%})")
                    print("Implementing anti-bias correction...")
                    
                    # Apply anti-bias correction if enabled
                    if anti_bias_checks:
                        # Temporarily increase exploration
                        agent.epsilon = min(0.5, agent.epsilon * 1.5)
                        print(f"Increased exploration rate to {agent.epsilon:.2f} to encourage diversity")
            
            success, criteria_results = check_phase_completion(eval_stats_extended, success_criteria)
            
            # Save criteria results
            with open(os.path.join(run_dir, f'criteria_eval_step_{steps_done}.json'), 'w') as f:
                json.dump(criteria_results, f, indent=2)
            
            if success:
                print("\n🎉 SUCCESS: Phase 1 completion criteria met! 🎉")
                agent.save(os.path.join(run_dir, 'final_model'))
                training_complete = True
        
        # Regular checkpoints - check if we've crossed a checkpoint threshold
        next_save_step = (steps_done // save_interval) * save_interval
        if next_save_step > 0 and steps_done >= next_save_step and (last_save_step := getattr(agent, 'last_save_step', -save_interval)) < next_save_step:
            checkpoint_path = os.path.join(run_dir, f'checkpoint_step_{next_save_step}')
            agent.save(checkpoint_path)
            print(f"Checkpoint saved at step {next_save_step}")
            
            # Plot and save training metrics
            plot_training_metrics(agent.training_metrics, os.path.join(run_dir, 'plots'))
            
            # Store the last save step to avoid repeated saves
            agent.last_save_step = next_save_step
    
    # Final save
    if not training_complete:
        agent.save(os.path.join(run_dir, 'final_model'))
    
    # Force a final evaluation regardless of steps
    print("\n=== Running final evaluation... ===")
    
    # Make sure the evaluation environment is reset properly
    eval_env.reset()
    
    # Use more episodes for final evaluation
    eval_episodes = 10
    final_eval_stats = agent.evaluate(eval_env, num_episodes=eval_episodes)
    
    print(f"\nFINAL EVALUATION RESULTS:")
    print(f"Avg Reward: {final_eval_stats['avg_reward']:.2f} ± {final_eval_stats['std_reward']:.2f}")
    print(f"Avg Episode Length: {final_eval_stats['avg_length']:.2f} ± {final_eval_stats['std_length']:.2f}")
    
    if 'avg_score' in final_eval_stats:
        print(f"Avg Score: {final_eval_stats['avg_score']:.2f}")
    
    if 'forward_success_rate' in final_eval_stats:
        print(f"Forward Success Rate: {final_eval_stats['forward_success_rate']:.2%}")
    
    if 'collision_rate' in final_eval_stats:
        print(f"Collision Rate: {final_eval_stats['collision_rate']:.2%}")
    
    if 'vertical_movement_rate' in final_eval_stats:
        print(f"Vertical Movement Rate: {final_eval_stats['vertical_movement_rate']:.2%}")
    
    print(f"Episodes completed during training: {episode}")
         
    # Update best reward if final evaluation is better or if this is the first evaluation
    if best_eval_reward == float('-inf') or final_eval_stats['avg_reward'] > best_eval_reward:
        best_eval_reward = final_eval_stats['avg_reward']
        agent.save(os.path.join(run_dir, 'best_model'))
        print(f"New best model saved with average reward: {best_eval_reward:.2f}")
    
    # Plot final training metrics
    plot_training_metrics(agent.training_metrics, os.path.join(run_dir, 'plots'))
    
    # Calculate and print training statistics
    total_duration = time.time() - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining completed:")
    print(f"Total episodes: {episode}")
    print(f"Total steps: {steps_done}")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    print(f"Total duration: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save final stats
    final_stats = {
        'episodes': episode,
        'steps': steps_done,
        'best_eval_reward': float(best_eval_reward),
        'training_duration_seconds': total_duration,
        'training_complete': training_complete,
        'final_metrics': {
            'avg_reward': float(final_eval_stats['avg_reward']),
            'avg_score': float(final_eval_stats.get('avg_score', 0)),
            'forward_success_rate': float(final_eval_stats.get('forward_success_rate', 0)),
            'collision_rate': float(final_eval_stats.get('collision_rate', 0)),
            'vertical_movement_rate': float(final_eval_stats.get('vertical_movement_rate', 0))
        }
    }
    
    with open(os.path.join(run_dir, 'training_summary.json'), 'w') as f:
        json.dump(final_stats, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Phase 1 of Infinite Maze AI')
    parser.add_argument('--steps', type=int, default=600000, help='Number of training steps (600K default per training plan)')
    parser.add_argument('--log-interval', type=int, default=100, help='Episodes between logging')
    parser.add_argument('--eval-interval', type=int, default=10000, help='Steps between evaluations')
    parser.add_argument('--save-interval', type=int, default=50000, help='Steps between checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default='ai/checkpoints/phase1', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--disable-enhanced-rewards', action='store_true', help='Disable enhanced reward structure')
    parser.add_argument('--disable-anti-bias', action='store_true', help='Disable anti-bias mechanisms')
    
    args = parser.parse_args()
    
    render_mode = 'human' if args.render else None
    
    # Auto-adjust intervals for small step counts
    if args.steps < args.eval_interval:
        print(f"NOTE: Auto-adjusting evaluation interval to {args.steps // 5} for short training run")
        args.eval_interval = max(100, args.steps // 5)
        
    if args.steps < args.save_interval:
        print(f"NOTE: Auto-adjusting save interval to {args.steps // 2} for short training run")
        args.save_interval = max(500, args.steps // 2)
    
    print("Starting Phase 1 training with improved anti-bias mechanisms")
    print(f"Training for {args.steps} steps with evaluations every {args.eval_interval} steps")
    print(f"Using device: {args.device}")
    print(f"Enhanced rewards: {'Disabled' if args.disable_enhanced_rewards else 'Enabled'}")
    print(f"Anti-bias mechanisms: {'Disabled' if args.disable_anti_bias else 'Enabled'}")
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    
    train_phase_1(
        steps=args.steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        render_mode=render_mode,
        use_enhanced_rewards=not args.disable_enhanced_rewards,
        anti_bias_checks=not args.disable_anti_bias
    )
