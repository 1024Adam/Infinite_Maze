"""
Training script for Phase 2 of the Infinite Maze AI.

This script implements the Phase 2 training as described in the training plan,
focusing on complex navigation with constant, slow pace and improved path discovery.
"""

import os
import numpy as np
import torch
import time
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from infinite_maze.ai.environments.environment_phase2 import Phase2MazeEnv
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
    if 'avg_loss' in metrics and metrics['avg_loss']:
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
    
    # Phase 2 specific: Plot pace distance
    if 'pace_distances' in metrics and metrics['pace_distances']:
        plt.figure(figsize=(10, 5))
        distances = np.array(metrics['pace_distances'])
        plt.plot(distances, alpha=0.3, label='Raw')
        plt.plot(smooth_curve(distances), label='Smoothed')
        plt.xlabel('Episode')
        plt.ylabel('Average Distance')
        plt.title('Average Distance to Pace Line')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'pace_distances.png'))
        plt.close()
    
    # Phase 2 specific: Plot oscillation counts
    if 'oscillation_counts' in metrics and metrics['oscillation_counts']:
        plt.figure(figsize=(10, 5))
        oscillations = np.array(metrics['oscillation_counts'])
        plt.plot(oscillations, alpha=0.3, label='Raw')
        plt.plot(smooth_curve(oscillations), label='Smoothed')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        plt.title('Oscillation Incidents per Episode')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'oscillations.png'))
        plt.close()

def check_phase_completion(eval_stats: Dict[str, float], success_criteria: Dict[str, Dict]) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if the Phase 2 success criteria have been met.
    
    Args:
        eval_stats: Evaluation statistics
        success_criteria: Success criteria thresholds
        
    Returns:
        Tuple of (success_flag, details)
    """
    # Phase 2 criteria from the training plan:
    # - Maintaining safe distance from pace line (average distance >100 pixels)
    # - Survival time of at least 2 minutes per episode
    # - Vertical movement utilization in 25-45% of actions
    # - Minimal oscillation behavior (<5% of navigation attempts)
    
    results = {}
    
    # Check each criterion
    for criterion, threshold_info in success_criteria.items():
        if criterion in eval_stats:
            # Handle different comparison types (greater than, less than, or range)
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
                'threshold': threshold if isinstance(threshold, (int, float)) else [float(v) for v in threshold],
                'comparison': comparison,
                'passed': bool(passed)
            }
    
    # Overall success if all criteria are met
    success = all(result['passed'] for result in results.values())
    
    return success, results

def train_phase_2(checkpoint_path: Optional[str] = None,
                  steps: int = 1200000,  # 1.2M steps per training plan
                  log_interval: int = 100,
                  eval_interval: int = 20000,
                  save_interval: int = 50000,
                  checkpoint_dir: str = 'ai/checkpoints/phase2',
                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                  render_mode: str = None,
                  pace_start_speed: float = 0.2,  # 20% of normal
                  pace_acceleration: bool = False,
                  gradually_increase_pace: bool = True) -> None:
    """
    Train the agent for Phase 2 of the curriculum based on the training plan.
    
    Args:
        checkpoint_path: Path to the best Phase 1 checkpoint to start from
        steps: Total training steps (1.2M per plan)
        log_interval: Episodes between logging
        eval_interval: Steps between evaluations
        save_interval: Steps between saving checkpoints
        checkpoint_dir: Directory to save checkpoints
        device: Device to use for training ('cuda' or 'cpu')
        render_mode: Rendering mode (None, 'human', 'rgb_array')
        pace_start_speed: Initial pace speed (0.2 = 20% of normal)
        pace_acceleration: Whether to enable pace acceleration
        gradually_increase_pace: Whether to gradually increase pace over training
    """
    # Adjust evaluation and save intervals for short training sessions
    if steps < eval_interval:
        eval_interval = max(500, steps // 5)  # Ensure multiple evaluations
    if steps < save_interval:
        save_interval = max(1000, steps // 2)  # Ensure at least one save checkpoint
    print(f"Starting Phase 2 training on device: {device}")
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
        'phase1_checkpoint': checkpoint_path,
        'pace_start_speed': pace_start_speed,
        'pace_acceleration': pace_acceleration,
        'gradually_increase_pace': gradually_increase_pace
    }
    
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize the training environment for Phase 2
    # Start with slower pace and gradually increase if specified
    current_pace_speed = pace_start_speed
    
    train_env = Phase2MazeEnv(
        pace_enabled=True,
        pace_speed=current_pace_speed,
        pace_acceleration=pace_acceleration,
        render_mode=render_mode,
        grid_size=11,
        max_steps=10000,  # Allow longer episodes for better learning
        maze_density=1.0,  # Standard maze density to start
        start_position_difficulty=0.1  # Start with easier positions
    )
    
    # Initialize a separate evaluation environment
    eval_env = Phase2MazeEnv(
        pace_enabled=True,
        pace_speed=0.2,  # Fixed speed for evaluation consistency
        pace_acceleration=False,  # No acceleration during eval for consistency
        render_mode=None,  # No rendering for evaluation
        grid_size=11,
        maze_density=1.0,
        start_position_difficulty=0.5  # Medium difficulty for evaluation
    )
    
    # Get the observation shape from the environment
    observation, _ = train_env.reset()  # Gymnasium returns (obs, info)
    state_shape = {
        'grid': observation['grid'].shape,
        'numerical': observation['numerical'].shape
    }
    
    # Phase 2: Must start from a Phase 1 checkpoint
    if not checkpoint_path:
        raise ValueError("Phase 2 training must start from a Phase 1 checkpoint")
    
    # Load agent from Phase 1 checkpoint
    print(f"Loading agent from Phase 1 checkpoint: {checkpoint_path}")
    agent = RainbowDQNAgent(
        state_shape=state_shape,
        num_actions=train_env.action_space.n,
        device=device,
        # Phase 2: Activate dueling architecture as specified in training plan
        use_dueling=True
    )
    agent.load(checkpoint_path)
    
    # For Phase 2, we want to partially reset the exploration rate for better adaptation
    # This helps the agent adjust to the new pace mechanics
    agent.epsilon = 0.3  # Reset to higher exploration for Phase 2 adaptation
    
    # Update learning parameters for Phase 2
    # Slightly lower learning rate for fine-tuning
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = 1.5e-4
    
    # Define Phase 2 success criteria from training plan
    success_criteria = {
        'avg_pace_distance': {
            'value': 100,          # >100 pixels average distance from pace line
            'comparison': 'greater'
        },
        'avg_survival_time': {
            'value': 120,          # ≥2 minutes (120 seconds) per episode
            'comparison': 'greater'
        },
        'vertical_movement_rate': {
            'value': (0.25, 0.45), # 25-45% vertical movement utilization
            'comparison': 'range'
        },
        'oscillation_rate': {
            'value': 0.05,         # <5% of navigation attempts show oscillation
            'comparison': 'less'
        }
    }
    
    # Training loop
    episode = 0
    steps_done = 0
    best_eval_reward = float('-inf')
    best_eval_survival_time = 0
    training_complete = False
    
    # Enhanced metrics tracking for Phase 2
    enhanced_metrics = {
        'pace_distances': [],      # Average distance to pace line per episode
        'survival_times': [],      # Episode survival time in seconds
        'oscillation_counts': [],  # Oscillation incidents per episode
        'vertical_movement_rates': []  # Percentage of vertical movements per episode
    }
    
    # For tracking progress and timing
    progress_interval = min(10000, steps // 20)  # Show progress ~20 times during training
    last_progress_time = start_time
    last_progress_step = 0
    
    # For curriculum progression
    pace_increase_steps = 150000   # Per training plan, gradually increase over 150K steps
    pace_adjustment_interval = pace_increase_steps // 6  # 6 steps to reach full speed
    maze_density_increases = 5     # Number of maze density increases
    maze_density_interval = steps // maze_density_increases
    difficulty_increases = 10      # Number of difficulty increases
    difficulty_interval = steps // difficulty_increases
    
    print("Starting Phase 2 training loop...")
    print(f"Target: {steps} steps | Evaluations every {eval_interval} steps | Checkpoints every {save_interval} steps")
    
    while steps_done < steps and not training_complete:
        # Curriculum adjustments
        if gradually_increase_pace and steps_done > 0 and steps_done % pace_adjustment_interval == 0:
            # Calculate progress through the pace increase phase
            pace_progress = min(1.0, steps_done / pace_increase_steps)
            # Adjust pace from start_speed to 1.0 linearly
            target_pace = pace_start_speed + (1.0 - pace_start_speed) * pace_progress
            
            # Only increase if we haven't reached full speed
            if current_pace_speed < target_pace:
                current_pace_speed = target_pace
                train_env.pace_speed = current_pace_speed
                print(f"\nPace increased to {current_pace_speed:.2f} at step {steps_done}")
        
        # Maze density curriculum
        if steps_done > 0 and steps_done % maze_density_interval == 0:
            # Increase maze density gradually
            density_progress = min(1.0, steps_done / (steps * 0.8))  # Cap at 80% of training
            new_density = 1.0 + density_progress * 0.5  # Increase up to 50% denser
            train_env.maze_density = new_density
            print(f"Maze density increased to {new_density:.2f}x at step {steps_done}")
        
        # Starting position difficulty curriculum
        if steps_done > 0 and steps_done % difficulty_interval == 0:
            # Increase start position difficulty gradually
            difficulty_progress = min(1.0, steps_done / (steps * 0.6))  # Cap at 60% of training
            new_difficulty = difficulty_progress
            train_env.start_position_difficulty = new_difficulty
            print(f"Starting position difficulty increased to {new_difficulty:.2f} at step {steps_done}")
        
        # Train for one episode
        episode_start_time = time.time()
        state, _ = train_env.reset()  # Gymnasium returns (obs, info)
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Enhanced tracking for Phase 2
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Track actions for vertical movement rate
        episode_pace_distances = []
        
        while not done and episode_length < train_env.max_steps:
            # Select and take action
            action = agent.train_step(state)
            action_counts[action] += 1
            
            next_state, reward, terminated, truncated, info = train_env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, terminated or truncated)
            
            # Track metrics
            episode_reward += reward
            episode_pace_distances.append(info['distance_to_pace'])
            
            # Update state
            state = next_state
            done = terminated or truncated
            episode_length += 1
            
            # Force render every few steps if render is enabled
            if render_mode is not None and episode_length % 5 == 0:
                train_env.render()
        
        # Episode complete, update metrics
        episode_duration = time.time() - episode_start_time
        steps_done = agent.steps_done
        episode += 1
        
        # Calculate additional metrics
        avg_pace_distance = np.mean(episode_pace_distances) if episode_pace_distances else 0
        total_actions = sum(action_counts.values())
        vertical_actions = action_counts[0] + action_counts[2]  # UP + DOWN
        vertical_rate = vertical_actions / total_actions if total_actions > 0 else 0
        oscillation_count = train_env.oscillation_count
        
        # Update enhanced metrics
        enhanced_metrics['pace_distances'].append(avg_pace_distance)
        enhanced_metrics['survival_times'].append(episode_length / 10)  # Assuming 10 FPS
        enhanced_metrics['oscillation_counts'].append(oscillation_count)
        enhanced_metrics['vertical_movement_rates'].append(vertical_rate)
        
        # Update agent's training metrics
        if not hasattr(agent.training_metrics, 'pace_distances'):
            # Add these fields if they don't exist
            agent.training_metrics['pace_distances'] = []
            agent.training_metrics['oscillation_counts'] = []
            
        agent.training_metrics['pace_distances'].append(avg_pace_distance)
        agent.training_metrics['oscillation_counts'].append(oscillation_count)
        agent.training_metrics['episode_rewards'].append(episode_reward)
        agent.training_metrics['episode_lengths'].append(episode_length)
        agent.training_metrics['exploration_rate'].append(agent.epsilon)
        
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
                print(f"Current reward: {episode_reward:.2f} | Pace speed: {train_env.pace_speed:.2f} | "
                      f"Avg distance to pace: {avg_pace_distance:.1f} | Oscillations: {oscillation_count}")
                print(f"Current epsilon: {agent.epsilon:.3f} | Vertical movement: {vertical_rate:.2%}")
                if best_eval_reward != float('-inf'):
                    print(f"Best evaluation reward: {best_eval_reward:.2f} | Best survival time: {best_eval_survival_time:.1f}s")
                print("-" * 80)
            
            last_progress_step = steps_done
            last_progress_time = current_time
        
        # Detailed logging (less frequent)
        elif episode % log_interval == 0:
            print(f"Episode {episode} | Steps: {steps_done}/{steps} | "
                 f"Reward: {episode_reward:.2f} | Distance: {avg_pace_distance:.1f} | Eps: {agent.epsilon:.3f}")
        
        # Evaluation - check if we've crossed an evaluation threshold
        next_eval_step = (steps_done // eval_interval) * eval_interval
        if next_eval_step > 0 and steps_done >= next_eval_step and (last_eval_step := getattr(agent, 'last_eval_step', -eval_interval)) < next_eval_step:
            print("\n=== Running evaluation... ===")
            eval_start_time = time.time()
            eval_stats = agent.evaluate(eval_env, num_episodes=10)
            eval_duration = time.time() - eval_start_time
            # Store the last evaluation step to avoid repeated evaluations
            agent.last_eval_step = next_eval_step
            
            progress_percent = (steps_done / steps) * 100
            print(f"[Progress: {progress_percent:.1f}%] Evaluation Results:")
            print(f"Avg Reward: {eval_stats['avg_reward']:.2f} ± {eval_stats['std_reward']:.2f} | "
                 f"Avg Length: {eval_stats['avg_length']:.2f} | "
                 f"Time: {eval_duration:.2f}s")
            
            # Convert episode length to survival time in seconds
            avg_survival_time = eval_stats['avg_length'] / 10  # Assuming 10 FPS
            
            # Save if best model so far based on combination of reward and survival time
            combined_score = eval_stats['avg_reward'] + avg_survival_time * 0.1  # Weighted sum
            
            if not hasattr(agent, 'best_combined_score') or combined_score > agent.best_combined_score:
                agent.best_combined_score = combined_score
                best_eval_reward = eval_stats['avg_reward']
                best_eval_survival_time = avg_survival_time
                agent.save(os.path.join(run_dir, 'best_model'))
                print(f"New best model saved with score {combined_score:.2f} "
                     f"(reward: {best_eval_reward:.2f}, survival: {best_eval_survival_time:.1f}s)")
            
            # Calculate Phase 2 specific metrics
            avg_pace_distance = np.mean([info.get('distance_to_pace', 0) for info in eval_stats.get('infos', [])])
            
            # Build extended evaluation stats for Phase 2 criteria
            eval_stats_extended = {
                'avg_reward': eval_stats.get('avg_reward', 0.0),
                'avg_score': eval_stats.get('avg_score', 0.0),
                'avg_survival_time': avg_survival_time,
                'avg_pace_distance': avg_pace_distance,
                'vertical_movement_rate': eval_stats.get('vertical_movement_rate', 0.0),
                'oscillation_rate': eval_stats.get('oscillation_rate', 1.0)  # Default to worst case
            }
            
            # Print available metrics for Phase 2
            print("Phase 2 Evaluation Metrics:")
            print(f"Avg Survival Time: {avg_survival_time:.1f}s (Target: ≥120s)")
            print(f"Avg Distance to Pace: {avg_pace_distance:.1f} (Target: >100)")
            print(f"Vertical Movement Rate: {eval_stats_extended['vertical_movement_rate']:.2%} (Target: 25-45%)")
            print(f"Oscillation Rate: {eval_stats_extended['oscillation_rate']:.2%} (Target: <5%)")
            
            # Check Phase 2 success criteria
            success, criteria_results = check_phase_completion(eval_stats_extended, success_criteria)
            
            # Save criteria results
            with open(os.path.join(run_dir, f'criteria_eval_step_{steps_done}.json'), 'w') as f:
                json.dump(criteria_results, f, indent=2)
            
            if success:
                print("\n🎉 SUCCESS: Phase 2 completion criteria met! 🎉")
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
    
    # Use fewer episodes for quicker evaluation in short training runs
    eval_episodes = min(5, max(2, steps // 250))
    final_eval_stats = agent.evaluate(eval_env, num_episodes=eval_episodes)
    
    # Calculate Phase 2 specific metrics for final evaluation
    avg_pace_distance = np.mean([info.get('distance_to_pace', 0) for info in final_eval_stats.get('infos', [])])
    avg_survival_time = final_eval_stats['avg_length'] / 10  # Assuming 10 FPS
    
    print(f"\nFINAL EVALUATION RESULTS:")
    print(f"Avg Reward: {final_eval_stats['avg_reward']:.2f} ± {final_eval_stats['std_reward']:.2f}")
    print(f"Avg Episode Length: {final_eval_stats['avg_length']:.2f} ± {final_eval_stats['std_length']:.2f}")
    print(f"Avg Survival Time: {avg_survival_time:.1f}s")
    print(f"Avg Distance to Pace: {avg_pace_distance:.1f}")
    print(f"Episodes completed during training: {episode}")
    
    # Update best reward if final evaluation is better
    combined_score = final_eval_stats['avg_reward'] + avg_survival_time * 0.1
    if not hasattr(agent, 'best_combined_score') or combined_score > agent.best_combined_score:
        agent.best_combined_score = combined_score
        best_eval_reward = final_eval_stats['avg_reward']
        best_eval_survival_time = avg_survival_time
        agent.save(os.path.join(run_dir, 'best_model'))
        print(f"New best model saved from final evaluation with score {combined_score:.2f}")
    
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
    print(f"Best survival time: {best_eval_survival_time:.1f}s")
    print(f"Total duration: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save final stats
    final_stats = {
        'episodes': episode,
        'steps': steps_done,
        'best_eval_reward': float(best_eval_reward),
        'best_survival_time': float(best_eval_survival_time),
        'training_duration_seconds': total_duration,
        'training_complete': training_complete
    }
    
    with open(os.path.join(run_dir, 'training_summary.json'), 'w') as f:
        json.dump(final_stats, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Phase 2 of Infinite Maze AI')
    parser.add_argument('--checkpoint-path', type=str, required=True, 
                       help='Path to the best Phase 1 checkpoint directory to start from')
    parser.add_argument('--steps', type=int, default=1200000, 
                       help='Number of training steps (1.2M default per training plan)')
    parser.add_argument('--log-interval', type=int, default=100, 
                       help='Episodes between logging')
    parser.add_argument('--eval-interval', type=int, default=20000, 
                       help='Steps between evaluations')
    parser.add_argument('--save-interval', type=int, default=50000, 
                       help='Steps between checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default='ai/checkpoints/phase2', 
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to use')
    parser.add_argument('--render', action='store_true', 
                       help='Enable rendering')
    parser.add_argument('--pace-speed', type=float, default=0.2, 
                       help='Initial pace speed (0.2 = 20% of normal)')
    parser.add_argument('--pace-acceleration', action='store_true', 
                       help='Enable pace acceleration')
    parser.add_argument('--no-gradual-pace-increase', dest='gradually_increase_pace', action='store_false',
                       help='Disable gradual pace speed increase during training')
    parser.set_defaults(gradually_increase_pace=True)
    
    args = parser.parse_args()
    
    render_mode = 'human' if args.render else None
    
    # Auto-adjust intervals for small step counts
    if args.steps < args.eval_interval:
        print(f"NOTE: Auto-adjusting evaluation interval to {args.steps // 5} for short training run")
        args.eval_interval = max(1000, args.steps // 5)
        
    if args.steps < args.save_interval:
        print(f"NOTE: Auto-adjusting save interval to {args.steps // 2} for short training run")
        args.save_interval = max(5000, args.steps // 2)
    
    print("Starting Phase 2 training with parameters from the training plan")
    print(f"Starting from Phase 1 checkpoint: {args.checkpoint_path}")
    print(f"Training for {args.steps} steps with evaluations every {args.eval_interval} steps")
    print(f"Using device: {args.device}")
    print(f"Pace settings: Initial speed={args.pace_speed}, "
         f"Acceleration={'enabled' if args.pace_acceleration else 'disabled'}, "
         f"Gradual increase={'enabled' if args.gradually_increase_pace else 'disabled'}")
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    
    train_phase_2(
        checkpoint_path=args.checkpoint_path,
        steps=args.steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        render_mode=render_mode,
        pace_start_speed=args.pace_speed,
        pace_acceleration=args.pace_acceleration,
        gradually_increase_pace=args.gradually_increase_pace
    )
