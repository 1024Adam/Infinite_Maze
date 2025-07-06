"""
Training script for Phase 1 of the Infinite Maze AI.

This script implements the Phase 1 training as described in the training plan,
focusing on basic navigation with static mazes and no advancing pace line.
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

def train_phase_1(steps: int = 600000,  # Updated to 600K steps per training plan
                  log_interval: int = 100,
                  eval_interval: int = 10000,
                  save_interval: int = 50000,
                  checkpoint_dir: str = 'ai/checkpoints/phase1',
                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                  render_mode: str = None) -> None:
    """
    Train the agent for Phase 1 of the curriculum based on the updated training plan.
    
    Args:
        steps: Total training steps (600K per updated plan)
        log_interval: Episodes between logging
        eval_interval: Steps between evaluations
        save_interval: Steps between saving checkpoints
        checkpoint_dir: Directory to save checkpoints
        device: Device to use for training ('cuda' or 'cpu')
        render_mode: Rendering mode (None, 'human', 'rgb_array')
    """
    # Adjust evaluation and save intervals for short training sessions
    if steps < eval_interval:
        eval_interval = max(500, steps // 2)  # Ensure at least one evaluation
    if steps < save_interval:
        save_interval = max(1000, steps // 2)  # Ensure at least one save checkpoint
    print(f"Starting Phase 1 training on device: {device}")
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
        'timestamp': timestamp
    }
    
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize the training environment (Phase 1 specific)
    train_env = InfiniteMazeEnv(
        training_phase=1,
        use_maze_from_start=True,  # Start with maze structures for training
        pace_enabled=False,        # No pace line in Phase 1
        render_mode=render_mode,
        grid_size=11,             # 11x11 grid as specified in training plan
        max_steps=10000           # Allow longer episodes for better learning
    )
    
    # Initialize a separate evaluation environment
    eval_env = InfiniteMazeEnv(
        training_phase=1,
        use_maze_from_start=True,
        pace_enabled=False,
        render_mode=None,  # No rendering for evaluation
        grid_size=11       # Same grid size as training
    )
    
    # Get the observation shape from the environment
    observation, _ = train_env.reset()  # Gymnasium returns (obs, info)
    state_shape = {
        'grid': observation['grid'].shape,
        'numerical': observation['numerical'].shape
    }
    
    # Initialize the agent with enhanced parameters based on evaluation recommendations
    agent = RainbowDQNAgent(
        state_shape=state_shape,
        num_actions=train_env.action_space.n,
        learning_rate=3.0e-4,  # Slightly increased from 2.5e-4 for faster learning
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=250000,  # Increased from 150K to 250K for more exploration
        epsilon_warmup=50000,  # Kept warmup period for better initial exploration
        target_update=5000,    # More frequent updates (decreased from 8000)
        batch_size=256,        # Increased from 128 for more stable learning
        replay_capacity=1000000,
        device=device,
        use_dueling=True,      # Enabled dueling architecture for better performance
        n_steps=3,             # Increased from 2 to 3 for better credit assignment
        alpha=0.7,             # Increased from 0.6 for stronger prioritization
        beta_start=0.4         # Kept the same IS correction exponent
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
    
    print("Starting training loop...")
    print(f"Target: {steps} steps | Evaluations every {eval_interval} steps | Checkpoints every {save_interval} steps")
    
    while steps_done < steps and not training_complete:
        # Train for one episode
        episode_start_time = time.time()
        episode_stats = agent.train_episode(train_env)
        episode_duration = time.time() - episode_start_time
        
        steps_done = agent.steps_done
        episode += 1
        
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
                print(f"Current reward: {episode_stats['reward']:.2f} | Exploration rate: {agent.epsilon:.3f}")
                if best_eval_reward != float('-inf'):
                    print(f"Best evaluation reward so far: {best_eval_reward:.2f}")
                print("-" * 80)
            
            last_progress_step = steps_done
            last_progress_time = current_time
        
        # Detailed logging (less frequent)
        elif episode % log_interval == 0:
            print(f"Episode {episode} | Steps: {steps_done}/{steps} | "
                 f"Reward: {episode_stats['reward']:.2f} | Eps: {agent.epsilon:.3f}")
        
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
            
            # Save if best model so far
            if eval_stats['avg_reward'] > best_eval_reward:
                best_eval_reward = eval_stats['avg_reward']
                agent.save(os.path.join(run_dir, 'best_model'))
                print(f"New best model saved with average reward: {best_eval_reward:.2f}")
            
            # Check success criteria
            # For a full implementation, we'd need to calculate:
            # - Forward movement success rate
            # - Collision rate
            # But for simplicity, we'll just use the reward for now
            
            # Now using actual calculated metrics from evaluation
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
    
    # Use fewer episodes and steps for quicker evaluation in short training runs
    eval_episodes = min(5, max(2, steps // 250))
    final_eval_stats = agent.evaluate(eval_env, num_episodes=eval_episodes)
    
    print(f"\nFINAL EVALUATION RESULTS:")
    print(f"Avg Reward: {final_eval_stats['avg_reward']:.2f} ± {final_eval_stats['std_reward']:.2f}")
    print(f"Avg Episode Length: {final_eval_stats['avg_length']:.2f} ± {final_eval_stats['std_length']:.2f}")
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
        'training_complete': training_complete
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
    
    args = parser.parse_args()
    
    render_mode = 'human' if args.render else None
    
    # Auto-adjust intervals for small step counts
    if args.steps < args.eval_interval:
        print(f"NOTE: Auto-adjusting evaluation interval to {args.steps // 5} for short training run")
        args.eval_interval = max(100, args.steps // 5)
        
    if args.steps < args.save_interval:
        print(f"NOTE: Auto-adjusting save interval to {args.steps // 2} for short training run")
        args.save_interval = max(500, args.steps // 2)
    
    print("Starting Phase 1 training with updated parameters from the training plan")
    print(f"Training for {args.steps} steps with evaluations every {args.eval_interval} steps")
    print(f"Using device: {args.device}")
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    
    train_phase_1(
        steps=args.steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        render_mode=render_mode
    )
