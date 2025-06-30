"""
Training script for Infinite Maze RL agent using DQN algorithm with enhanced navigation rewards.

This script sets up the training environment with improved reward functions that encourage
intelligent navigation behavior, including vertical movement when stuck going right.

Usage:
    python train_agent.py                    # Train new model
    python train_agent.py --test             # Test existing model
    python train_agent.py --continue model   # Continue training from model
"""

import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.utils import get_linear_fn
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.environment import InfiniteMazeEnv

def create_env():
    """Create a single environment instance."""
    return InfiniteMazeEnv(headless=True)

def create_monitored_env():
    """Create a monitored environment instance."""
    env = InfiniteMazeEnv(headless=True)
    return Monitor(env, "rl/logs/training")

def create_eval_env():
    """Create an evaluation environment instance."""
    env = InfiniteMazeEnv(headless=True)
    return Monitor(env, "rl/logs/evaluation")

def train_dqn_agent(continue_from_model: str = None, total_timesteps: int = 500000):
    """Train a DQN agent on the Infinite Maze environment.
    
    Args:
        continue_from_model: Path to a previously saved model to continue training from
        total_timesteps: Total number of timesteps to train for
    """
    
    # Create training environment - wrap individual envs with Monitor first, then vectorize
    print("Creating training environment...")
    env = DummyVecEnv([create_monitored_env])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([create_eval_env])
    
    # Define DQN hyperparameters
    model_config = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': 1e-4,
        'buffer_size': 100000,
        'learning_starts': 10000,
        'batch_size': 32,
        'tau': 1.0,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 1000,
        'exploration_fraction': 0.3,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.05,
        'max_grad_norm': 10,
        'tensorboard_log': "./rl/tensorboard_logs/",
        'verbose': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {model_config['device']}")
    
    # Create DQN model
    if continue_from_model:
        print(f"Loading existing model from {continue_from_model}...")
        model = DQN.load(continue_from_model, env=env)
        print("Continuing training from loaded model...")
        
        # Get the current number of timesteps the model has been trained for
        current_timesteps = getattr(model, 'num_timesteps', 0)
        print(f"Model has been trained for {current_timesteps} timesteps")
        
        # Calculate current exploration rate based on the original exploration schedule
        # These should match the original training config
        exploration_fraction = 0.3
        exploration_initial_eps = 1.0
        exploration_final_eps = 0.05
        
        # Use the original training timesteps (500000) for exploration calculation
        original_total_timesteps = 500000
        exploration_timesteps = int(exploration_fraction * original_total_timesteps)  # 150000
        
        # Calculate what the exploration rate should be at this point
        if current_timesteps < exploration_timesteps:
            # Linear decay from initial to final
            exploration_progress = current_timesteps / exploration_timesteps
            current_eps = exploration_initial_eps - (exploration_initial_eps - exploration_final_eps) * exploration_progress
        else:
            # Exploration phase is over, use final epsilon
            current_eps = exploration_final_eps
        
        print(f"Calculated current exploration rate: {current_eps:.4f}")
        print(f"Model's stored exploration rate: {model.exploration_rate:.4f}")
        
        # Create a new exploration schedule that starts from the current exploration rate
        # and continues to decay (if still in exploration phase) or stays constant
        if current_timesteps < exploration_timesteps:
            # Still in exploration phase - continue the decay for remaining timesteps
            remaining_exploration_timesteps = exploration_timesteps - current_timesteps
            exploration_fraction_remaining = remaining_exploration_timesteps / total_timesteps
            
            print(f"Continuing exploration decay from {current_eps:.4f} to {exploration_final_eps:.4f}")
            print(f"Remaining exploration fraction: {exploration_fraction_remaining:.3f}")
            
            # Update the model's exploration parameters to continue from current point
            model.exploration_initial_eps = current_eps
            model.exploration_final_eps = exploration_final_eps
            model.exploration_fraction = exploration_fraction_remaining
            
            # Create new exploration schedule using get_linear_fn
            # Note: get_linear_fn uses progress_remaining (1.0 -> 0.0)
            # start: value when progress_remaining = 1.0 (beginning of training)
            # end: value when progress_remaining = (1.0 - end_fraction) 
            model.exploration_schedule = get_linear_fn(
                start=current_eps,
                end=exploration_final_eps,
                end_fraction=exploration_fraction_remaining
            )
        else:
            # Past exploration phase - use constant final epsilon
            print(f"Exploration phase completed, using constant rate: {exploration_final_eps:.4f}")
            
            model.exploration_initial_eps = exploration_final_eps
            model.exploration_final_eps = exploration_final_eps
            model.exploration_fraction = 0.0  # No exploration decay needed
            
            # Create constant exploration schedule
            model.exploration_schedule = get_linear_fn(
                start=exploration_final_eps,
                end=exploration_final_eps,
                end_fraction=1.0
            )
        
        # Set the current exploration rate
        model.exploration_rate = current_eps
        
        print(f"Updated exploration parameters:")
        print(f"  Initial eps: {model.exploration_initial_eps:.4f}")
        print(f"  Final eps: {model.exploration_final_eps:.4f}")
        print(f"  Fraction: {model.exploration_fraction:.4f}")
        print(f"  Current rate: {model.exploration_rate:.4f}")
        
        # Test the exploration schedule at a few points to verify it's working
        print(f"Exploration schedule test:")
        test_progress = [1.0, 0.75, 0.5, 0.25, 0.0]
        for progress in test_progress:
            eps = model.exploration_schedule(progress)
            print(f"  Progress {progress:.2f}: eps = {eps:.4f}")
        
        # Optionally adjust other hyperparameters for continued training
        # You might want to use a lower learning rate for fine-tuning
        # model.learning_rate = 5e-5  # Half the original learning rate
        
    else:
        print("Creating new DQN model...")
        model = DQN(**model_config)
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Optionally stop training if reward threshold is reached
    # stop_callback = StopTrainingOnRewardThreshold(
    #     reward_threshold=1000,
    #     verbose=1
    # )
    
    callbacks = [eval_callback]
    
    # Train the model
    print("Starting training...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        model.save("models/navigation_agent_final")
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        model.save("models/navigation_agent_interrupted")
    
    except Exception as e:
        print(f"Training failed with error: {e}")
        model.save("models/navigation_agent_error")
    
    finally:
        env.close()
        eval_env.close()
    
    return model

def test_trained_agent(model_path: str = "models/best_model.zip", num_episodes: int = 5):
    """Test a trained agent on the environment."""
    
    print(f"Loading model from {model_path}")
    model = DQN.load(model_path)
    
    # Create test environment with rendering
    env = InfiniteMazeEnv(render_mode='human', headless=False)
    
    total_rewards = []
    total_scores = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            # Render the game
            env.render()
            
            if done:
                final_score = info.get('score', 0)
                print(f"Episode {episode + 1} finished:")
                print(f"  Total Reward: {episode_reward:.2f}")
                print(f"  Final Score: {final_score}")
                print(f"  Survival Time: {info.get('survival_time', 0)}")
                
                total_rewards.append(episode_reward)
                total_scores.append(final_score)
    
    print(f"\nTest Results over {num_episodes} episodes:")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Score: {np.mean(total_scores):.2f} ± {np.std(total_scores):.2f}")
    print(f"Best Score: {np.max(total_scores)}")
    
    env.close()

def create_baseline_comparison():
    """Create baseline comparison with random agent."""
    
    print("Testing random baseline agent...")
    env = InfiniteMazeEnv(headless=True)
    
    num_episodes = 10
    random_rewards = []
    random_scores = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if done:
                random_rewards.append(episode_reward)
                random_scores.append(info.get('score', 0))
    
    print(f"Random Baseline Results over {num_episodes} episodes:")
    print(f"Average Reward: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    print(f"Average Score: {np.mean(random_scores):.2f} ± {np.std(random_scores):.2f}")
    print(f"Best Score: {np.max(random_scores)}")
    
    env.close()
    return np.mean(random_rewards), np.mean(random_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test Infinite Maze RL agent')
    parser.add_argument('--test', action='store_true', 
                      help='Test existing model instead of training')
    parser.add_argument('--continue', dest='continue_model', type=str,
                      help='Continue training from specified model path')
    parser.add_argument('--model', '-m', type=str, default='models/best_model.zip',
                      help='Model path for testing (default: models/best_model.zip)')
    parser.add_argument('--episodes', '-e', type=int, default=5,
                      help='Number of episodes for testing (default: 5)')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    if args.test:
        print("Testing trained agent...")
        test_trained_agent(args.model, args.episodes)
        
    elif args.continue_model:
        print("Infinite Maze RL Continued Training")
        print("=" * 40)
        print(f"Continuing from: {args.continue_model}")
        
        # Test environment first
        print("Testing environment...")
        env = InfiniteMazeEnv(headless=True)
        obs, info = env.reset()
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print(f"Initial observation shape: {obs.shape}")
        env.close()
        
        # Continue training
        print("\nContinuing DQN training...")
        model = train_dqn_agent(continue_from_model=args.continue_model)
        
        print("\nContinued training completed!")
        
    else:
        print("Infinite Maze RL Training")
        print("=" * 40)
        
        # Test environment first
        print("Testing environment...")
        env = InfiniteMazeEnv(headless=True)
        obs, info = env.reset()
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print(f"Initial observation shape: {obs.shape}")
        env.close()
        
        # Create baseline
        baseline_reward, baseline_score = create_baseline_comparison()
        
        # Train the agent
        print("\nStarting DQN training...")
        model = train_dqn_agent()
        
        print("\nTraining pipeline completed!")
        print("To test the trained agent with visualization, run:")
        print("python train_agent.py --test")
        print("\nTo continue training from the best model, run:")
        print("python train_agent.py --continue models/best_model.zip")
        print("Or from the final model:")
        print("python train_agent.py --continue models/navigation_agent_final.zip")
