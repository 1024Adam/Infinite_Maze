"""
Training script for Infinite Maze RL agent using DQN algorithm.

This script sets up the training environment, creates a DQN agent,
and trains it to play the Infinite Maze game.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
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

def train_dqn_agent():
    """Train a DQN agent on the Infinite Maze environment."""
    
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
    print("Creating DQN model...")
    model = DQN(**model_config)
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./rl/models/',
        log_path='./rl/logs/',
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
            total_timesteps=500000,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        model.save("rl/models/dqn_infinite_maze_final")
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        model.save("rl/models/dqn_infinite_maze_interrupted")
    
    except Exception as e:
        print(f"Training failed with error: {e}")
        model.save("rl/models/dqn_infinite_maze_error")
    
    finally:
        env.close()
        eval_env.close()
    
    return model

def test_trained_agent(model_path: str = "rl/models/best_model.zip", num_episodes: int = 5):
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
    # Create necessary directories
    os.makedirs("rl/models", exist_ok=True)
    os.makedirs("rl/logs", exist_ok=True)
    os.makedirs("rl/tensorboard_logs", exist_ok=True)
    
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
    
    # Test the trained agent (uncomment to test with rendering)
    # print("\nTesting trained agent...")
    # test_trained_agent()
    
    print("\nTraining pipeline completed!")
    print("To test the trained agent with visualization, run:")
    print("python rl/train_agent.py --test")
