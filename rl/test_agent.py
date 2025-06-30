"""
Test and demonstration script for trained Infinite Maze RL agents with enhanced navigation.

This script loads trained models and allows testing them either with
or without visualization. Works with models trained using the enhanced
reward system that includes navigation intelligence.
"""

import argparse
import os
import sys
import numpy as np
import time
from stable_baselines3 import DQN

# Add parent directory to path for infinite_maze imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import InfiniteMazeEnv, InfiniteMazeWrapper

def test_agent_with_visualization(model_path: str, num_episodes: int = 3):
    """Test agent with pygame visualization."""
    print(f"Loading model: {model_path}")
    model = DQN.load(model_path)
    
    env = InfiniteMazeEnv(render_mode='human', headless=False)
    
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}")
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Render
            env.render()
            time.sleep(0.016)  # ~60 FPS
            
            if terminated or truncated:
                print(f"Episode {episode + 1} Results:")
                print(f"  Steps: {step_count}")
                print(f"  Total Reward: {episode_reward:.2f}")
                print(f"  Final Score: {info['score']}")
                print(f"  Survival Time: {info['survival_time']}")
                print(f"  Final Pace: {info['pace']}")
                break
    
    env.close()

def test_agent_headless(model_path: str, num_episodes: int = 10):
    """Test agent without visualization for performance evaluation."""
    print(f"Loading model: {model_path}")
    model = DQN.load(model_path)
    
    env = InfiniteMazeEnv(headless=True)
    
    results = {
        'rewards': [],
        'scores': [],
        'survival_times': [],
        'steps': []
    }
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                results['rewards'].append(episode_reward)
                results['scores'].append(info['score'])
                results['survival_times'].append(info['survival_time'])
                results['steps'].append(step_count)
                break
    
    env.close()
    
    # Print statistics
    print(f"\nPerformance over {num_episodes} episodes:")
    print(f"Rewards: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
    print(f"Scores: {np.mean(results['scores']):.2f} ± {np.std(results['scores']):.2f}")
    print(f"Survival Time: {np.mean(results['survival_times']):.2f} ± {np.std(results['survival_times']):.2f}")
    print(f"Steps: {np.mean(results['steps']):.2f} ± {np.std(results['steps']):.2f}")
    print(f"Max Score: {np.max(results['scores'])}")
    
    return results

def compare_agents(model_paths: list, num_episodes: int = 10):
    """Compare multiple trained agents."""
    all_results = {}
    
    for model_path in model_paths:
        print(f"\nTesting {model_path}...")
        results = test_agent_headless(model_path, num_episodes)
        all_results[model_path] = results
    
    # Print comparison
    print("\n" + "="*60)
    print("AGENT COMPARISON")
    print("="*60)
    
    for model_path, results in all_results.items():
        print(f"\n{model_path}:")
        print(f"  Avg Reward: {np.mean(results['rewards']):.2f}")
        print(f"  Avg Score: {np.mean(results['scores']):.2f}")
        print(f"  Avg Survival: {np.mean(results['survival_times']):.2f}")
        print(f"  Max Score: {np.max(results['scores'])}")

def test_with_legacy_interface(model_path: str):
    """Test agent using the legacy controlled_run interface."""
    print(f"Testing with legacy interface: {model_path}")
    
    model = DQN.load(model_path)
    wrapper = InfiniteMazeWrapper(model)
    
    # Import the controlled_run function
    from infinite_maze.infinite_maze import controlled_run
    
    # This would normally be called from the main game
    # controlled_run(wrapper, 0)
    print("Legacy interface test completed")

def create_random_baseline(num_episodes: int = 10):
    """Create random baseline for comparison."""
    print("Creating random baseline...")
    
    env = InfiniteMazeEnv(headless=True)
    
    results = {
        'rewards': [],
        'scores': [],
        'survival_times': [],
        'steps': []
    }
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                results['rewards'].append(episode_reward)
                results['scores'].append(info['score'])
                results['survival_times'].append(info['survival_time'])
                results['steps'].append(step_count)
                break
    
    env.close()
    
    print(f"Random Baseline over {num_episodes} episodes:")
    print(f"Rewards: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
    print(f"Scores: {np.mean(results['scores']):.2f} ± {np.std(results['scores']):.2f}")
    print(f"Survival Time: {np.mean(results['survival_times']):.2f} ± {np.std(results['survival_times']):.2f}")
    print(f"Max Score: {np.max(results['scores'])}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test Infinite Maze RL Agents')
    parser.add_argument('--model', type=str, default='models/best_model.zip',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of test episodes')
    parser.add_argument('--headless', action='store_true',
                       help='Run without visualization')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple models')
    parser.add_argument('--baseline', action='store_true',
                       help='Test random baseline')
    
    args = parser.parse_args()
    
    if args.baseline:
        create_random_baseline(args.episodes)
        return
    
    if args.compare:
        compare_agents(args.compare, args.episodes)
        return
    
    if args.headless:
        test_agent_headless(args.model, args.episodes)
    else:
        test_agent_with_visualization(args.model, args.episodes)

if __name__ == "__main__":
    main()
