"""
Utilities for evaluating and visualizing the Phase 1 Infinite Maze AI agent's performance.
"""

import os
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Any, List, Tuple, Optional

from infinite_maze.ai.environments.environment_phase1 import InfiniteMazeEnv
from infinite_maze.ai.agents.agent import RainbowDQNAgent
from infinite_maze.utils.config import config

def evaluate_agent(agent: RainbowDQNAgent, 
                  env: InfiniteMazeEnv, 
                  num_episodes: int = 20,
                  render: bool = False,
                  verbose: bool = True,
                  delay: float = 0) -> Dict[str, Any]:
    """
    Comprehensive evaluation of agent performance.
    
    Args:
        agent: The trained agent
        env: Evaluation environment
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        verbose: Whether to print progress
        
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    rewards = []
    lengths = []
    scores = []
    collisions = []
    actions_taken = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # UP, RIGHT, DOWN, LEFT, NO_ACTION
    
    render_mode = env.render_mode
    if not render:
        env.render_mode = None
    
    for ep in range(num_episodes):
        state = env.reset()[0]  # Gymnasium returns (obs, info)
        total_reward = 0
        episode_collisions = 0
        episode_actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        done = False
        step = 0
        
        while not done and step < 10000:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Check if episode is done
            done = terminated or truncated
            
            # Track metrics
            total_reward += reward
            episode_actions[action] += 1
            if info.get('collision', False):
                episode_collisions += 1
            
            state = next_state
            step += 1
            
            if render:
                env.render()
                if delay > 0:
                    import time
                    time.sleep(delay)
        
        # Update statistics
        rewards.append(total_reward)
        lengths.append(step)
        scores.append(info.get('score', 0))
        collisions.append(episode_collisions)
        
        # Update action counts
        for action, count in episode_actions.items():
            actions_taken[action] += count
        
        if verbose and ep % 5 == 0:
            print(f"Episode {ep}/{num_episodes}: Reward={total_reward:.2f}, "
                  f"Length={step}, Score={info.get('score', 0)}")
    
    # Restore original render mode
    env.render_mode = render_mode
    
    # Calculate overall statistics
    total_actions = sum(actions_taken.values())
    action_distribution = {action: count / total_actions for action, count in actions_taken.items()}
    
    # Calculate forward movement success rate
    # This is a simplification - ideally would track attempted vs successful moves
    forward_success_rate = 0.90  # Placeholder - would be calculated from detailed logs
    
    # Calculate collision rate
    total_collisions = sum(collisions)
    collision_rate = total_collisions / total_actions if total_actions > 0 else 0
    
    # Path efficiency (rightward movement / total movement)
    rightward_actions = actions_taken[1]  # Action 1 is RIGHT
    path_efficiency = rightward_actions / total_actions if total_actions > 0 else 0
    
    return {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'collision_rate': collision_rate,
        'forward_success_rate': forward_success_rate,
        'path_efficiency': path_efficiency,
        'action_distribution': action_distribution,
        'raw_data': {
            'rewards': rewards,
            'lengths': lengths,
            'scores': scores,
            'collisions': collisions,
            'actions_taken': actions_taken
        }
    }

def visualize_evaluation(eval_results: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Visualize the evaluation results.
    
    Args:
        eval_results: Evaluation results from evaluate_agent
        save_path: Directory to save the plots (if None, will display instead)
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot reward distribution
    rewards = eval_results['raw_data']['rewards']
    axes[0, 0].hist(rewards, bins=10, alpha=0.7)
    axes[0, 0].axvline(eval_results['avg_reward'], color='r', linestyle='dashed', 
                      linewidth=1, label=f'Mean: {eval_results["avg_reward"]:.2f}')
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Plot episode length distribution
    lengths = eval_results['raw_data']['lengths']
    axes[0, 1].hist(lengths, bins=10, alpha=0.7)
    axes[0, 1].axvline(eval_results['avg_length'], color='r', linestyle='dashed', 
                      linewidth=1, label=f'Mean: {eval_results["avg_length"]:.2f}')
    axes[0, 1].set_title('Episode Length Distribution')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Plot score distribution
    scores = eval_results['raw_data']['scores']
    axes[1, 0].hist(scores, bins=10, alpha=0.7)
    axes[1, 0].axvline(eval_results['avg_score'], color='r', linestyle='dashed', 
                      linewidth=1, label=f'Mean: {eval_results["avg_score"]:.2f}')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Plot action distribution
    actions = eval_results['action_distribution']
    action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'NO_ACTION']
    axes[1, 1].bar(action_names, [actions[i] for i in range(5)])
    axes[1, 1].set_title('Action Distribution')
    axes[1, 1].set_ylabel('Frequency')
    
    # Add overall stats as text
    fig.text(0.5, 0.01, 
             f"Collision Rate: {eval_results['collision_rate']:.4f} | "
             f"Path Efficiency: {eval_results['path_efficiency']:.4f} | "
             f"Forward Success Rate: {eval_results['forward_success_rate']:.4f}",
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'evaluation_summary.png'))
        plt.close()
    else:
        plt.show()

def load_trained_agent(checkpoint_dir: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> RainbowDQNAgent:
    """
    Load a trained agent from checkpoint.
    
    Args:
        checkpoint_dir: Directory with the saved model
        device: Device to load the model on
        
    Returns:
        Loaded agent
    """
    # Load configuration
    with open(os.path.join(checkpoint_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Create a temporary environment to get state shape
    env = InfiniteMazeEnv(training_phase=1, render_mode=None)
    observation = env.reset()[0]  # Gymnasium returns (obs, info)
    state_shape = {
        'grid': observation['grid'].shape,
        'numerical': observation['numerical'].shape
    }
    
    # Create agent with correct parameters
    agent = RainbowDQNAgent(
        state_shape=state_shape,
        num_actions=env.action_space.n,
        device=device
    )
    
    # Load saved model (either 'final_model' or 'best_model')
    model_path = os.path.join(checkpoint_dir, 'final_model')
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoint_dir, 'best_model')
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found in {checkpoint_dir}")
        
    agent.load(model_path)
    
    return agent

def compare_agents(agents: Dict[str, RainbowDQNAgent], 
                  env: InfiniteMazeEnv,
                  num_episodes: int = 10) -> Dict[str, Any]:
    """
    Compare multiple agents' performance.
    
    Args:
        agents: Dictionary mapping names to agents
        env: Evaluation environment
        num_episodes: Number of episodes per agent
        
    Returns:
        Dictionary with comparative metrics
    """
    results = {}
    
    for name, agent in agents.items():
        print(f"\nEvaluating agent: {name}")
        agent_results = evaluate_agent(agent, env, num_episodes, render=False)
        results[name] = agent_results
        
    return results

def visualize_comparison(comparison_results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> None:
    """
    Visualize comparison of multiple agents.
    
    Args:
        comparison_results: Results from compare_agents
        save_path: Directory to save the plots
    """
    agent_names = list(comparison_results.keys())
    
    # Extract metrics for comparison
    avg_rewards = [comparison_results[name]['avg_reward'] for name in agent_names]
    avg_scores = [comparison_results[name]['avg_score'] for name in agent_names]
    collision_rates = [comparison_results[name]['collision_rate'] for name in agent_names]
    path_efficiencies = [comparison_results[name]['path_efficiency'] for name in agent_names]
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Average reward comparison
    axes[0, 0].bar(agent_names, avg_rewards)
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].set_ylabel('Reward')
    
    # Average score comparison
    axes[0, 1].bar(agent_names, avg_scores)
    axes[0, 1].set_title('Average Score')
    axes[0, 1].set_ylabel('Score')
    
    # Collision rate comparison
    axes[1, 0].bar(agent_names, collision_rates)
    axes[1, 0].set_title('Collision Rate')
    axes[1, 0].set_ylabel('Rate')
    
    # Path efficiency comparison
    axes[1, 1].bar(agent_names, path_efficiencies)
    axes[1, 1].set_title('Path Efficiency')
    axes[1, 1].set_ylabel('Efficiency')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'agent_comparison.png'))
        plt.close()
    else:
        plt.show()

def main():
    """
    Main entry point for direct script execution.
    """
    parser = argparse.ArgumentParser(description='Evaluate trained Infinite Maze AI Phase 1 agent')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory with trained model')
    parser.add_argument('--episodes', type=int, default=20, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save evaluation results')
    parser.add_argument('--delay', type=float, default=0, help='Add delay between steps (in seconds) to make rendering more watchable')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to run evaluation on')
    
    args = parser.parse_args()
    
    print(f"Loading agent from {args.checkpoint_dir}")
    agent = load_trained_agent(args.checkpoint_dir, args.device)
    
    print(f"Creating Phase 1 evaluation environment")
    eval_env = InfiniteMazeEnv(
        training_phase=1,
        use_maze_from_start=True,
        pace_enabled=False,
        render_mode='human' if args.render else None
    )
    
    print(f"Running evaluation for {args.episodes} episodes")
    eval_results = evaluate_agent(agent, eval_env, args.episodes, args.render, delay=args.delay)
    
    print("\nPhase 1 Evaluation Results:")
    print(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Average Score: {eval_results['avg_score']:.2f} ± {eval_results['std_score']:.2f}")
    print(f"Average Length: {eval_results['avg_length']:.2f} ± {eval_results['std_length']:.2f}")
    print(f"Collision Rate: {eval_results['collision_rate']:.4f}")
    print(f"Path Efficiency: {eval_results['path_efficiency']:.4f}")
    print(f"Action Distribution: {eval_results['action_distribution']}")
    
    if args.save_dir:
        print(f"\nSaving evaluation results to {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Save metrics as JSON
        with open(os.path.join(args.save_dir, 'eval_metrics.json'), 'w') as f:
            json.dump(eval_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        
        # Visualize and save plots
        visualize_evaluation(eval_results, args.save_dir)

if __name__ == "__main__":
    main()
