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
from infinite_maze.ai.agent import RainbowDQNAgent
from infinite_maze.utils.config import config

def evaluate_agent(agent: RainbowDQNAgent, 
                  env: InfiniteMazeEnv, 
                  num_episodes: int = 20,
                  render: bool = False,
                  verbose: bool = True) -> Dict[str, Any]:
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
    right_attempts = 0
    right_successes = 0
    vertical_movements = 0
    oscillation_incidents = 0
    
    render_mode = env.render_mode
    if not render:
        env.render_mode = None
    
    for ep in range(num_episodes):
        state = env.reset()[0]  # Gymnasium returns (obs, info)
        total_reward = 0
        episode_collisions = 0
        episode_actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        episode_right_attempts = 0
        episode_right_successes = 0
        episode_oscillations = 0
        done = False
        step = 0
        
        # Store previous position for movement success tracking
        prev_pos = None
        
        while not done and step < 10000:
            # Store previous position
            if prev_pos is None:
                prev_pos = (env.player.getX(), env.player.getY())
                
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Get current position
            curr_pos = (env.player.getX(), env.player.getY())
            
            # Track right movement success
            if action == 1:  # RIGHT
                episode_right_attempts += 1
                if curr_pos[0] > prev_pos[0]:  # Successful rightward movement
                    episode_right_successes += 1
            
            # Check for oscillation
            if hasattr(env, '_detect_oscillation') and env._detect_oscillation(env.action_history, env.position_history):
                episode_oscillations += 1
                
            # Update previous position
            prev_pos = curr_pos
            
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
        
        # Update statistics
        rewards.append(total_reward)
        lengths.append(step)
        scores.append(info.get('score', 0))
        collisions.append(episode_collisions)
        right_attempts += episode_right_attempts
        right_successes += episode_right_successes
        oscillation_incidents += episode_oscillations
        
        # Update action counts
        for action, count in episode_actions.items():
            actions_taken[action] += count
        
        # Track vertical movements
        vertical_movements += episode_actions[0] + episode_actions[2]  # UP + DOWN
        
        if verbose and (ep % 5 == 0 or ep == num_episodes - 1):
            print(f"Episode {ep+1}/{num_episodes}: Reward={total_reward:.2f}, "
                  f"Length={step}, Score={info.get('score', 0)}")
    
    # Restore original render mode
    env.render_mode = render_mode
    
    # Calculate overall statistics
    total_actions = sum(actions_taken.values())
    action_distribution = {action: count / total_actions for action, count in actions_taken.items()}
    
    # Calculate forward movement success rate
    forward_success_rate = right_successes / max(1, right_attempts)
    
    # Calculate collision rate
    total_collisions = sum(collisions)
    collision_rate = total_collisions / total_actions if total_actions > 0 else 0
    
    # Path efficiency (rightward movement / total movement)
    rightward_actions = actions_taken[1]  # Action 1 is RIGHT
    path_efficiency = rightward_actions / total_actions if total_actions > 0 else 0
    
    # Calculate vertical movement utilization
    vertical_movement_rate = vertical_movements / total_actions if total_actions > 0 else 0
    
    # Oscillation rate
    oscillation_rate = oscillation_incidents / num_episodes
    
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
        'vertical_movement_rate': vertical_movement_rate,
        'oscillation_rate': oscillation_rate,
        'action_distribution': action_distribution,
        'raw_data': {
            'rewards': rewards,
            'lengths': lengths,
            'scores': scores,
            'collisions': collisions,
            'actions_taken': actions_taken,
            'right_attempts': right_attempts,
            'right_successes': right_successes,
            'oscillation_incidents': oscillation_incidents
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
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
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
    
    # Success criteria metrics (specific to Phase 1)
    metric_names = [
        'Forward Success',
        'Collision Rate', 
        'Path Efficiency', 
        'Vert. Movement',
        'Oscillation Rate'
    ]
    
    metric_values = [
        eval_results['forward_success_rate'],
        eval_results['collision_rate'],
        eval_results['path_efficiency'],
        eval_results['vertical_movement_rate'],
        eval_results['oscillation_rate']
    ]
    
    # Plot success criteria metrics
    axes[2, 0].bar(metric_names, metric_values)
    axes[2, 0].set_title('Success Criteria Metrics')
    axes[2, 0].set_ylabel('Value')
    axes[2, 0].set_ylim(0, 1.0)  # All metrics are rates between 0 and 1
    
    # Phase 1 success thresholds
    thresholds = {
        'Forward Success': 0.85,  # >85% successful rightward attempts
        'Collision Rate': 0.08,   # <8% of actions result in wall collisions
        'Path Efficiency': 0.5,   # No specific threshold but higher is better
        'Vert. Movement': 0.2,    # 15-25% of actions are vertical movement
        'Oscillation Rate': 0.1   # Lower is better
    }
    
    # Add threshold markers
    for i, name in enumerate(metric_names):
        if name in thresholds:
            threshold = thresholds[name]
            if name == 'Collision Rate' or name == 'Oscillation Rate':
                # For these metrics, below threshold is good
                color = 'green' if metric_values[i] < threshold else 'red'
            else:
                # For other metrics, above threshold is good
                color = 'green' if metric_values[i] >= threshold else 'red'
                
            axes[2, 0].axhline(threshold, xmin=i/len(metric_names), xmax=(i+1)/len(metric_names), 
                              color=color, linestyle='--')
    
    # Collisions per episode
    collisions = eval_results['raw_data']['collisions']
    axes[2, 1].hist(collisions, bins=range(0, max(collisions) + 2), alpha=0.7)
    axes[2, 1].set_title('Collisions per Episode')
    axes[2, 1].set_xlabel('Number of Collisions')
    axes[2, 1].set_ylabel('Frequency')
    
    # Add phase 1 criteria assessment
    success_criteria = [
        f"Forward Success: {eval_results['forward_success_rate']:.2%} (Target: >85%)",
        f"Collision Rate: {eval_results['collision_rate']:.2%} (Target: <8%)",
        f"Avg Score: {eval_results['avg_score']:.1f} (Target: ≥180)",
        f"Vertical Movement: {eval_results['vertical_movement_rate']:.2%} (Target: 15-25%)"
    ]
    
    fig.text(0.5, 0.01, 
             "Phase 1 Success Criteria:\n" + " | ".join(success_criteria),
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
    eval_results = evaluate_agent(agent, eval_env, args.episodes, args.render)
    
    print("\nPhase 1 Evaluation Results:")
    print(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Average Score: {eval_results['avg_score']:.2f} ± {eval_results['std_score']:.2f}")
    print(f"Average Length: {eval_results['avg_length']:.2f} ± {eval_results['std_length']:.2f}")
    
    # Print Phase 1 success criteria metrics with pass/fail indicators
    print("\nPhase 1 Success Criteria:")
    
    forward_success = eval_results['forward_success_rate']
    print(f"Forward Movement Success: {forward_success:.2%} {'✅' if forward_success >= 0.85 else '❌'} (Target: >85%)")
    
    collision_rate = eval_results['collision_rate']
    print(f"Collision Rate: {collision_rate:.2%} {'✅' if collision_rate < 0.08 else '❌'} (Target: <8%)")
    
    avg_score = eval_results['avg_score']
    print(f"Average Score: {avg_score:.1f} {'✅' if avg_score >= 180 else '❌'} (Target: ≥180)")
    
    vert_rate = eval_results['vertical_movement_rate']
    print(f"Vertical Movement: {vert_rate:.2%} {'✅' if 0.15 <= vert_rate <= 0.25 else '❌'} (Target: 15-25%)")
    
    print(f"\nOscillation Rate: {eval_results['oscillation_rate']:.2%} (Lower is better)")
    print(f"Path Efficiency: {eval_results['path_efficiency']:.4f} (Higher is better)")
    
    # Print action distribution
    print("\nAction Distribution:")
    action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'NO_ACTION']
    for i, name in enumerate(action_names):
        print(f"{name}: {eval_results['action_distribution'][i]:.2%}")
        
    # Overall assessment
    criteria_met = (
        forward_success >= 0.85 and
        collision_rate < 0.08 and
        avg_score >= 180 and
        0.15 <= vert_rate <= 0.25
    )
    
    print("\n" + "=" * 50)
    if criteria_met:
        print("🎉 SUCCESS: Phase 1 completion criteria met! 🎉")
    else:
        print("❌ INCOMPLETE: Phase 1 completion criteria not fully met.")
    print("=" * 50)
    
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
