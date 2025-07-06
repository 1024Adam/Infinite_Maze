"""
Utilities for evaluating and visualizing the Phase 2 Infinite Maze AI agent's performance.
"""

import os
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Any, List, Tuple, Optional

from infinite_maze.ai.environments.environment_phase2 import Phase2MazeEnv
from infinite_maze.ai.agent import RainbowDQNAgent
from infinite_maze.utils.config import config

def evaluate_agent(agent: RainbowDQNAgent, 
                  env: Phase2MazeEnv, 
                  num_episodes: int = 20,
                  render: bool = False,
                  verbose: bool = True) -> Dict[str, Any]:
    """
    Comprehensive evaluation of Phase 2 agent performance.
    
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
    pace_distances = []
    actions_taken = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # UP, RIGHT, DOWN, LEFT, NO_ACTION
    oscillation_incidents = 0
    vertical_movements = 0
    episode_infos = []
    
    render_mode = env.render_mode
    if not render:
        env.render_mode = None
    
    for ep in range(num_episodes):
        state = env.reset()[0]  # Gymnasium returns (obs, info)
        total_reward = 0
        episode_collisions = 0
        episode_actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        episode_pace_distances = []
        done = False
        step = 0
        
        while not done and step < 10000:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Track metrics
            total_reward += reward
            episode_actions[action] += 1
            if info.get('collision', False):
                episode_collisions += 1
            if 'distance_to_pace' in info:
                episode_pace_distances.append(info['distance_to_pace'])
            
            state = next_state
            step += 1
            
            if render:
                env.render()
            
            # Check if episode is done
            done = terminated or truncated
        
        # Update statistics
        rewards.append(total_reward)
        lengths.append(step)
        scores.append(info.get('score', 0))
        collisions.append(episode_collisions)
        oscillation_incidents += info.get('oscillation_count', 0)
        
        # Calculate average distance to pace line for this episode
        avg_episode_pace_distance = np.mean(episode_pace_distances) if episode_pace_distances else 0
        pace_distances.append(avg_episode_pace_distance)
        
        # Update action counts
        for action, count in episode_actions.items():
            actions_taken[action] += count
        
        # Track vertical movements
        vertical_movements += episode_actions[0] + episode_actions[2]  # UP + DOWN
        
        # Store episode info for detailed analysis
        episode_infos.append({
            'reward': total_reward,
            'length': step,
            'score': info.get('score', 0),
            'collisions': episode_collisions,
            'avg_pace_distance': avg_episode_pace_distance,
            'oscillations': info.get('oscillation_count', 0),
            'pace_level': info.get('pace_level', 1)
        })
        
        if verbose and (ep % 5 == 0 or ep == num_episodes - 1):
            print(f"Episode {ep+1}/{num_episodes}: Reward={total_reward:.2f}, "
                  f"Length={step}, Score={info.get('score', 0)}, "
                  f"Avg Distance to Pace={avg_episode_pace_distance:.1f}")
    
    # Restore original render mode
    env.render_mode = render_mode
    
    # Calculate overall statistics
    total_actions = sum(actions_taken.values())
    action_distribution = {action: count / total_actions for action, count in actions_taken.items()}
    
    # Calculate collision rate
    total_collisions = sum(collisions)
    collision_rate = total_collisions / total_actions if total_actions > 0 else 0
    
    # Calculate vertical movement utilization
    vertical_movement_rate = vertical_movements / total_actions if total_actions > 0 else 0
    
    # Calculate oscillation rate
    oscillation_rate = oscillation_incidents / num_episodes
    
    # Path efficiency (rightward movement / total movement)
    rightward_actions = actions_taken[1]  # Action 1 is RIGHT
    path_efficiency = rightward_actions / total_actions if total_actions > 0 else 0
    
    # Survival time (seconds)
    avg_survival_time = np.mean(lengths) / 10  # Assuming 10 FPS
    
    # Phase 2 specific: Average distance to pace line
    avg_pace_distance = np.mean(pace_distances)
    
    return {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'avg_survival_time': avg_survival_time,
        'avg_pace_distance': avg_pace_distance,
        'std_pace_distance': np.std(pace_distances),
        'collision_rate': collision_rate,
        'path_efficiency': path_efficiency,
        'vertical_movement_rate': vertical_movement_rate,
        'oscillation_rate': oscillation_rate,
        'action_distribution': action_distribution,
        'raw_data': {
            'rewards': rewards,
            'lengths': lengths,
            'scores': scores,
            'collisions': collisions,
            'pace_distances': pace_distances,
            'actions_taken': actions_taken,
            'oscillation_incidents': oscillation_incidents
        },
        'infos': episode_infos
    }

def visualize_evaluation(eval_results: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Visualize the Phase 2 evaluation results.
    
    Args:
        eval_results: Evaluation results from evaluate_agent
        save_path: Directory to save the plots (if None, will display instead)
    """
    # Create figure with subplots - enhanced for Phase 2 metrics
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    
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
    
    # Plot pace distance distribution - Phase 2 specific
    pace_distances = eval_results['raw_data']['pace_distances']
    axes[1, 0].hist(pace_distances, bins=10, alpha=0.7)
    axes[1, 0].axvline(eval_results['avg_pace_distance'], color='r', linestyle='dashed', 
                      linewidth=1, label=f'Mean: {eval_results["avg_pace_distance"]:.2f}')
    axes[1, 0].set_title('Distance to Pace Distribution')
    axes[1, 0].set_xlabel('Distance')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Plot action distribution
    actions = eval_results['action_distribution']
    action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'NO_ACTION']
    axes[1, 1].bar(action_names, [actions[i] for i in range(5)])
    axes[1, 1].set_title('Action Distribution')
    axes[1, 1].set_ylabel('Frequency')
    
    # Success criteria metrics (specific to Phase 2)
    metric_names = [
        'Pace Distance',
        'Survival Time (s)', 
        'Vertical Movement',
        'Oscillation Rate',
        'Path Efficiency'
    ]
    
    metric_values = [
        eval_results['avg_pace_distance'] / 100,  # Normalize to [0,1] scale (target is 100)
        min(1.0, eval_results['avg_survival_time'] / 120),  # Normalize to [0,1] (target is 120s)
        eval_results['vertical_movement_rate'],
        eval_results['oscillation_rate'],
        eval_results['path_efficiency']
    ]
    
    # Plot success criteria metrics
    axes[2, 0].bar(metric_names, metric_values)
    axes[2, 0].set_title('Success Criteria Metrics (Normalized)')
    axes[2, 0].set_ylabel('Value')
    axes[2, 0].set_ylim(0, 1.2)  # Allow values slightly above 1.0
    
    # Phase 2 success thresholds
    thresholds = {
        'Pace Distance': 1.0,  # >100 pixels (normalized to 1.0)
        'Survival Time (s)': 1.0,   # >120 seconds (normalized to 1.0)
        'Vertical Movement': 0.35,  # 25-45% of actions are vertical
        'Oscillation Rate': 0.05,   # <5% oscillation
        'Path Efficiency': 0.6    # No specific threshold but higher is better
    }
    
    # Add threshold markers
    for i, name in enumerate(metric_names):
        if name in thresholds:
            threshold = thresholds[name]
            if name == 'Oscillation Rate':
                # For oscillation, below threshold is good
                color = 'green' if metric_values[i] < threshold else 'red'
            elif name == 'Vertical Movement':
                # For vertical movement, within range is good (25-45%)
                color = 'green' if 0.25 <= metric_values[i] <= 0.45 else 'red'
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
    
    # Add phase 2 criteria assessment
    success_criteria = [
        f"Pace Distance: {eval_results['avg_pace_distance']:.1f} (Target: >100)",
        f"Survival Time: {eval_results['avg_survival_time']:.1f}s (Target: ≥120s)",
        f"Vertical Movement: {eval_results['vertical_movement_rate']:.2%} (Target: 25-45%)",
        f"Oscillation Rate: {eval_results['oscillation_rate']:.2%} (Target: <5%)"
    ]
    
    fig.text(0.5, 0.01, 
             "Phase 2 Success Criteria:\n" + " | ".join(success_criteria),
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
    config_path = os.path.join(checkpoint_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    else:
        config_data = {}  # Default empty config
    
    # Create a temporary environment to get state shape
    env = Phase2MazeEnv(pace_enabled=True, render_mode=None)
    observation = env.reset()[0]  # Gymnasium returns (obs, info)
    state_shape = {
        'grid': observation['grid'].shape,
        'numerical': observation['numerical'].shape
    }
    
    # Create agent with correct parameters
    # For Phase 2, we use dueling architecture
    agent = RainbowDQNAgent(
        state_shape=state_shape,
        num_actions=env.action_space.n,
        device=device,
        use_dueling=True  # Phase 2 uses dueling architecture
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
                  env: Phase2MazeEnv,
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
    avg_survival_times = [comparison_results[name]['avg_survival_time'] for name in agent_names]
    avg_pace_distances = [comparison_results[name]['avg_pace_distance'] for name in agent_names]
    oscillation_rates = [comparison_results[name]['oscillation_rate'] for name in agent_names]
    vertical_movement_rates = [comparison_results[name]['vertical_movement_rate'] for name in agent_names]
    
    # Create comparison plots - expanded for Phase 2
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    
    # Average reward comparison
    axes[0, 0].bar(agent_names, avg_rewards)
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Average score comparison
    axes[0, 1].bar(agent_names, avg_scores)
    axes[0, 1].set_title('Average Score')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Phase 2: Survival time comparison
    axes[1, 0].bar(agent_names, avg_survival_times)
    axes[1, 0].set_title('Average Survival Time (seconds)')
    axes[1, 0].set_ylabel('Seconds')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Phase 2: Pace distance comparison
    axes[1, 1].bar(agent_names, avg_pace_distances)
    axes[1, 1].set_title('Average Distance to Pace')
    axes[1, 1].set_ylabel('Distance')
    axes[1, 1].axhline(y=100, color='r', linestyle='--', label='Target (>100)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend()
    
    # Phase 2: Oscillation rate comparison
    axes[2, 0].bar(agent_names, oscillation_rates)
    axes[2, 0].set_title('Oscillation Rate')
    axes[2, 0].set_ylabel('Rate')
    axes[2, 0].axhline(y=0.05, color='r', linestyle='--', label='Target (<5%)')
    axes[2, 0].tick_params(axis='x', rotation=45)
    axes[2, 0].legend()
    
    # Phase 2: Vertical movement comparison
    axes[2, 1].bar(agent_names, vertical_movement_rates)
    axes[2, 1].set_title('Vertical Movement Rate')
    axes[2, 1].set_ylabel('Rate')
    axes[2, 1].axhline(y=0.25, color='g', linestyle='--', label='Min Target (25%)')
    axes[2, 1].axhline(y=0.45, color='r', linestyle='--', label='Max Target (45%)')
    axes[2, 1].tick_params(axis='x', rotation=45)
    axes[2, 1].legend()
    
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
    parser = argparse.ArgumentParser(description='Evaluate trained Infinite Maze AI Phase 2 agent')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory with trained model')
    parser.add_argument('--episodes', type=int, default=20, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to run evaluation on')
    parser.add_argument('--pace-speed', type=float, default=0.2, help='Pace speed for evaluation (default: 0.2)')
    parser.add_argument('--pace-acceleration', action='store_true', help='Enable pace acceleration')
    
    args = parser.parse_args()
    
    print(f"Loading agent from {args.checkpoint_dir}")
    agent = load_trained_agent(args.checkpoint_dir, args.device)
    
    print(f"Creating Phase 2 evaluation environment")
    eval_env = Phase2MazeEnv(
        pace_enabled=True,
        pace_speed=args.pace_speed,
        pace_acceleration=args.pace_acceleration,
        render_mode='human' if args.render else None,
        start_position_difficulty=0.5  # Medium difficulty for evaluation
    )
    
    print(f"Running evaluation for {args.episodes} episodes")
    eval_results = evaluate_agent(agent, eval_env, args.episodes, args.render)
    
    print("\nPhase 2 Evaluation Results:")
    print(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Average Score: {eval_results['avg_score']:.2f} ± {eval_results['std_score']:.2f}")
    print(f"Average Survival Time: {eval_results['avg_survival_time']:.2f} seconds")
    print(f"Average Distance to Pace: {eval_results['avg_pace_distance']:.2f} ± {eval_results['std_pace_distance']:.2f}")
    
    # Print Phase 2 success criteria metrics with pass/fail indicators
    print("\nPhase 2 Success Criteria:")
    
    avg_pace_distance = eval_results['avg_pace_distance']
    print(f"Pace Line Distance: {avg_pace_distance:.2f} {'✅' if avg_pace_distance >= 100 else '❌'} (Target: >100 pixels)")
    
    avg_survival_time = eval_results['avg_survival_time']
    print(f"Survival Time: {avg_survival_time:.2f}s {'✅' if avg_survival_time >= 120 else '❌'} (Target: ≥120 seconds)")
    
    vert_rate = eval_results['vertical_movement_rate']
    print(f"Vertical Movement: {vert_rate:.2%} {'✅' if 0.25 <= vert_rate <= 0.45 else '❌'} (Target: 25-45%)")
    
    oscillation_rate = eval_results['oscillation_rate']
    print(f"Oscillation Rate: {oscillation_rate:.2%} {'✅' if oscillation_rate < 0.05 else '❌'} (Target: <5%)")
    
    print(f"\nPath Efficiency: {eval_results['path_efficiency']:.4f} (Higher is better)")
    
    # Print action distribution
    print("\nAction Distribution:")
    action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'NO_ACTION']
    for i, name in enumerate(action_names):
        print(f"{name}: {eval_results['action_distribution'][i]:.2%}")
        
    # Overall assessment
    criteria_met = (
        avg_pace_distance >= 100 and
        avg_survival_time >= 120 and
        0.25 <= vert_rate <= 0.45 and
        oscillation_rate < 0.05
    )
    
    print("\n" + "=" * 50)
    if criteria_met:
        print("🎉 SUCCESS: Phase 2 completion criteria met! 🎉")
    else:
        print("❌ INCOMPLETE: Phase 2 completion criteria not fully met.")
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
