"""
Main entry point for the Infinite Maze AI module.

This script provides a command-line interface to train and evaluate
reinforcement learning agents for the Infinite Maze game.
"""

import argparse
import os
import sys
import torch

def run_comprehensive_evaluation(model_path, save_dir, device='cpu', render_mode=None):
    """
    Run a comprehensive evaluation of a trained agent.
    
    Args:
        model_path: Path to the trained model
        save_dir: Directory to save evaluation results
        device: Device to load the model on
        render_mode: Rendering mode (None, 'human', 'rgb_array')
        
    Returns:
        Evaluation results
    """
    from infinite_maze.ai.evaluation.evaluate_phase2 import evaluate_agent, load_trained_agent, visualize_evaluation
    from infinite_maze.ai.environments.environment_phase2 import Phase2MazeEnv
    import os
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the agent
    agent = load_trained_agent(model_path, device)
    
    # Create evaluation environments with different configurations
    # 1. Standard evaluation
    standard_env = Phase2MazeEnv(
        pace_enabled=True,
        pace_speed=0.2,
        pace_acceleration=False,
        render_mode=render_mode,
        start_position_difficulty=0.5
    )
    
    # Run standard evaluation
    print("Running standard evaluation...")
    standard_results = evaluate_agent(agent, standard_env, num_episodes=10, render=(render_mode is not None))
    
    # Save and visualize results
    visualize_evaluation(standard_results, save_dir)
    
    # Return the standard evaluation results
    return standard_results

def main():
    """
    Main entry point for the AI module.
    """
    parser = argparse.ArgumentParser(description='Infinite Maze AI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train an agent')
    train_parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3, 4, 5],
                            help='Training phase (1-5)')
    train_parser.add_argument('--steps', type=int, default=600000, 
                            help='Number of training steps')
    train_parser.add_argument('--checkpoint-dir', type=str, default='ai/checkpoints',
                            help='Directory to save checkpoints')
    train_parser.add_argument('--device', type=str, 
                            default='cuda' if torch.cuda.is_available() else 'cpu',
                            help='Device to train on (cuda or cpu)')
    train_parser.add_argument('--render', action='store_true',
                            help='Enable rendering during training')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained agent')
    eval_parser.add_argument('--checkpoint-dir', type=str, required=True,
                           help='Directory with trained model')
    eval_parser.add_argument('--episodes', type=int, default=20,
                           help='Number of evaluation episodes')
    eval_parser.add_argument('--render', action='store_true',
                           help='Render the environment during evaluation')
    eval_parser.add_argument('--save-dir', type=str, default=None,
                           help='Directory to save evaluation results')
    
    # Play command (watch the agent play)
    play_parser = subparsers.add_parser('play', help='Watch the agent play')
    play_parser.add_argument('--checkpoint-dir', type=str, required=True,
                           help='Directory with trained model')
    play_parser.add_argument('--episodes', type=int, default=5,
                           help='Number of episodes to play')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        if args.phase == 1:
            from infinite_maze.ai.training.train_phase1 import train_phase_1
            
            train_phase_1(
                steps=args.steps,
                checkpoint_dir=os.path.join(args.checkpoint_dir, f'phase{args.phase}'),
                device=args.device,
                render_mode='human' if args.render else None
            )
        elif args.phase == 2:
            from infinite_maze.ai.training.train_phase2 import train_phase_2
            
            # For Phase 2, we need to specify the Phase 1 checkpoint to build upon
            phase1_checkpoint = os.path.join(args.checkpoint_dir, 'phase1', 'best_model')
            
            # Check if Phase 1 checkpoint exists
            if not os.path.exists(phase1_checkpoint):
                print(f"ERROR: Phase 1 checkpoint not found at {phase1_checkpoint}")
                print("Please train a Phase 1 model first.")
                sys.exit(1)
                
            train_phase_2(
                steps=args.steps,
                phase1_checkpoint=phase1_checkpoint,
                checkpoint_dir=os.path.join(args.checkpoint_dir, f'phase{args.phase}'),
                device=args.device,
                render_mode='human' if args.render else None
            )
        else:
            print(f"Training for phase {args.phase} is not yet implemented.")
            print("This implementation currently supports Phase 1 and Phase 2 training.")
            print("Future phases will be implemented in subsequent versions.")
    
    elif args.command == 'evaluate':
        # Determine which phase we're evaluating based on the checkpoint path
        phase = 1  # Default to Phase 1
        if 'phase2' in args.checkpoint_dir.lower():
            phase = 2
        
        # Import necessary modules based on phase
        if phase == 1:
            from infinite_maze.ai.evaluation.evaluate_phase1 import (
                evaluate_agent, load_trained_agent, visualize_evaluation
            )
            from infinite_maze.ai.environments.environment_phase1 import InfiniteMazeEnv
            
            # Load the agent
            agent = load_trained_agent(args.checkpoint_dir, 'cpu')
            
            # Create Phase 1 evaluation environment
            eval_env = InfiniteMazeEnv(
                training_phase=1,
                use_maze_from_start=True,
                pace_enabled=False,
                render_mode='human' if args.render else None
            )
            
            # Run standard evaluation
            eval_results = evaluate_agent(
                agent=agent,
                env=eval_env,
                num_episodes=args.episodes,
                render=args.render,
                verbose=True
            )
            
            # Print results
            print("\nEvaluation Results (Phase 1):")
            print(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"Average Score: {eval_results['avg_score']:.2f} ± {eval_results['std_score']:.2f}")
            print(f"Collision Rate: {eval_results['collision_rate']:.4f}")
            print(f"Path Efficiency: {eval_results['path_efficiency']:.4f}")
            
            # Save results if requested
            if args.save_dir:
                visualize_evaluation(eval_results, args.save_dir)
                
        elif phase == 2:
            # Note: This is for future implementation
            try:
                from infinite_maze.ai.evaluation.evaluate_phase2 import evaluate_agent, load_trained_agent
            except ImportError:
                print("Error: Phase 2 evaluation module is not yet implemented.")
                print("Please use Phase 1 evaluation instead.")
                sys.exit(1)
            
            # For Phase 2, use the specialized evaluation function
            print("\nRunning comprehensive Phase 2 evaluation...")
            save_dir = args.save_dir if args.save_dir else 'evaluation_results'
            
            # Load the agent
            agent = load_trained_agent(args.checkpoint_dir, 'cpu')
            
            # Create Phase 2 evaluation environment
            from infinite_maze.ai.environments.environment_phase2 import Phase2MazeEnv
            eval_env = Phase2MazeEnv(
                pace_enabled=True,
                pace_speed=0.2,  # Standard pace speed for evaluation
                pace_acceleration=False,  # No acceleration for evaluation
                render_mode='human' if args.render else None,
                start_position_difficulty=0.5  # Medium difficulty for evaluation
            )
            
            # Run the evaluation
            eval_results = evaluate_agent(
                agent=agent,
                env=eval_env,
                num_episodes=args.episodes,
                render=args.render,
                verbose=True
            )
            
            # Print results
            print("\nEvaluation Results (Phase 2):")
            print(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"Average Score: {eval_results['avg_score']:.2f} ± {eval_results['std_score']:.2f}")
            print(f"Average Survival Time: {eval_results['avg_survival_time']:.2f} seconds")
            print(f"Average Distance to Pace: {eval_results['avg_pace_distance']:.2f}")
            print(f"Oscillation Rate: {eval_results['oscillation_rate']:.2%}")
            print(f"Vertical Movement Rate: {eval_results['vertical_movement_rate']:.2%}")
            
            # Phase 2 Success Criteria assessment
            pace_ok = eval_results['avg_pace_distance'] >= 100
            survival_ok = eval_results['avg_survival_time'] >= 120
            vertical_ok = 0.25 <= eval_results['vertical_movement_rate'] <= 0.45
            oscillation_ok = eval_results['oscillation_rate'] < 0.05
            
            print("\nPhase 2 Success Criteria:")
            print(f"Pace Line Distance: {eval_results['avg_pace_distance']:.1f} {'✅' if pace_ok else '❌'} (Target: >100 pixels)")
            print(f"Survival Time: {eval_results['avg_survival_time']:.1f}s {'✅' if survival_ok else '❌'} (Target: ≥120 seconds)")
            print(f"Vertical Movement: {eval_results['vertical_movement_rate']:.2%} {'✅' if vertical_ok else '❌'} (Target: 25-45%)")
            print(f"Oscillation Rate: {eval_results['oscillation_rate']:.2%} {'✅' if oscillation_ok else '❌'} (Target: <5%)")
            
            # Overall assessment
            criteria_met = pace_ok and survival_ok and vertical_ok and oscillation_ok
            overall = "✅ PASSED" if criteria_met else "❌ FAILED"
            print(f"\nOverall Phase 2 Criteria: {overall}")
            
            # Save results if requested
            if args.save_dir:
                from infinite_maze.ai.evaluation.evaluate_phase2 import visualize_evaluation
                visualize_evaluation(eval_results, save_dir)
    
    elif args.command == 'play':
        # Determine which phase we're playing based on the checkpoint path
        phase = 1  # Default to Phase 1
        if 'phase2' in args.checkpoint_dir.lower():
            phase = 2
        
        if phase == 1:
            from infinite_maze.ai.evaluation.evaluate_phase1 import load_trained_agent
            from infinite_maze.ai.environments.environment_phase1 import InfiniteMazeEnv
            
            # Load the agent
            agent = load_trained_agent(args.checkpoint_dir, 'cpu')
            
            # Create environment for playing
            play_env = InfiniteMazeEnv(
                training_phase=1,
                use_maze_from_start=True,
                pace_enabled=False,
                render_mode='human'
            )
            print(f"Watching Phase 1 agent play for {args.episodes} episodes...")
        else:  # phase == 2
            # Note: This is for future implementation
            try:
                from infinite_maze.ai.evaluation.evaluate_phase2 import load_trained_agent
                from infinite_maze.ai.environments.environment_phase2 import Phase2MazeEnv
            except ImportError:
                print("Error: Phase 2 modules are not yet implemented.")
                print("Please use Phase 1 agent for playing instead.")
                sys.exit(1)
            
            # Load the agent
            agent = load_trained_agent(args.checkpoint_dir, 'cpu')
            
            play_env = Phase2MazeEnv(
                training_phase=2,
                use_maze_from_start=True,
                pace_enabled=True,
                pace_speed=1.0,  # Start with slow pace
                render_mode='human'
            )
            print(f"Watching Phase 2 agent play for {args.episodes} episodes...")
        
        print("Press Ctrl+C to stop")
        
        try:
            for episode in range(args.episodes):
                state = play_env.reset()[0]  # Gymnasium returns (obs, info)
                done = False
                total_reward = 0
                steps = 0
                
                print(f"\nEpisode {episode+1}/{args.episodes}")
                
                while not done:
                    # Select action
                    action = agent.select_action(state, evaluate=True)
                    
                    # Execute action
                    next_state, reward, terminated, truncated, info = play_env.step(action)
                    
                    # Check if episode is done
                    done = terminated or truncated
                    
                    # Update state and metrics
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    # Render
                    play_env.render()
                    
                    # Short delay to make it watchable
                    import time
                    time.sleep(0.05)
                
                print(f"Episode {episode+1} finished: Steps={steps}, Reward={total_reward:.2f}, Score={info.get('score', 0)}")
                
        except KeyboardInterrupt:
            print("\nPlayback stopped by user")
        
        play_env.close()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    # Import torch here to avoid import errors if not installed
    try:
        import torch
    except ImportError:
        print("Error: PyTorch is required for the Infinite Maze AI module.")
        print("Please install it with: pip install torch")
        sys.exit(1)
        
    main()
