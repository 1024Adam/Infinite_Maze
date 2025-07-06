"""
Main entry point for the Infinite Maze AI module.

This script provides a command-line interface to train and evaluate
reinforcement learning agents for the Infinite Maze game.
"""

import argparse
import os
import sys
from infinite_maze.ai.train_phase1 import train_phase_1
from infinite_maze.ai.train_phase2 import train_phase_2
from infinite_maze.ai.evaluate_phase1 import evaluate_agent, load_trained_agent, visualize_evaluation
from infinite_maze.ai.evaluate_phase2 import run_comprehensive_evaluation
from infinite_maze.ai.phase1_env import InfiniteMazeEnv
from infinite_maze.ai.phase2_env import Phase2MazeEnv

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
    train_parser.add_argument('--steps', type=int, default=500000, 
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
            train_phase_1(
                steps=args.steps,
                checkpoint_dir=os.path.join(args.checkpoint_dir, f'phase{args.phase}'),
                device=args.device,
                render_mode='human' if args.render else None
            )
        elif args.phase == 2:
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
        
        # Load the agent
        agent = load_trained_agent(args.checkpoint_dir, 'cpu')
        
        if phase == 1:
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
            # For Phase 2, use the specialized evaluation function
            print("\nRunning comprehensive Phase 2 evaluation...")
            save_dir = args.save_dir if args.save_dir else 'evaluation_results'
            
            # Run the comprehensive evaluation
            eval_results = run_comprehensive_evaluation(
                model_path=args.checkpoint_dir,
                save_dir=save_dir,
                device='cpu',
                render_mode='human' if args.render else None
            )
            
            # Results are automatically saved and displayed by the comprehensive evaluation function
            print(f"\nDetailed evaluation results saved to: {save_dir}")
            
            # Optional: Extract and print key metrics for quick reference
            if eval_results:
                criteria_check = eval_results.get('criteria_check', {})
                if criteria_check:
                    print("\nSuccess Criteria Check:")
                    for name, criterion in criteria_check.get('criteria', {}).items():
                        status = "✅ PASSED" if criterion.get('passed', False) else "❌ FAILED"
                        print(f"{name}: {criterion.get('value', 0):.2f} vs {criterion.get('target', 'N/A')} - {status}")
                    
                    overall = "✅ PASSED" if criteria_check.get('main_criteria_passed', False) else "❌ FAILED"
                    print(f"\nOverall Phase 2 Criteria: {overall}")
    
    elif args.command == 'play':
        # Determine which phase we're playing based on the checkpoint path
        phase = 1  # Default to Phase 1
        if 'phase2' in args.checkpoint_dir.lower():
            phase = 2
        
        # Load the agent
        agent = load_trained_agent(args.checkpoint_dir, 'cpu')
        
        # Create environment for playing based on the phase
        if phase == 1:
            play_env = InfiniteMazeEnv(
                training_phase=1,
                use_maze_from_start=True,
                pace_enabled=False,
                render_mode='human'
            )
            print(f"Watching Phase 1 agent play for {args.episodes} episodes...")
        else:  # phase == 2
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
