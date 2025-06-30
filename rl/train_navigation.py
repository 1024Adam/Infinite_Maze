#!/usr/bin/env python3
"""
Enhanced training script with improved reward structure for navigation learning.

This script uses the updated reward function that encourages vertical movement
when the agent gets stuck trying to move right.
"""

import argparse
import os
import sys
from train_agent import train_dqn_agent

def train_navigation_agent(total_timesteps: int = 500000, model_name: str = "navigation_v1"):
    """Train an agent with enhanced navigation rewards."""
    
    print("Enhanced Navigation Training")
    print("=" * 50)
    print(f"Training timesteps: {total_timesteps}")
    print(f"Model name: {model_name}")
    print("\nEnhanced reward features:")
    print("- Progress-based rewards (actual movement vs. attempted movement)")
    print("- Penalties for getting stuck against walls")
    print("- Rewards for intelligent vertical navigation")
    print("- Look-ahead rewards for finding alternative paths")
    print("- Tracking of consecutive failed right moves")
    
    # Create necessary directories
    os.makedirs("rl/models", exist_ok=True)
    os.makedirs("rl/logs", exist_ok=True)
    os.makedirs("rl/tensorboard_logs", exist_ok=True)
    
    # Train the agent with enhanced rewards
    print("\nStarting enhanced navigation training...")
    model = train_dqn_agent(total_timesteps=total_timesteps)
    
    # Save with specific name
    model_path = f"rl/models/{model_name}"
    model.save(model_path)
    print(f"\nModel saved as: {model_path}.zip")
    
    print("\nTraining completed!")
    print("The agent should now be better at:")
    print("- Recognizing when rightward movement is blocked")
    print("- Using vertical movement to navigate around obstacles")
    print("- Finding alternative paths when stuck")
    print("- Balancing immediate rewards with long-term navigation")
    
    return model

def compare_models():
    """Compare the old and new training approaches."""
    print("Model Comparison Guide")
    print("=" * 30)
    print("Old model (best_model.zip):")
    print("- Simple rightward movement reward")
    print("- Limited vertical movement incentive")
    print("- Gets stuck against walls")
    
    print("\nNew model (navigation_v1.zip):")
    print("- Progress-based rewards")
    print("- Navigation intelligence rewards")
    print("- Penalty for getting stuck")
    print("- Look-ahead path finding")
    
    print("\nTo test both models:")
    print("python rl/train_agent.py --test --model rl/models/best_model.zip")
    print("python rl/train_agent.py --test --model rl/models/navigation_v1.zip")

def main():
    parser = argparse.ArgumentParser(description='Train agent with enhanced navigation rewards')
    parser.add_argument('--timesteps', '-t', type=int, default=500000,
                      help='Total timesteps to train (default: 500000)')
    parser.add_argument('--name', '-n', type=str, default='navigation_v1',
                      help='Model name (default: navigation_v1)')
    parser.add_argument('--compare', '-c', action='store_true',
                      help='Show comparison information between old and new approaches')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
        return
    
    train_navigation_agent(args.timesteps, args.name)

if __name__ == "__main__":
    main()
