#!/usr/bin/env python3
"""
Train agent with anti-oscillation rewards to prevent left-right cycling behavior.

This training script uses enhanced rewards that specifically address:
- Oscillating left-right movement
- Getting stuck in local optima
- Poor long-term navigation strategy
"""

import argparse
import os
from train_agent import train_dqn_agent

def train_anti_oscillation_agent(timesteps: int = 500000, model_name: str = "anti_oscillation_v1"):
    """Train agent with anti-oscillation reward structure."""
    
    print("Anti-Oscillation Training for Enhanced Navigation")
    print("=" * 60)
    print(f"Training timesteps: {timesteps}")
    print(f"Model name: {model_name}")
    print("\nKey improvements in this version:")
    print("- Stronger penalties for left movement (-0.8 vs -0.5)")
    print("- Oscillation detection and penalties")
    print("- Reduced base reward for right movement (focus on progress)")
    print("- Enhanced progress tracking with position history")
    print("- Faster stuck detection (8 vs 10 failed moves)")
    print("- Rewards for consistent forward progress")
    print("- Penalty for DO_NOTHING to encourage action")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    print(f"\nStarting training for {timesteps} timesteps...")
    print("Monitor TensorBoard to see if oscillation is reduced!")
    
    # Train with enhanced anti-oscillation rewards
    model = train_dqn_agent(total_timesteps=timesteps)
    
    # Save with descriptive name
    model_path = f"models/{model_name}"
    model.save(model_path)
    print(f"\nModel saved as: {model_path}.zip")
    
    print("\nTraining completed!")
    print("Expected improvements:")
    print("- Less left-right oscillation")
    print("- Better obstacle navigation")
    print("- More consistent forward progress")
    print("- Reduced getting stuck behavior")
    
    print(f"\nTo test: python test_agent.py --model {model_path}.zip")
    
    return model

def compare_with_old_model(new_model: str = "models/anti_oscillation_v1.zip", 
                          old_model: str = "models/best_model.zip"):
    """Compare new anti-oscillation model with previous version."""
    
    if not os.path.exists(old_model):
        print(f"Old model {old_model} not found, skipping comparison")
        return
    
    print("Model Comparison Script")
    print("=" * 30)
    print(f"Old model: {old_model}")
    print(f"New model: {new_model}")
    print("\nTo compare both models:")
    print(f"python test_agent.py --compare {old_model} {new_model} --episodes 10")
    print("\nTo test with visualization:")
    print(f"python test_agent.py --model {old_model}")
    print(f"python test_agent.py --model {new_model}")

def main():
    parser = argparse.ArgumentParser(description='Train anti-oscillation navigation agent')
    parser.add_argument('--timesteps', '-t', type=int, default=500000,
                      help='Training timesteps (default: 500000)')
    parser.add_argument('--name', '-n', type=str, default='anti_oscillation_v1',
                      help='Model name (default: anti_oscillation_v1)')
    parser.add_argument('--compare', '-c', action='store_true',
                      help='Show comparison commands')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_old_model()
        return
    
    train_anti_oscillation_agent(args.timesteps, args.name)

if __name__ == "__main__":
    main()
