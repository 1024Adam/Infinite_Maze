"""
Script to continue training from a previously saved model.

Usage:
    python continue_training.py --model rl/models/best_model.zip --steps 100000
    python continue_training.py --model rl/models/dqn_infinite_maze_final.zip
"""

import argparse
import os
import sys
from train_agent import train_dqn_agent, create_baseline_comparison

def continue_training_from_model(model_path: str, additional_timesteps: int = 100000):
    """Continue training from a saved model."""
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist!")
        print("Available models:")
        models_dir = "rl/models"
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.zip'):
                    print(f"  {os.path.join(models_dir, file)}")
        return
    
    print("Continue Training from Existing Model")
    print("=" * 40)
    print(f"Loading model: {model_path}")
    print(f"Additional timesteps: {additional_timesteps}")
    
    # Create necessary directories
    os.makedirs("rl/models", exist_ok=True)
    os.makedirs("rl/logs", exist_ok=True)
    os.makedirs("rl/tensorboard_logs", exist_ok=True)
    
    # Continue training
    model = train_dqn_agent(continue_from_model=model_path)
    
    print("\nContinued training completed!")
    print("New models saved in rl/models/")

def main():
    parser = argparse.ArgumentParser(description='Continue training from a saved model')
    parser.add_argument('--model', '-m', type=str, default='rl/models/best_model.zip',
                      help='Path to the saved model (default: rl/models/best_model.zip)')
    parser.add_argument('--steps', '-s', type=int, default=100000,
                      help='Additional timesteps to train (default: 100000)')
    parser.add_argument('--list-models', '-l', action='store_true',
                      help='List available saved models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available saved models:")
        models_dir = "rl/models"
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.zip'):
                    full_path = os.path.join(models_dir, file)
                    size = os.path.getsize(full_path) / (1024 * 1024)  # MB
                    print(f"  {full_path} ({size:.1f} MB)")
        else:
            print("  No models directory found!")
        return
    
    continue_training_from_model(args.model, args.steps)

if __name__ == "__main__":
    main()
