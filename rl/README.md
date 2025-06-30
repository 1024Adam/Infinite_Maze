# Reinforcement Learning for Infinite Maze

This directory contains a complete RL implementation for training agents to play the Infinite Maze game.

## Overview

The RL implementation includes:
- **Gymnasium Environment**: A custom environment wrapper that conforms to OpenAI Gym standards
- **DQN Agent**: Deep Q-Network implementation using Stable-Baselines3
- **Training Pipeline**: Complete training infrastructure with logging and evaluation
- **Testing Framework**: Tools for evaluating and comparing trained agents

## Files

### Core Components
- `environment.py`: Custom Gymnasium environment for the Infinite Maze game
- `train_agent.py`: Training script for DQN agents with support for continuing training
- `test_agent.py`: Testing and evaluation script for trained agents
- `continue_training.py`: Convenient script for continuing training from saved models

### Directory Structure
```
rl/
├── __init__.py          # Package initialization
├── environment.py       # RL environment wrapper
├── train_agent.py      # Training script with continue support
├── test_agent.py       # Testing script
├── continue_training.py # Continue training script
├── README.md           # This file
├── models/             # Trained model storage (created during training)
├── logs/               # Training logs (created during training)
└── tensorboard_logs/   # TensorBoard logs (created during training)
```

## Environment Details

### State Space
The environment provides a 109-dimensional observation vector containing:
- Player position (x, y) - normalized
- Game pace - normalized
- Current score - normalized  
- Time survived - normalized
- Wall distances in 4 directions - normalized
- Local maze structure (10x10 grid around player)

### Action Space
5 discrete actions:
- 0: DO_NOTHING
- 1: RIGHT
- 2: LEFT
- 3: UP
- 4: DOWN

### Reward Function
- **+1.0**: Moving right (score increase)
- **-1.0**: Moving left (score decrease) 
- **+0.1**: Surviving each step
- **+0.05**: Staying safely ahead of the pace
- **-0.1**: Being dangerously close to the pace
- **-100**: Collision/death

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the main infinite_maze package is available in your Python path.

## Usage

### Training an Agent

To train a new DQN agent from scratch:
```bash
cd rl
python train_agent.py
```

To continue training from a previously saved model:
```bash
cd rl
# Continue from the best model
python train_agent.py --continue rl/models/best_model.zip

# Or use the dedicated continue training script
python continue_training.py --model rl/models/best_model.zip --steps 100000
```

To list available saved models:
```bash
cd rl
python continue_training.py --list-models
```

Training parameters can be modified in the `train_dqn_agent()` function in `train_agent.py`.

### Testing a Trained Agent

To test with visualization:
```bash
cd rl
python train_agent.py --test --model models/best_model.zip
# Or using the dedicated test script
python test_agent.py --model models/best_model.zip
```

To test without visualization (headless):
```bash
cd rl
python test_agent.py --model models/best_model.zip --headless --episodes 10
```

To test the random baseline:
```bash
cd rl
python test_agent.py --baseline --episodes 10
```

To compare multiple models:
```bash
cd rl
python test_agent.py --compare models/model1.zip models/model2.zip --episodes 10
```

### Monitoring Training

Use TensorBoard to monitor training progress:
```bash
tensorboard --logdir rl/tensorboard_logs
```

## Hyperparameters

The DQN agent uses the following hyperparameters (configurable in `train_agent.py`):

- **Learning Rate**: 1e-4
- **Buffer Size**: 100,000
- **Batch Size**: 32
- **Gamma**: 0.99
- **Exploration**: Epsilon-greedy with linear decay from 1.0 to 0.05
- **Target Update**: Every 1,000 steps
- **Training Frequency**: Every 4 steps

## Model Architecture

The DQN uses a Multi-Layer Perceptron (MLP) policy with:
- Input: 109-dimensional observation vector
- Hidden layers: Configurable (default Stable-Baselines3 MLP)
- Output: 5 Q-values for each action

## Training Tips

1. **Start Small**: Begin with shorter training runs to verify the environment works correctly
2. **Continue Training**: Use `--continue` to build upon previous training sessions instead of starting from scratch
3. **Monitor Rewards**: Watch for positive trends in episode rewards and survival times
4. **Best vs Final Models**: Continue from `best_model.zip` (best performance) rather than `final.zip` (end of training)
5. **Hyperparameter Tuning**: Adjust learning rate, exploration parameters, and network architecture based on performance
6. **Curriculum Learning**: Consider gradually increasing difficulty or maze complexity
7. **Multiple Seeds**: Train multiple agents with different random seeds for robust evaluation

## Integration with Original Game

The `InfiniteMazeWrapper` class provides compatibility with the original game's `controlled_run` function, allowing trained agents to play in the original game interface.

## Performance Metrics

Key metrics to track:
- **Survival Time**: How long the agent stays alive
- **Final Score**: Game score achieved before death
- **Episode Reward**: Total RL reward accumulated
- **Learning Curve**: Improvement over training steps

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the parent directory is in your Python path
2. **Pygame Issues**: Make sure pygame is properly installed and display is available (use headless=True for servers)
3. **CUDA Issues**: The code automatically detects and uses GPU if available
4. **Memory Issues**: Reduce buffer size or batch size if running out of memory

### Common Usage Patterns

**Initial Training:**
```bash
python train_agent.py                    # Train new agent
```

**Continued Training:**
```bash
python train_agent.py --continue rl/models/best_model.zip  # Continue from best
python continue_training.py --steps 200000                 # Continue with more steps
```

**Testing:**
```bash
python train_agent.py --test             # Test with visualization
python test_agent.py --headless          # Test without graphics
```

### Performance Issues

1. **Slow Training**: Ensure you're using GPU acceleration if available
2. **Poor Convergence**: Try adjusting learning rate or exploration parameters
3. **Unstable Training**: Reduce learning rate or increase target update frequency

## Future Enhancements

Potential improvements:
- **PPO Implementation**: Try Proximal Policy Optimization for better sample efficiency
- **CNN Architecture**: Use convolutional networks for better spatial understanding
- **Hierarchical RL**: Implement multi-level decision making
- **Curriculum Learning**: Progressive difficulty increase during training
- **Multi-Agent Training**: Train multiple agents simultaneously

## Contributing

When modifying the RL implementation:
1. Update this README if adding new features
2. Add proper documentation to new functions
3. Test changes with both headless and visual modes
4. Update requirements.txt if adding new dependencies
