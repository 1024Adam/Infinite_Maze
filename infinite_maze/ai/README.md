# Infinite Maze AI

This module implements reinforcement learning agents for the Infinite Maze game, following the training plan detailed in `docs/ai/`.

## Phase 1 Implementation

The current implementation focuses on Phase 1 of the training plan, which teaches the agent basic navigation skills in a static maze environment without the advancing pace line. This serves as the foundation for more advanced skills in future phases.

## Requirements

The AI module has additional dependencies beyond the main game:

```
torch>=1.7.0
numpy>=1.19.0
matplotlib>=3.3.0
gym>=0.17.0
tqdm>=4.48.0
```

You can install them with:

```
pip install -r infinite_maze/ai/requirements.txt
```

## Usage

### Training

To train a new agent for Phase 1:

```
python -m infinite_maze.ai.main train --phase 1 --steps 500000 --checkpoint-dir checkpoints
```

Options:
- `--phase`: Training phase (currently only 1 is supported)
- `--steps`: Number of training steps
- `--checkpoint-dir`: Directory to save checkpoints
- `--device`: Device to train on (cuda or cpu)
- `--render`: Enable rendering during training

### Evaluation

To evaluate a trained agent:

```
python -m infinite_maze.ai.main evaluate --checkpoint-dir ai/checkpoints/phase1/run_TIMESTAMP/best_model --episodes 20
```

Options:
- `--checkpoint-dir`: Directory with trained model
- `--episodes`: Number of evaluation episodes
- `--render`: Render the environment during evaluation
- `--save-dir`: Directory to save evaluation results

### Play Mode

To watch the trained agent play:

```
python -m infinite_maze.ai.main play --checkpoint-dir ai/checkpoints/phase1/run_TIMESTAMP/best_model --episodes 5
```

## Model Architecture

The implemented model architecture follows the training plan specification:

- **Input**: Grid representation (11×11×4) + numerical features
- **CNN Backbone**:
  - Conv2D: 32 filters, 3×3 kernel
  - Conv2D: 64 filters, 3×3 kernel  
  - Conv2D: 64 filters, 3×3 kernel
- **Dense Processing**:
  - 256 neurons (CNN output)
  - 64 neurons (numerical features)
  - 128 neurons (combined features)
  - 128 neurons
- **Output**: Q-values for 5 actions (UP, RIGHT, DOWN, LEFT, NO_ACTION)

## Training Algorithm

The implementation uses Rainbow DQN, which combines several improvements to the base DQN algorithm:

- Double Q-Learning
- Prioritized Experience Replay
- Dueling Network (prepared for future phases)

## Phase 1 Training Specifics

- **Environment**: Static maze with no advancing pace line
- **Reward Function**: Emphasizes rightward movement while avoiding collisions
- **Success Criteria**:
  - Forward movement success rate: >90%
  - Collision rate: <5%
  - Average score: ≥200 points per episode

## Future Work

This initial implementation provides the foundation for Phase 1 of the training plan. Future work will include:

1. Implementation of Phases 2-5
2. Advanced model architectures including recurrent components
3. Full implementation of all Rainbow DQN features
4. Integration with the main game for AI-assisted play

## Implementation Notes

The code architecture is designed with extensibility in mind, making it straightforward to add support for future training phases while reusing the core components.