# Infinite Maze AI

This module implements reinforcement learning agents for the Infinite Maze game, following the training plan detailed in `docs/ai/`.

## Phase 1 Implementation

The implementation focuses on Phase 1 of the updated training plan, which teaches the agent basic navigation skills in a static maze environment without the advancing pace line. This serves as the foundation for more advanced skills in future phases.

The Phase 1 implementation has been updated to match the revised training plan, with the following improvements:

### Environment Updates
- Enhanced reward function with path-finding incentives and strategic vertical movement rewards
- Advanced oscillation detection and mitigation with multiple detection methods
- More varied starting positions (8 different configurations) for better generalization 
- 11×11 grid representation with 4 channels (walls, player position, pace line proximity, visited cells)
- Context-aware collision penalties that adapt during early training

### Model Updates
- Rainbow DQN implementation with:
  - Double Q-Learning to reduce overestimation bias
  - Prioritized Experience Replay for more efficient learning
  - Smart exploration strategy for better maze navigation
  - Multi-step learning (2-step returns)
  - Extended exploration (epsilon) decay over 150K steps

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
python -m infinite_maze.ai.training.train_phase1 --steps 600000 --checkpoint-dir checkpoints
```

The training script:
1. Creates a timestamp-based run directory for experiment tracking
2. Trains the agent with updated parameters from the training plan:
   - Enhanced exploration strategy (epsilon decay over 150K steps)
   - Prioritized experience replay for more efficient learning
   - Multi-step (2-step) returns for better credit assignment
   - Smart checkpoint saving based on evaluation performance
3. Evaluates performance against success criteria at regular intervals
4. Automatically stops if all success criteria are met before maximum steps

Options:
- `--steps`: Number of training steps (default: 600,000 as per training plan)
- `--checkpoint-dir`: Directory to save checkpoints
- `--device`: Device to train on (cuda or cpu)
- `--render`: Enable rendering during training
- `--eval-interval`: Steps between evaluations (default: 10,000)
- `--save-interval`: Steps between saving checkpoints (default: 50,000)
- `--log-interval`: Episodes between logging (default: 100)

### Evaluation

To evaluate a trained agent:

```
python -m infinite_maze.ai.evaluation.evaluate_phase1 --checkpoint-dir ai/checkpoints/phase1/run_TIMESTAMP/best_model --episodes 20
```

The evaluation script:
1. Runs the agent through multiple episodes in the Phase 1 environment
2. Measures key metrics including:
   - Forward movement success rate (target: >85%)
   - Collision rate (target: <8%)
   - Average score (target: ≥180 points per episode)
   - Vertical movement utilization (target: 15-25% of actions)
   - Oscillation rate and path efficiency
3. Generates comprehensive visualizations of agent performance
4. Provides clear pass/fail status for each success criterion

Options:
- `--checkpoint-dir`: Directory with trained model
- `--episodes`: Number of evaluation episodes (default: 20)
- `--render`: Render the environment during evaluation
- `--save-dir`: Directory to save evaluation results and visualizations
- `--device`: Device to run evaluation on (default: auto-select)

### Play Mode

To watch the trained agent play:

```
python -m infinite_maze.ai.main play --checkpoint-dir ai/checkpoints/phase1/run_TIMESTAMP/best_model --episodes 5
```

## Model Architecture

The implemented model architecture follows the training plan specification:

- **Input**:
  - Grid representation (11×11×4) with separate channels for:
    - Channel 1: Walls (1 for wall, 0 for open)
    - Channel 2: Player position (1 at player position, 0 elsewhere)
    - Channel 3: Pace line proximity (values decreasing with distance)
    - Channel 4: Visited cells (1 for visited, 0 for unvisited)
  - Numerical features (11 values):
    - Distance to pace line (normalized)
    - Current pace level
    - Time until next pace increase
    - Available directions (binary features for each direction)
    - Previous action encoding (one-hot)

- **CNN Backbone**:
  - Conv2D: 32 filters, 3×3 kernel, ReLU
  - Conv2D: 64 filters, 3×3 kernel, ReLU
  - Conv2D: 64 filters, 3×3 kernel, ReLU

- **Dense Processing**:
  - 256 neurons (CNN output), ReLU
  - 64 neurons (numerical features), ReLU
  - 128 neurons (combined features), ReLU
  - 128 neurons, ReLU

- **Output**: Q-values for 5 actions (UP, RIGHT, DOWN, LEFT, NO_ACTION)

## Training Algorithm

The implementation uses Rainbow DQN, which combines several improvements to the base DQN algorithm:

- **Double Q-Learning**: Reduces overestimation bias by using separate networks for action selection and evaluation
- **Prioritized Experience Replay**: Focuses learning on important transitions by sampling based on TD error magnitude
- **Multi-step Learning**: Uses 2-step returns for faster reward propagation and better credit assignment
- **Smart Exploration Strategy**: Enhanced epsilon-greedy exploration with 150K step decay and 50K step warmup period
- **Dueling Network Architecture**: Prepared for activation in Phase 2 to separately estimate state values and action advantages

## Phase 1 Training Specifics

- **Environment**: Static maze with no advancing pace line, varied starting positions
- **Reward Function**: Enhanced rewards for beneficial movements, path discovery, and oscillation prevention
- **Training Duration**: 600,000 steps per updated training plan
- **Success Criteria**:
  - Forward movement success rate: >85% successful rightward attempts
  - Collision rate: <8% of actions result in wall collisions
  - Average score: ≥180 points per episode
  - Vertical movement utilization: 15-25% of actions

## Future Work

This initial implementation provides the foundation for Phase 1 of the training plan. Future work will include:

1. Implementation of Phases 2-5
2. Advanced model architectures including recurrent components
3. Full implementation of all Rainbow DQN features
4. Integration with the main game for AI-assisted play

## Implementation Notes

### Key Features

- **Comprehensive Oscillation Detection**: Multiple detection algorithms that analyze both action patterns and position history to identify and discourage oscillatory behavior
- **Path Analysis**: Advanced functions that evaluate path quality, discover better routes, and reward strategic vertical movements
- **Adaptive Rewards**: Context-aware reward function that adjusts based on training progress and agent behavior
- **Visualization Tools**: Detailed plots and metrics to analyze agent performance and verify success criteria
- **Progress Tracking**: Time-based progress reporting with ETA estimates during long training sessions

### Architecture

The code architecture is designed with extensibility in mind, making it straightforward to add support for future training phases while reusing the core components. Each phase builds upon the previous one through transfer learning, with clear criteria for phase transitions.

### Success Criteria Verification

The evaluation system tracks all required metrics from the training plan and provides clear pass/fail indicators for each criterion. This ensures that the agent must fully satisfy all requirements before progressing to the next phase.
