# Training Safeguards for Infinite Maze AI

This document outlines specific safeguards to prevent common training pitfalls in the Infinite Maze AI model development, with particular focus on preventing the "always go right" bias problem.

## The "Always Go Right" Bias Problem

In the Infinite Maze game, the model can develop a strong bias to always move right because:
1. The game rewards rightward movement (+1 point)
2. The initial starting area has no walls, allowing unrestricted rightward movement
3. The simplest policy (always go right) works perfectly in the open starting area

This leads to a model that fails catastrophically once it encounters the actual maze structure, as it continues attempting to move right even when walls block its path.

> **Important Note**: All modifications described in this document apply only to the AI training environment, not to the actual game. The game itself should remain unchanged with its original mechanics including the open starting area.

## Implementation Safeguards

### 1. Modified Training Environment

> These modifications are for the training environment only and do not affect the actual game.

- **Start With Maze Structures**: Begin training with maze walls present from the first step
- **Training-Only Environment**: Create a specialized training version that differs from the actual game
- **No Open Starting Area**: Remove the open starting area during initial training phases
- **Variable Starting Positions**: Initialize the agent at different positions within the maze
- **Wall-Dense Scenarios**: Include training scenarios with high wall density requiring navigation

### 2. Enhanced Reward Function

```python
# Additional helper functions for reward calculation

def path_is_open_ahead(state, steps=3):
    """Check if the path ahead is open for n steps"""
    # Implementation details - uses raycasting to detect walls
    # Returns True if path is clear, False if obstacles detected
    
def path_improves(old_state, new_state):
    """Determine if the move led to a better path opportunity"""
    # Implementation details - compares path options before and after move
    # Returns True if new position offers better rightward pathing options
    
def discovered_better_path(old_state, new_state):
    """Check if the agent discovered a more efficient path"""
    # Implementation details - analyzes maze structure visibility
    # Returns True if new viable paths were discovered
    
def calculate_repeated_actions(action_history):
    """Count consecutive repeated actions"""
    # Implementation details - tracks action repetition
    # Returns count of same action repeated consecutively
```

### 3. Training Progress Monitoring

- **Action Distribution Analysis**: Track percentage of actions in each direction
  - Alert if RIGHT actions exceed 60% of total actions
  - Implement corrective measures if bias detected
  
- **Collision Rate Monitoring**: Track wall collision frequency
  - Expect <5% during early training, decreasing to <1% 
  - Retrain with adjusted rewards if collision rate remains high
  
- **Path Efficiency Metrics**: Calculate ratio of distance gained to actions taken
  - Identify if model is using UP/DOWN movements strategically
  - Flag training if vertical movement is below expected threshold
  
- **Maze Navigation Benchmarks**: Test on specific challenging mazes
  - Dead-end recovery scenarios
  - Narrow passage navigation
  - Complex path planning requirements

### 4. Curriculum Design Safeguards

- **Phase 1: Basic Maze Navigation**
  - Begin with simple mazes that require some vertical navigation
  - No initial open areas allowed
  - Verify learning of basic wall avoidance before proceeding
  
- **Phase 2: Strategic Movement**
  - Introduce more complex maze structures
  - Test on scenarios requiring backtracking
  - Ensure model recognizes dead-ends and reroutes effectively
  
- **Phase 3: Gradual Open Area Introduction**
  - Slowly introduce segments with open areas
  - Verify model doesn't revert to "always right" behavior when encountering open spaces
  - Test transitions between open areas and maze sections
  
- **Phase 4: Full Game Conditions**
  - Train with standard game starting conditions (including the open starting area)
  - Use transfer learning from previous phases to preserve navigation skills
  - Monitor for bias regression during this phase
  - Implement corrective mini-batches if bias detected

### 5. Model Architecture Adjustments

- **Short-term Memory Enhancement**: Strengthen LSTM components to remember wall encounters
- **Spatial Awareness Features**: Add explicit wall configuration inputs
- **Path Planning Auxiliary Task**: Add secondary prediction task for dead-end detection
- **Action Consequence Prediction**: Train model to predict outcome of potential moves

## Testing and Validation

### Bias Detection Tests

1. **Open Area Test**: Place agent in open area and observe behavior
   - Expected: Strategic right movement with periodic scanning for optimal paths
   - Failure: Continuous right movement without environmental awareness

2. **Wall Following Test**: Place agent facing wall on right side
   - Expected: Navigate around wall using up/down movements
   - Failure: Repeatedly attempting to move right into the wall

3. **Maze Entry Test**: Test transition from open area to maze
   - Expected: Smooth adaptation to maze navigation
   - Failure: Performance collapse upon maze entry

4. **Deliberate Dead-End Test**: Force navigation into dead-end
   - Expected: Recognition and efficient backtracking
   - Failure: Continuous attempts to progress through the wall

### Quantitative Metrics

- **Direction Balance Score**: Measure appropriate use of all movement directions
  - Healthy model: 40-50% RIGHT, 5-15% LEFT, 20-30% UP, 20-30% DOWN
  - Biased model: >70% RIGHT, <5% LEFT, <15% UP, <15% DOWN

- **Wall Collision Rate**: Percentage of actions resulting in wall collisions
  - Target: <1% after training
  - Concern threshold: >5% after training
  
- **Path Efficiency**: Distance gained per action taken
  - Target: >0.7 distance units per action
  - Concern threshold: <0.5 distance units per action

## Recovery Strategies

If bias is detected during or after training:

1. **Targeted Scenario Retraining**:
   - Create a dataset of challenging navigation scenarios
   - Train specifically on these scenarios with higher learning rate
   
2. **Reward Function Adjustment**:
   - Further reduce rewards for rightward movement
   - Increase rewards for strategic vertical movement
   
3. **Exploration Boost**:
   - Temporarily increase exploration rate (ε) 
   - Force balanced direction sampling
   
4. **Catastrophic Forgetting Mitigation**:
   - Interleave simple maze navigation tasks with complex scenarios
   - Use experience replay with higher priority for maze navigation samples
   
5. **Architecture Expansion**:
   - Add dedicated path planning module
   - Implement explicit wall memory representation

## Training vs. Game Environment

### Separation of Concerns

It's essential to maintain a clear separation between the training environment and the actual game:

1. **Game Environment (Unchanged)**:
   - Preserves the original game design with open starting area
   - Maintains all original game mechanics and scoring
   - Provides the ultimate target environment where the AI will operate

2. **Training Environment (Modified)**:
   - Custom environment that deliberately differs from the game in early training
   - Progressively adapts to match the real game environment
   - Final training phases use the actual game environment
   - Functions solely as a learning tool, not affecting player experience

### Implementation Approach

To implement these training safeguards without modifying the game:

1. **Create a separate training simulator** that mirrors the game's core mechanics
2. **Add configuration options** for training-specific features (wall density, starting position, etc.)
3. **Implement a curriculum manager** that progressively adjusts the training environment
4. **Establish a clear transition path** from modified training to actual game environment

## Conclusion

These safeguards directly address the "always go right" bias problem by modifying the training environment, reward structure, and monitoring processes - all without changing the actual game. By implementing these measures, the model will develop a sophisticated navigation strategy that balances rightward progress with effective maze navigation, resulting in significantly better performance when deployed in the actual game environment with its open starting area.
