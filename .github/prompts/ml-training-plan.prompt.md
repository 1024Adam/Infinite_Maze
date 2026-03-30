---
description: "Audit ML module readiness and produce a phased RL training plan with per-phase success criteria for the Infinite Maze agent."
agent: ml-trainer
---
Produce a concrete, phased training plan for teaching an RL agent to beat Infinite Maze. The plan must be grounded in the actual codebase state — check what exists before recommending what to build.

## Step 1 — Readiness Audit

Inspect the current state of the ML module and config:

1. Does `infinite_maze/ml/` exist? If so, list every file and its purpose.
2. Does `utils/config.py` contain an `ML_CONFIG` dict? List any ML constants already defined (learning rate, reward weights, episode cap, MAX_PACE, etc.).
3. Do `tests/unit/test_ml_environment.py` and `tests/integration/test_ml_training.py` exist and pass?

Summarise what is **ready** and what is **missing** before continuing.

## Step 2 — Environment Health Check

If `InfiniteMazeEnv` exists, verify:
- `reset()` returns `(obs, info)` with obs shape matching `observation_space`
- `step(action)` returns a 5-tuple for all 5 actions (0–4)
- A blocked move does not change `player.getX()` / `player.getY()`
- A terminal step sets `terminated=True` when `player.getX() < game.X_MIN`

If the environment does not yet exist, flag this as **Blocker — must implement `InfiniteMazeEnv` first** and stop.

## Step 3 — Training Plan

Produce the plan as a numbered sequence of phases. For each phase include:

| Field | Content |
|---|---|
| **Goal** | What capability the agent should develop |
| **Algorithm** | Which SB3 algorithm to use and why |
| **Timesteps** | Target timestep budget for this phase |
| **Key hyperparameters** | Values to set (learning rate, gamma, n_steps, etc.) |
| **Reward shaping** | Any adjustments to reward weights for this phase |
| **Success criterion** | Measurable threshold before advancing (e.g. mean episode reward > X over last 10 episodes) |
| **Files to create/modify** | Exact file paths |

Use these reference targets from the player guide as success benchmarks:
- Phase 1 complete: agent consistently scores > 50 (Beginner tier)
- Phase 2 complete: agent consistently scores > 150 (Intermediate tier)
- Phase 3 complete: agent consistently scores > 300 (Advanced tier)
- Stretch goal: agent consistently scores > 500 (Elite tier)

Suggested phase structure (adapt based on readiness audit):
1. **Random baseline** — confirm the env is gym-compliant and log untrained performance
2. **Survival** — learn to avoid instant termination; reward staying alive
3. **Rightward bias** — learn that rightward movement scores and delays termination
4. **Maze navigation** — learn to navigate vertical gaps to find rightward paths
5. **Pace adaptation** — generalise to higher pace levels without collapsing

## Step 4 — Curriculum & Implementation Order

List the exact files to create or modify, in the order they should be tackled:

1. Config constants (`ML_CONFIG` additions)
2. `environment.py` (if not done)
3. `features.py` (observation encoding)
4. `rewards.py` (reward shaping)
5. `train.py` (training script)
6. Tests for each

For each file, write one sentence describing the acceptance condition.

## Step 5 — Output

Produce:
1. A **phase table** (phases × fields from Step 3)
2. An **implementation checklist** (ordered file list with acceptance conditions)
3. A **one-paragraph summary** of the overall strategy and the biggest risk to successful training
