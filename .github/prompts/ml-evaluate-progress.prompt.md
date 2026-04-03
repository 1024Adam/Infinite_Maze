---
description: "Benchmark the latest saved model, summarise TensorBoard metrics, diagnose training problems, and recommend next actions."
agent: ml-trainer
---
Evaluate the current state of training and produce a diagnostic report. Work through each step in order — do not skip a step if artifacts are missing, instead report what is absent.

## Step 1 — Artifact Inventory

Check what training artifacts exist:

1. **Saved models** — look for `.zip` files under `infinite_maze/ml/models/` or any path referenced in `train.py`
2. **TensorBoard logs** — look for a `runs/` or `logs/` directory
3. **Config snapshot** — read `ML_CONFIG` from `utils/config.py` and list every key/value

Report: model filenames with timestamps, log directory path, and a copy of `ML_CONFIG`.

## Step 2 — TensorBoard Metrics Summary

If a log directory exists, extract the following metrics from the most recent training run (use `tensorboard --inspect` or read event files directly):

| Metric | Value | Trend (↑ ↓ →) |
|---|---|---|
| Mean episode reward (last 10 episodes) | | |
| Mean episode length (last 10 episodes) | | |
| Max pace reached | | |
| Policy loss | | |
| Value loss (PPO) / TD loss (DQN) | | |
| Entropy (if PPO) | | |

If no logs exist, note this and skip to Step 4.

## Step 3 — Benchmark the Latest Model

Load the most recently saved model and run **20 evaluation episodes** using `InfiniteMazeEnv` in headless mode. For each episode capture: score, episode length, max pace reached, and terminal cause.

Report:
- Mean ± std of score, length, pace
- Min and max score observed
- What fraction of episodes ended due to pace catch-up vs. wall collision (if distinguishable)
- Compare mean score against these tiers: Beginner (50), Intermediate (150), Advanced (300), Elite (500)

## Step 4 — Diagnose Training Health

Analyse the metrics and benchmark results to identify problems. Check each item:

| Symptom | How to detect | Likely cause |
|---|---|---|
| Score not improving after phase 1 | Flat reward curve | Reward too sparse; try denser shaping |
| Agent always moves right, ignores walls | Collisions dominate episode ends | Needs better wall-distance features in obs |
| Agent idles or moves randomly | Near-zero mean reward | Learning rate too high or obs not informative |
| Episode length collapsing | Short episodes getting shorter | Pace reward not balanced; agent not surviving |
| High policy loss but low episode reward | Optimising wrong objective | Check reward sign conventions |
| Score improves then collapses | Catastrophic forgetting | Consider replay buffer tuning (DQN) or smaller clip range (PPO) |

Run `diagnose.py` with sequence analysis enabled and include the output in your assessment:

```bash
python -m infinite_maze.ml.diagnose \
	--model <model_path> --phase <phase> --steps 5000 --seed 42 \
	--top-k 10 --trace-episodes 2 --trace-steps 120
```

In addition to action distribution, explicitly evaluate:
- Top 2-action and 3-action motifs (are they diverse or dominated by one loop?)
- Max consecutive run length by action (look for long DOWN-only or RIGHT-only streaks)
- Transition concentration (e.g. mostly `DOWN -> DOWN`)
- Early-step traces for repeated habits and local looping patterns

Report which symptoms are present and their likely causes.

## Step 5 — Recommendations

Based on Steps 2–4, produce a prioritised action list:

1. **Immediate fix** (blocking issue, do this now)
2. **Hyperparameter adjustment** (specific param → specific new value with justification)
3. **Observation / reward change** (what to add or reweight)
4. **Next training phase** (advance to next phase if current success criterion is met, or repeat with adjustments)

For each recommendation, reference the exact file to change (`config.py`, `rewards.py`, `features.py`, `train.py`) and the specific constant or function to modify.

## Step 6 — Summary

Produce a one-paragraph assessment: where is the agent right now relative to the training plan phases, what is the most important thing to fix, and what does success look like at the next checkpoint.
