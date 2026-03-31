---
name: ml-workflow
description: "Use when: running, resuming, or managing the Infinite Maze RL training pipeline — launching a training run, resuming from a checkpoint, diagnosing behaviour (action distribution, stuck-right analysis), watching a model play live, evaluating model scores, or deciding which checkpoint to use next. Covers train.py, diagnose.py, watch.py, and evaluate.py."
argument-hint: "e.g. 'train phase 1 100k steps', 'diagnose latest checkpoint', 'watch best model'"
---

# Infinite Maze ML Workflow

Covers the four ML scripts in `infinite_maze/ml/`:

| Script | Module | Purpose |
|--------|--------|---------|
| `train.py` | `infinite_maze.ml.train` | Train or resume a PPO agent |
| `diagnose.py` | `infinite_maze.ml.diagnose` | Analyse action distribution & blocked-right behaviour |
| `evaluate.py` | `infinite_maze.ml.evaluate` | Headless episode scoring |
| `watch.py` | `infinite_maze.ml.watch` | Live pygame window playback |

---

## Checkpoints

All runs save under `checkpoints/phase{N}_{YYYYMMDD_HHMMSS}/`:
- `ppo_maze_final_{T}_steps.zip` — final model
- `best/best_model.zip` — best eval checkpoint (use this for watch/diagnose)
- `eval_logs/` — numpy evaluation data

To find the latest run:
```
ls -lt checkpoints/ | head
```

---

## 1. Training

### Fresh run
```bash
python -m infinite_maze.ml.train --phase 1
```
Default: 200,000 timesteps, `n_envs=1`.

### Common overrides
```bash
# Custom timestep count
python -m infinite_maze.ml.train --phase 1 --timesteps 100000

# Faster training with parallel envs (Phase 3+)
python -m infinite_maze.ml.train --phase 1 --timesteps 200000 --n-envs 4

# Enable TensorBoard logging
python -m infinite_maze.ml.train --phase 1 --tensorboard

# Resume from checkpoint (reuses existing weights, extends training)
python -m infinite_maze.ml.train --phase 1 --timesteps 300000 \
    --resume checkpoints/phase1_YYYYMMDD_HHMMSS/ppo_maze_final_200000_steps.zip
```

### Phase defaults (from training plan)

| Phase | Timesteps | `n-envs` | Notes |
|-------|-----------|----------|-------|
| 0 | 1,000 | 1 | Smoke test only |
| 1 | 100k–200k | 1 | Core nav + pace awareness |
| 2 | 200k | 1 | Survival under pace pressure |
| 3 | 500k | 4 | Speed with parallel envs |
| 4+ | 750k+ | 4 | Elite tier |

### Checkpoint / eval frequency (from `ML_CONFIG`)
- Checkpoint saved every **10,000 steps**
- Eval runs every **20,000 steps** → saved to `best/` if new best score

---

## 2. Diagnosing

Reveals whether a model has a genuine navigation policy or is stuck in a degenerate loop (e.g. spam RIGHT).

```bash
# Quick diagnosis (500 steps)
python -m infinite_maze.ml.diagnose \
    --model checkpoints/phase1_YYYYMMDD_HHMMSS/best/best_model.zip

# Reliable estimate (5000 steps, reproducible)
python -m infinite_maze.ml.diagnose \
    --model checkpoints/phase1_YYYYMMDD_HHMMSS/best/best_model.zip \
    --steps 5000 --seed 42
```

### Interpreting output

| Signal | Healthy | Degenerate |
|--------|---------|-----------|
| RIGHT % | 40–70% | >90% |
| UP/DOWN % | 15–35% | <5% |
| Blocked-right → vertical % | >50% | <10% |
| Avg score | >50 | <10 |

A healthy Phase 1 model should take UP/DOWN actions when blocked right at least half the time.

---

## 3. Evaluating

Headless scoring — no window opened. Use to benchmark checkpoints quickly.

```bash
# 10 episodes, report per-episode score
python -m infinite_maze.ml.evaluate \
    --model checkpoints/phase1_YYYYMMDD_HHMMSS/best/best_model.zip \
    --episodes 10

# Reproducible benchmark
python -m infinite_maze.ml.evaluate \
    --model checkpoints/phase1_YYYYMMDD_HHMMSS/best/best_model.zip \
    --episodes 20 --seed 42
```

### Phase score targets (from training plan)

| Phase | Target avg score |
|-------|-----------------|
| 1 | ≥ 30 |
| 2 | ≥ 80 |
| 3 | ≥ 150 |
| 4+ | ≥ 300 |

---

## 4. Watching

Opens a live pygame window. Requires a display — do **not** run in headless CI.

```bash
# Watch 3 episodes at normal speed
python -m infinite_maze.ml.watch \
    --model checkpoints/phase1_YYYYMMDD_HHMMSS/best/best_model.zip \
    --episodes 3

# Slow playback (50 ms ≈ 20 fps)
python -m infinite_maze.ml.watch \
    --model checkpoints/phase1_YYYYMMDD_HHMMSS/best/best_model.zip \
    --episodes 5 --delay 50

# Press ESC or close window to stop early.
```

---

## Typical Workflow

```
1. Train:    python -m infinite_maze.ml.train --phase 1 --timesteps 100000
2. Evaluate: python -m infinite_maze.ml.evaluate --model checkpoints/phase1_.../best/best_model.zip --episodes 10
3. Diagnose: python -m infinite_maze.ml.diagnose --model checkpoints/phase1_.../best/best_model.zip --steps 2000
4. Watch:    python -m infinite_maze.ml.watch    --model checkpoints/phase1_.../best/best_model.zip --episodes 3
5. Decide:   If avg score meets phase target → move to next phase.
             If score is low but policy looks healthy → resume with more timesteps.
             If spam-RIGHT degenerate → tune reward weights in rewards.py and retrain.
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: gymnasium` | deps not installed | `pip install -r requirements-test.txt` |
| Blank pygame window | display driver issue | check `SDL_VIDEODRIVER` env var; unset it for watch |
| Score does not improve after 200k | reward shaping | run diagnose; if spam-RIGHT, increase `REWARD_DO_NOTHING` penalty |
| `--resume` model has wrong obs shape | env/model mismatch | ensure same `--phase` as original run |
