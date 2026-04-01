# Infinite Maze — RL Agent Training Plan

**Generated:** 2026-03-29  
**Agent mode:** ml-trainer  
**Scope:** Full training pipeline from scratch to elite-tier play

---

## Step 1 — Readiness Audit

### 1.1 `infinite_maze/ml/` module

| Status | Detail |
|--------|--------|
| ❌ Does not exist | The entire `infinite_maze/ml/` directory is absent |

**All ML files are missing.** Nothing has been built yet. This is the ground-zero starting point.

### 1.2 `utils/config.py` — ML constants

`GameConfig` contains only game constants. **No `ML_CONFIG` dict exists.** The following constants are available and reusable for ML work:

| Constant | Value | ML Use |
|----------|-------|--------|
| `PLAYER_START_X` | 80 | `X_MIN` / spawn x |
| `PLAYER_START_Y` | 223 | spawn y |
| `PLAYER_SPEED` | 1 | action step size |
| `PLAYER_WIDTH / HEIGHT` | 10 / 10 | AABB collision |
| `SCREEN_WIDTH / HEIGHT` | 640 / 480 | observation normalisation |
| `MAZE_ROWS / COLS` | 15 / 20 | maze structure |
| `MAZE_CELL_SIZE` | 22 | wall geometry |
| `PACE_UPDATE_INTERVAL` | 30 | seconds per pace increment |
| `MOVEMENT_CONSTANTS` | `{DO_NOTHING:0, RIGHT:1, LEFT:2, UP:3, DOWN:4}` | direct action mapping |

**Missing constants that must be added to `config.py`:**

```python
ML_CONFIG = {
    # Observation
    "MAX_PACE": 10,              # normalisation ceiling; at pace=10 the world shifts 10px per 10 ticks, which exactly cancels the player's maximum 1px/step throughput — any higher pace makes rightward progress mathematically impossible
    "EPISODE_SCORE_CAP": 600,    # truncation threshold (elite tier + buffer)
    "MAX_WALL_SCAN_DIST": 200,   # px, ceiling for wall-distance normalisation
    "GAP_SCAN_RADIUS": 110,      # px: vertical scan range for nearest-right-gap feature (5 cells × 22px)
    "CONSECUTIVE_BLOCKED_CAP": 50,  # normalisation ceiling for stuck-right counter

    # Pace simulation: reduced from the real-time equivalent (1800 = 30s × 60 ticks/s) to 300 ticks
    # so agents naturally encounter pace pressure during Phases 2–3 without curriculum injection.
    # At 300 ticks per increment, any episode lasting > 300 steps hits pace = 1; a Phase 3 agent
    # reaching score > 150 will regularly operate at pace 2–3. This eliminates the distribution-shift
    # failure where the policy trains entirely at pace = 0 before Phase 4 curriculum is introduced.
    "TICKS_PER_PACE_UPDATE": 300,

    # Pace shift cadence (mirrors engine.py: every 10 ticks shift world left)
    "PACE_SHIFT_INTERVAL": 10,

    # Reward weights
    "REWARD_MOVE_RIGHT":          1.0,
    "REWARD_MOVE_RIGHT_BLOCKED":  0.0,   # explicit — blocked RIGHT attempt yields no reward (no score change in engine)
    "REWARD_MOVE_LEFT":          -0.5,
    "REWARD_DO_NOTHING":    -0.1,
    "REWARD_MOVE_VERTICAL":  0.0,
    "REWARD_TERMINAL":      -10.0,

    # Training
    "DEFAULT_TIMESTEPS": 200_000,
    "CHECKPOINT_FREQ":    10_000,
    "EVAL_FREQ":          20_000,
    "EVAL_EPISODES":      10,
    "TENSORBOARD_LOG":    "runs/",
}
```

### 1.3 Test files

| File | Status |
|------|--------|
| `tests/unit/test_ml_environment.py` | ❌ Does not exist |
| `tests/integration/test_ml_training.py` | ❌ Does not exist |

### 1.4 Remaining codebase — what IS ready

| Component | File | ML-readiness |
|-----------|------|-------------|
| `Game(headless=True)` | `core/game.py` | ✅ Headless mode complete; `reset()`, `end()`, `isActive()`, `getPace()`, `getScore()`, `incrementScore()`, `decrementScore()` all present |
| `Player(headless=True)` | `entities/player.py` | ✅ Getter/setter API clean; `moveX()`, `moveY()`, `reset()` present |
| `Line.generateMaze()` | `entities/maze.py` | ✅ Static factory; produces `Line` list with full getter API |
| `Line.getXMax()` | `entities/maze.py` | ✅ Used for recycling logic |
| Collision logic | `core/engine.py` (lines 30–170) | ✅ Complete AABB checks for all 4 directions; must be **extracted** into `features.py`, not imported from `engine.py` |
| `config.MOVEMENT_CONSTANTS` | `utils/config.py` | ✅ Direct 0–4 action mapping |
| Pace shift logic | `core/engine.py` (≈ line 160) | ✅ `if ticks % 10 == 0: player.x -= pace; shift all lines` — extract to env |
| Test fixtures | `tests/fixtures/` | ✅ `pygame_mocks.py`, `game_fixtures.py`, `conftest.py` all exist |

### 1.5 Audit Summary

**Ready to use immediately:** `Game`, `Player`, `Line`, all config constants, test fixtures.  
**Missing — must build:** `infinite_maze/ml/` module (all files), `ML_CONFIG` in `config.py`, both test files.  
**Blocker:** `InfiniteMazeEnv` does not exist. Training cannot begin until the gymnasium wrapper is implemented and tested.

---

## Step 2 — Environment Health Check

**Status: BLOCKED — `InfiniteMazeEnv` not yet implemented.**

Once implemented, the following conditions must all pass before training begins:

1. `env.reset()` returns `(obs, info)` where `obs.shape == env.observation_space.shape` and `obs.dtype == np.float32`  
2. `env.step(action)` for all `action ∈ {0,1,2,3,4}` returns a 5-tuple `(obs, reward, terminated, truncated, info)`  
3. A blocked right move: `player.getX()` is unchanged after `step(1)` when the right wall check returns True  
4. After pace pushes `player.getX()` below `game.X_MIN`, `terminated == True`  
5. `env.observation_space.contains(obs)` is True every step (gymnasium compliance)

These are encoded as acceptance tests in `tests/unit/test_ml_environment.py`.

---

## Step 3 — Training Plan

### Observation Space Design

Flat `np.float32` array, shape `(53,)`:

| Index | Feature | Normalisation |
|-------|---------|---------------|
| 0 | `player.getX()` | `/ config.SCREEN_WIDTH` |
| 1 | `player.getY()` | `/ config.SCREEN_HEIGHT` |
| 2 | blocked right (bool → float) | 0.0 / 1.0 |
| 3 | blocked left (bool → float) | 0.0 / 1.0 |
| 4 | blocked up (bool → float) | 0.0 / 1.0 |
| 5 | blocked down (bool → float) | 0.0 / 1.0 |
| 6 | distance to nearest wall right | `/ ML_CONFIG["MAX_WALL_SCAN_DIST"]`, clipped [0,1] |
| 7 | distance to nearest wall left | same |
| 8 | distance to nearest wall up | same |
| 9 | distance to nearest wall down | same |
| 10 | `game.getPace()` | `/ ML_CONFIG["MAX_PACE"]`, clipped [0,1] |
| 11 | distance from death: `(player.getX() − game.X_MIN) / (game.X_MAX − game.X_MIN)` | clipped [0,1]; 1.0 = at X_MAX (safe), approaching 0.0 = near termination boundary |
| 12 | consecutive ticks blocked right | `/ ML_CONFIG["CONSECUTIVE_BLOCKED_CAP"]`, clipped [0,1]; rises each tick the agent cannot move right, resets to 0 on a successful rightward step |
| 13–52 | local wall grid | 40 binary features — `ML_CONFIG["GRID_COLS"]` (4) × `ML_CONFIG["GRID_ROWS"]` (5) × 2; computed by `features.get_wall_grid()`; for each cell: `has_right_wall` then `has_bottom_wall`; columns 0–3 scan rightward from player right edge (22 px each); rows −2 to +2 centred on player |

### Action Space

`gymnasium.spaces.Discrete(5)` — maps directly to `config.MOVEMENT_CONSTANTS`.

### Episode Lifecycle (ML)

```
reset() → new Game(headless=True), new Player(headless=True), Line.generateMaze(...)
          tick_counter = 0

step(action):
  1. Compute collision flags (replicated from engine.py, no import)
  2. Apply action → move player if not blocked, update score
  3. tick_counter += 1
  4. if tick_counter % PACE_SHIFT_INTERVAL == 0:
       player.setX(player.getX() - game.getPace())
       shift all line X coords by -game.getPace()       # pace shift only; no clamp here
  5. # Position adjustments — applied EVERY step (mirrors engine.py "Position Adjustments" block)
     if player.getX() > game.X_MAX:
       player.setX(game.X_MAX)
       shift all line X coords by -player.getSpeed()   # maze scrolls instead of player advancing further
     player.setY(max(player.getY(), game.Y_MIN))
     player.setY(min(player.getY(), game.Y_MAX))
  6. # Line recycling — applied EVERY step (not only on pace-shift ticks)
     recycle off-screen lines to right edge (see §4 line recycling note)
  7. if tick_counter % TICKS_PER_PACE_UPDATE == 0:
       game.pace += 1
  8. terminated = player.getX() < game.X_MIN
  9. truncated  = game.getScore() >= EPISODE_SCORE_CAP
  10. reward = compute_reward(action, blocked_flags, terminated, game, config)
  11. return _get_obs(), reward, terminated, truncated, info
```

**Note on pace timing:** The live game uses real-time seconds via `Clock`. In the headless ML env, `Clock.update()` would require real elapsed time and is unusable for fast rollout. Instead, pace is incremented by tick count using `TICKS_PER_PACE_UPDATE = 300`. This is intentionally lower than the real-game equivalent (1800 = 30 s × 60 ticks/s) to ensure agents encounter pace pressure during normal training episodes in Phases 2–3. At 300 ticks, any episode lasting > 300 steps hits pace = 1; a Phase 3 agent at score > 150 will regularly operate at pace 2–3. This eliminates the distribution-shift failure that would occur if pace = 0 throughout the pre-Phase-4 curriculum.

---

## Phase Table

### Phase 0 — Random Baseline

| Field | Value |
|-------|-------|
| **Goal** | Confirm the env is gymnasium-compliant; establish untrained performance baseline |
| **Algorithm** | None (random policy) — `env.action_space.sample()` |
| **Timesteps** | 10,000 (≈ 100 short episodes) |
| **Key hyperparameters** | N/A |
| **Reward shaping** | Default weights from `ML_CONFIG` |
| **Success criterion** | Env runs 10,000 steps without exception; mean episode score recorded |
| **Files to create/modify** | `infinite_maze/ml/environment.py`, `infinite_maze/ml/features.py`, `infinite_maze/ml/rewards.py`, `tests/unit/test_ml_environment.py` |

**Expected baseline score:** 2–10 (random movement produces net-zero rightward progress; pace quickly terminates).  
**Why this matters:** Any subsequent trained agent that can't beat ~10 has a fundamental env or reward bug.

---

### Phase 1 — Survival (target: mean score > 25)

| Field | Value |
|-------|-------|
| **Goal** | Learn to not immediately die; avoid pace-boundary termination for multiple seconds |
| **Algorithm** | **PPO** — handles dense reward, discrete actions, and short episodes well; lower sample complexity than DQN for this action space |
| **Timesteps** | 100,000 |
| **Key hyperparameters** | `learning_rate=3e-4`, `gamma=0.99`, `n_steps=512`, `batch_size=64`, `n_epochs=10`, `ent_coef=0.01`, `clip_range=0.2` |
| **Reward shaping** | Standard weights from `ML_CONFIG` — no additional survival bonus. The `REWARD_DO_NOTHING=-0.1` step penalty is sufficient to prevent stalling; a `+0.01/step` bonus was considered but is ineffective (invisible against the `+1.0` right-move reward and does not change the relative ordering of actions). |
| **Success criterion** | Mean episode score > 25 over last 10 eval episodes (measured every 20k steps) |
| **Checkpoint frequency** | Every 10,000 timesteps |
| **Files to create/modify** | `infinite_maze/ml/train.py`, `infinite_maze/utils/config.py` (add `ML_CONFIG`) |

**Risk:** Agent may learn to stall (DO_NOTHING) to avoid leftward movement penalty while surviving briefly. Counter with the `REWARD_DO_NOTHING=-0.1` step penalty.

---

### Phase 2 — Rightward Bias (target: mean score > 50)

| Field | Value |
|-------|-------|
| **Goal** | Establish a consistent rightward bias; agent should prefer RIGHT action whenever unblocked |
| **Algorithm** | **PPO** (continue from Phase 1 checkpoint) |
| **Timesteps** | 200,000 |
| **Key hyperparameters** | `learning_rate=1e-4` (decay from Phase 1), `gamma=0.995` (longer horizon — rightward gains compound), `n_steps=1024`, `batch_size=128`, `ent_coef=0.005` |
| **Reward shaping** | Remove per-step survival bonus from Phase 1; restore standard weights: `RIGHT=+1.0`, `LEFT=-0.5`, `DO_NOTHING=-0.1`, `UP/DOWN=0.0`, `TERMINAL=-10.0` |
| **Success criterion** | Mean episode score ≥ 50 (Beginner tier) over last 10 eval episodes; RIGHT action frequency > 40% of non-blocked steps |
| **Checkpoint frequency** | Every 10,000 timesteps |
| **Files to create/modify** | `infinite_maze/ml/train.py` (add `--resume` flag and checkpoint loading) |

**Risk:** Agent may learn to zigzag purely right without using vertical movement, getting stuck at vertical walls. If phase 3 entry score plateaus below 80, re-examine entropy coefficient and wall-distance observation quality.

**Pace exposure:** With `TICKS_PER_PACE_UPDATE = 300`, Phase 2 episodes reaching score ≥ 50 will naturally encounter pace = 1. No `start_pace_range` override is needed in Phase 2. If evaluations show mean episode length is consistently below 300 steps (agent dies before pace kicks in), add `start_pace_range=[0, 1]` as an optional argument to `reset()` to begin gentle pace exposure.

---

### Phase 3 — Maze Navigation (target: mean score > 150)

| Field | Value |
|-------|-------|
| **Goal** | Learn to use vertical movement to find rightward gaps; solve multi-cell navigational detours |
| **Algorithm** | **PPO** (continue from Phase 2 checkpoint) with `n_envs=4` parallel environments using `DummyVecEnv` (in-process, no IPC overhead). Provides exploration diversity across different random mazes, which is critical for learning generalised vertical navigation. Switch to `SubprocVecEnv` only if `Line.generateMaze()` proves to be a CPU bottleneck during reset. |
| **Timesteps** | 500,000 |
| **Key hyperparameters** | `learning_rate=5e-5`, `gamma=0.995`, `n_steps=2048`, `batch_size=256`, `n_epochs=10`, `ent_coef=0.05` (**raised from originally-planned 0.02** — essential to prevent DOWN-only action collapse), `clip_range=0.1`, **`n_envs=4`** |
| **Reward shaping** | Full Phase 3 stack in effect: **(1)** `compute_reward()` with Phase 3 weights; **(2)** `phase3_shaping()` called from `environment.py`, tracking `_prev_gap_offset` and `_consecutive_blocked`; **(3)** BFS curriculum bonus via `bfs_optimal_action()`. |
| **Reward weights** | `RIGHT unblocked → +1.0`; `RIGHT blocked → −0.3`; `LEFT unblocked → −1.5` (**raised from −0.5** to close LEFT→RIGHT exploit); `LEFT blocked → −0.3` (symmetric); `DO_NOTHING → −0.1`; `UP/DOWN unblocked → 0.0`; `UP/DOWN when blocked-right → +0.1`; `TERMINAL → −10.0`; BFS match → `+0.05` when action equals `bfs_optimal_action()` |
| **`phase3_shaping()`** | Gap-approach bonus `+0.3` when `new_gap_offset < prev_gap_offset` (moving toward gap); escape bonus `+0.05` when stuck-counter drops. `GAP_SCAN_RADIUS=220` px (10-cell vertical scan alternating above/below player). |
| **BFS curriculum** | `features.bfs_optimal_action(player, lines, game)` runs a two-phase BFS each step: (1) find all right-unblocked candidate cells in the local wall grid; (2) filter pockets by forward reachability (`MIN_FORWARD=2` cells). When agent's action matches BFS optimal, `+REWARD_BFS_MATCH(0.05)` is added. BFS action is **never shown to the agent** — it is only used to shape the reward signal. |
| **BFS exit gate** | Before advancing to Phase 4: temporarily set `REWARD_BFS_MATCH=0.0` and re-evaluate deterministically over 20 episodes (seed 42). If mean score holds ≥ 250 → genuine navigation skill; advance to Phase 4. If score collapses → policy was cue-following; continue Phase 3 training. Restore `REWARD_BFS_MATCH=0.05` regardless (Phase 4 permanently zeroes it). |
| **Success criterion** | Mean episode score ≥ 150 (Intermediate tier) over last 10 eval episodes; LEFT action frequency < 5%; vertical action frequency > 20% when `blocked_right=True` |
| **Checkpoint frequency** | Every 10,000 timesteps |
| **Files modified** | `infinite_maze/ml/features.py` (added `is_blocked_right_at_y`, `nearest_right_gap_offset`, `_build_adjacency`, `bfs_optimal_action`); `infinite_maze/ml/rewards.py` (updated weights, extended `compute_reward` signature with `bfs_action`, `phase`); `infinite_maze/ml/environment.py` (wired `phase3_shaping` and `bfs_optimal_action`); `infinite_maze/utils/config.py` (added `REWARD_MOVE_RIGHT_BLOCKED`, `REWARD_VERTICAL_WHEN_BLOCKED_UP/DOWN`, `REWARD_BFS_MATCH`, updated `GAP_SCAN_RADIUS` and `REWARD_MOVE_LEFT`) |

#### Phase 3 Training History

Five consecutive runs identified and resolved a series of cascading reward engineering bugs before arriving at the current configuration.

| Run | Key config | Deterministic score | Dominant issue |
|-----|-----------|---------------------|----------------|
| 1 | `ent_coef=0.05`, base rewards | ~250, 87% DOWN, 0% UP | `phase3_shaping()` never wired — Phase 3 was functionally identical to Phase 2 |
| 2 | Gap bonus wired, `ent_coef=0.1` | 186 stochastic 564 | 3× stochastic/deterministic gap — high entropy caused chaotic test-time policy |
| 3 | `GAP_SCAN_RADIUS=220`, `REWARD_VERTICAL_WHEN_BLOCKED=0.3` | 52% UP but stoch/det gap persists | `REWARD_VERTICAL_WHEN_BLOCKED=0.3` rewarded oscillation near walls indefinitely |
| 4 | `REWARD_VERTICAL_WHEN_BLOCKED=0.0` | 41% LEFT, 0% UP | Zeroing blocked-vertical exposed the `LEFT→RIGHT = +0.5` net-profit loop |
| 5 | Restored 0.1 blocked-vert, `REWARD_MOVE_LEFT=−0.5`, fresh from Phase 2 | 144.2, 38.8% LEFT, 0% UP | `LEFT→RIGHT` loop still profitable at −0.5 |

**Root cause (runs 4–5):** `REWARD_MOVE_LEFT(−0.5) + REWARD_MOVE_RIGHT(+1.0) = +0.5` net profit per 2-step cycle. The agent discovered this loop and exploited it instead of navigating.

**Fix applied (run 6 — current):** `REWARD_MOVE_LEFT: −0.5 → −1.5` (loop now yields `−0.5` net, unprofitable) + `bfs_optimal_action()` providing a positive navigation signal via `REWARD_BFS_MATCH: +0.05`.

**Mitigation summary:** The gap-offset observation (obs[12]) and consecutive-blocked counter (obs[11]) give the network explicit signals for where the nearest gap is and how long it has been stuck. `phase3_shaping()` fires only when the agent moves *toward* the gap, preventing arbitrary vertical flailing from being rewarded. The BFS bonus reinforces correct navigational choices without exposing the path to the agent. The LEFT penalty at −1.5 permanently closes the retreat-and-advance exploit.

---

### Phase 4 — Pace Adaptation (target: mean score > 300)

| Field | Value |
|-------|-------|
| **Goal** | Generalise to increasing pace; avoid score collapse as `game.pace` grows beyond 5–8 |
| **Algorithm** | **PPO** (continue from Phase 3 checkpoint) |
| **Timesteps** | 1,000,000 |
| **Key hyperparameters** | `learning_rate=3e-5`, `gamma=0.999` (very long horizon — pace compounds over time), `n_steps=4096`, `batch_size=512`, `n_epochs=5`, `ent_coef=0.005`, `clip_range=0.05` |
| **Reward shaping** | Remove Phase 3 path-finding bonus. Add urgency shaping: scale the terminal penalty by `(1 + game.getPace() / ML_CONFIG["MAX_PACE"])` so later-game deaths are punished more heavily |
| **Curriculum** | Domain randomisation: at reset, sample `game.pace` uniformly from `[0, 5]` with probability 0.7, and `[5, 10]` with probability 0.3. With `TICKS_PER_PACE_UPDATE = 300`, Phases 2–3 already include organic pace growth up to ~3–5 during long episodes; this curriculum extends coverage to high *starting* paces (5–10) that would otherwise require an impractically long single episode to reach naturally. |
| **Success criterion** | Mean episode score ≥ 300 (Advanced tier) over last 10 eval episodes; agent maintains score > 50 when episode starts at pace=5 |
| **Checkpoint frequency** | Every 50,000 timesteps |
| **Files to create/modify** | `infinite_maze/ml/environment.py` (add `start_pace_range` parameter to `reset()`), `infinite_maze/ml/rewards.py` (add `urgency_scaled_terminal()`), `tests/integration/test_ml_training.py` |

**Risk:** This is the hardest phase. Pace shifts happen every 10 ticks; at pace=8 the world moves 8px per 10 ticks, leaving a net rightward gain of only 2px per 10 steps even with continuous unblocked RIGHT movement (player earns 10px, pace removes 8px). At `MAX_PACE=10` the net gain reaches exactly zero — the agent cannot advance at all even at perfect play. The agent must therefore reach high scores and survive before pace climbs past the survivable range (~8–9). If score plateaus here, inspect whether the `pace` signal (obs index 10) is being used effectively — consider adding pace *delta* (change since last step) as an additional observation feature.

---

### Phase 5 — Expert Play (stretch goal: mean score > 500)

| Field | Value |
|-------|-------|
| **Goal** | Achieve Master-tier performance; maintain high rightward velocity under maximum pace pressure |
| **Algorithm** | **PPO** (continue from Phase 4 checkpoint) or switch to **RecurrentPPO** (SB3-Contrib) if the agent struggles with non-stationary maze recycling |
| **Timesteps** | 2,000,000 |
| **Key hyperparameters** | PPO: `learning_rate=1e-5`, `gamma=0.999`, `n_steps=8192`, `batch_size=1024`; RecurrentPPO: same + `lstm_hidden_size=64`, `n_lstm_layers=1` |
| **Reward shaping** | Standard weights only — no shaping bonuses. Let the dense score signal drive learning |
| **Curriculum** | Randomise starting pace across `[0, MAX_PACE/2]` uniformly; include occasional high-pace starts `[MAX_PACE/2, MAX_PACE]` |
| **Success criterion** | Mean episode score ≥ 500 (Master tier) over last 10 eval episodes |
| **Checkpoint frequency** | Every 100,000 timesteps |
| **Files to create/modify** | `infinite_maze/ml/train.py` (add `--algorithm rcppo` option if RecurrentPPO is needed), add `stable-baselines3[extra]` to dependencies if RecurrentPPO is used |

**Note:** RecurrentPPO requires `sb3-contrib`. Only add it if Phase 4 results suggest the agent needs memory. Do not add it speculatively.

---

## Step 4 — Curriculum & Implementation Order

### Implementation Checklist

Tackle files in this exact order. Each file has one acceptance condition.

| # | File | Acceptance Condition |
|---|------|---------------------|
| 1 | `infinite_maze/utils/config.py` | `GameConfig.ML_CONFIG` dict exists with all keys listed in §1.2; `from infinite_maze.utils.config import config; config.ML_CONFIG["MAX_PACE"]` returns `10` |
| 2 | `infinite_maze/ml/__init__.py` | Module imports without error: `from infinite_maze.ml import environment` |
| 3 | `infinite_maze/ml/features.py` | `get_obs(player, lines, game)` returns a `np.float32` array of shape `(53,)`; all values in `[0.0, 1.0]`; `is_blocked(player, lines, direction)` matches engine.py collision results for a known wall configuration; `get_wall_grid(player, lines)` returns a `(40,)` binary float32 array where 0.0 indicates no wall in the cell and 1.0 indicates a wall present |
| 4 | `infinite_maze/ml/rewards.py` | `compute_reward(action, blocked_flags, terminated, game, bfs_action=-1, phase=1)` returns the correct float. Reward table: right unblocked → `+1.0`, right blocked → `−0.3`, left unblocked → `−1.5`, left blocked → `−0.3`, do-nothing → `−0.1`, up/down unblocked → `0.0`, up/down when blocked-right → `+0.1`, terminal → `−10.0`, BFS match (phase ≥ 3) → `+0.05`. Phase 3 shaping is a separate pure function: `phase3_shaping(action, prev_gap_offset, new_gap_offset, prev_blocked_count, new_blocked_count, blocked_right) → float`; `environment.py` tracks `_prev_gap_offset` and `_consecutive_blocked` between steps and passes them as arguments — `rewards.py` holds no state. |
| 5 | `infinite_maze/ml/environment.py` | `InfiniteMazeEnv` passes `gymnasium.utils.env_checker.check_env(env)` without errors; all 5 acceptance tests in §2 pass |
| 6 | `tests/unit/test_ml_environment.py` | All unit tests pass headlessly under `pytest tests/unit/test_ml_environment.py` |
| 7 | `infinite_maze/ml/train.py` | `python -m infinite_maze.ml.train --timesteps 1000` completes without error, writes a checkpoint to `checkpoints/`, accepts `--resume checkpoints/model_1000_steps.zip` to continue |
| 8 | `infinite_maze/ml/evaluate.py` | `python -m infinite_maze.ml.evaluate --model checkpoints/model_1000_steps.zip --episodes 3` prints per-episode scores and mean. Support an optional `--seed` argument passed to `env.reset(seed=seed)` to enable reproducible evaluation baselines across phases. |
| 9 | `tests/integration/test_ml_training.py` | `pytest tests/integration/test_ml_training.py` completes in < 60 s; PPO trains for 1,000 timesteps without exception; final mean reward is finite |

### Key Technical Notes for Implementation

#### Extracting collision logic (`features.py`)

The AABB collision checks in `engine.py` must be **replicated** (not imported) in `features.py`. The four directional checks are:

```
RIGHT blocked by horizontal line: player.y ≤ line.yStart ≤ player.y+h  AND  player.x+w+speed == line.xStart
RIGHT blocked by vertical line:   player.x+w ≤ line.xStart  AND  player.x+w+speed ≥ line.xStart  AND  y-overlap
LEFT blocked by horizontal line:  player.y ≤ line.yStart ≤ player.y+h  AND  player.x-speed == line.xEnd
LEFT blocked by vertical line:    player.x ≥ line.xEnd  AND  player.x-speed ≤ line.xEnd  AND  y-overlap
DOWN blocked by horizontal line:  player.y+h ≤ line.yStart  AND  player.y+h+speed ≥ line.yStart  AND  x-overlap
DOWN blocked by vertical line:    x-overlap  AND  player.y+h+speed == line.yStart
UP blocked by horizontal line:    player.y ≥ line.yStart  AND  player.y-speed ≤ line.yStart  AND  x-overlap
UP blocked by vertical line:      x-overlap  AND  player.y-speed == line.yEnd
```

#### Wall distance scanning (`features.py`)

Scan from the player's bounding box edge outward in each direction. For each line, compute the perpendicular distance only if the line would block movement in that direction. Return the minimum distance found, capped at `MAX_WALL_SCAN_DIST`.

#### Line recycling in `step()` (`environment.py`)

After every pace shift, replicate the recycling logic from `engine.py`:
```python
x_max = Line.getXMax(lines)
for line in lines:
    if line.getXStart() < config.PLAYER_START_X:
        line.setXStart(x_max)
        if line.getXStart() == line.getXEnd():   # vertical — both coords same x
            line.setXEnd(x_max)
        else:                                      # horizontal — span one cell
            line.setXEnd(x_max + config.MAZE_CELL_SIZE)
```

#### `nearest_right_gap_offset` algorithm (`features.py`)

Scan vertically within `GAP_SCAN_RADIUS` pixels of the player's current Y to find the nearest Y coordinate from which a RIGHT move would be unblocked:

1. Build a candidate list at **5-pixel increments** stepping outward from `player.getY()` in both directions, clamped to `[game.Y_MIN, game.Y_MAX − player.getHeight()]`. Do **not** use `MAZE_CELL_SIZE` (22 px) increments — gaps are valid at any Y within a cell, not only at cell-aligned positions. Scanning at 22 px steps would miss openings 5–15 px away and cause the feature to return 0.5 (no gap found) when a valid gap is nearby.
2. For each candidate `y_candidate`, call `is_blocked_right_at_y(player, lines, y_candidate)` — a variant of the RIGHT collision check that substitutes `y_candidate` for `player.getY()` while keeping `player.getX()` and all line positions fixed.
3. The first (closest by absolute vertical distance) `y_candidate` for which `is_blocked_right_at_y` returns `False` is the gap.
4. If no gap is found within the scan radius, return `0.5`.
5. Otherwise return `(gap_y − player.getY()) / GAP_SCAN_RADIUS * 0.5 + 0.5`, clipped to `[0.0, 1.0]`.
   - Value < 0.5 → gap is above the player.
   - Value > 0.5 → gap is below the player.
   - Value = 0.5 → gap is at the same Y (i.e. the current row is itself unblocked, or no gap found).

The acceptance condition in §4 ("returns 0.5 when no gap is found; returns a value reflecting the correct signed direction when a gap exists") must be verified with a hand-crafted line configuration in `tests/unit/test_ml_environment.py`.

---

#### Phase 3 reward shaping state (`environment.py`)

The Phase 3 directional shaping bonus compares gap offset between consecutive steps. `environment.py` must maintain two instance variables across steps:

- `self._prev_gap_offset: float` — gap offset computed at the end of the previous step (initialised to `0.5` at `reset()`).
- `self._consecutive_blocked: int` — count of consecutive steps where the RIGHT action was blocked; reset to `0` on any successful rightward step (initialised to `0` at `reset()`).

`rewards.phase3_shaping(action, prev_gap_offset, new_gap_offset, prev_blocked_count, new_blocked_count)` is a **pure function** — it takes both values as arguments. `rewards.py` holds no state. `environment.py` reads the values, calls the function, then overwrites `self._prev_gap_offset` with the new observation's gap offset before returning from `step()`.

---

#### `moveX` / `moveY` call convention

`Player.moveX(units)` computes `new_x = position[0] + (units * self.speed)`. At `PLAYER_SPEED=1`, `player.moveX(1)` and `player.moveX(player.getSpeed())` are equivalent. Always pass the integer direction (`1` or `−1`), not a pre-multiplied pixel value, so the call remains correct if speed changes in the future.

---

#### `game.X_MAX` is a float — use `int()` in collision checks (`features.py`)

`game.X_MAX` is defined as `WIDTH / 2 = 640 / 2 = 320.0` using Python 3 float division. After the X_MAX clamp fires, `player.getX()` becomes `320.0`. The RIGHT collision check uses `player.getX() + player.getWidth() + player.getSpeed() == line.getXStart()` — a float-vs-int equality that can silently evaluate to `False`. In `features.py`, cast with `int(game.X_MAX)` wherever X_MAX is compared against integer line coordinates. Simple inequality comparisons (`>`, `<`) are unaffected; only `==` checks against line coords are at risk.

---

#### Do not call `game.end()` inside `step()` (`environment.py`)

`game.end()` sets `game.over = True`. The `step()` method must **not** call it. Compute `terminated = player.getX() < game.X_MIN` and return it in the tuple; do not mutate game state for termination. While `game.reset()` does restore `game.over = False`, coupling `step()` to that side-effect makes the env harder to reason about and could cause subtle bugs if call order changes. Termination is a returned value, not a state mutation.

---

#### Maze generation performance (`entities/maze.py`)

`Line.generateMaze()` uses a randomised union-find loop that rescans the full wall list on every iteration. For a 15×20 grid this generates ~600 initial segments and the loop is O(n²) in the worst case. During training, `reset()` is called thousands of times per minute of wall-clock time.

**Benchmark before Phase 3:** Run 10,000 resets and measure total elapsed time *before* beginning Phase 3 training. With `n_envs=4` and `n_steps=2048`, there are roughly `750k / (2048 × 4) ≈ 92` rollout cycles, each triggering 4 resets. If mean reset latency exceeds **5 ms**, overhead approaches ~1.8 s per rollout cycle and will meaningfully slow Phase 3. If latency exceeds 5 ms, pre-generate a pool of 1,000 mazes at startup and sample from it at each `reset()` call. Do not optimise before measuring — validate latency first.

---

#### Checkpoint and resume (`train.py`)

```bash
# Start fresh
python -m infinite_maze.ml.train --timesteps 200000

# Resume from checkpoint
python -m infinite_maze.ml.train --timesteps 200000 --resume checkpoints/ppo_maze_100000_steps.zip

# Override timesteps for a short smoke run
python -m infinite_maze.ml.train --timesteps 1000
```

Use SB3's `CheckpointCallback` with `save_freq=ML_CONFIG["CHECKPOINT_FREQ"]` and `EvalCallback` with `eval_freq=ML_CONFIG["EVAL_FREQ"]`. When `--resume` is provided, load with `PPO.load(path, env=env)` — this restores the full policy, value network, and optimizer state.

---

## Step 5 — Summary

### Phase Table (condensed)

| Phase | Goal | Algorithm | Timesteps | Score Target | Checkpoint Freq |
|-------|------|-----------|-----------|-------------|-----------------|
| 0 — Baseline | Env compliance + random perf | Random policy | 10k | Baseline recorded | N/A |
| 1 — Survival | Stay alive | PPO (fresh) | 100k | > 25 | 10k |
| 2 — Rightward bias | Move right reliably | PPO (resume P1) | 200k | > 50 | 10k |
| 3 — Maze navigation | Use vertical gaps | PPO (resume P2, n_envs=4) | 500k | > 150 | 10k |
| 4 — Pace adaptation | Survive rising pace | PPO (resume P3) | 1M | > 300 | 50k |
| 5 — Expert play | Master-tier scoring | PPO or RecurrentPPO (resume P4) | 2M | > 500 | 100k |

**Total budget (excluding Phase 0):** ~3.55M timesteps across all phases (Phase 3 reduced from 750k to 500k).  
**Resume-friendly:** Every phase starts from the previous phase's final checkpoint. Training can be stopped at any checkpoint and resumed with `--resume`.

---

### Overall Strategy

The fundamental challenge of Infinite Maze for an RL agent is twofold: the observation space is local (the agent sees only its immediate surroundings, not the full maze ahead), and the termination condition is directional (you must move right to survive, but walls make that non-trivial). The training strategy builds capability incrementally — first teaching bare survival, then establishing a rightward bias, then the critical skill of vertical repositioning to unlock horizontal paths, and finally hardening the policy against accelerating pace pressure. The biggest risk to successful training is **getting stuck in Phase 3**: the agent may learn a locally-optimal policy (move right whenever possible, idle otherwise) that achieves ~50–80 points but never discovers that vertical moves followed by rightward moves yield far higher returns. This manifests as a premature plateau in episode reward after Phase 2. The mitigation is now built directly into the observation space: obs[12] (`nearest_right_gap_offset`) gives the agent explicit signal for *where* the nearest rightward gap is relative to its current Y position, and obs[13] (`consecutive_ticks_blocked_right`) signals urgency when it has been stuck. The directional shaping bonus in `rewards.py` only fires when the agent moves *toward* the gap — not for arbitrary vertical moves — which removes the degenerate reward path present in simpler designs. Running 4 parallel environments (`n_envs=4`) ensures the policy is exposed to diverse maze layouts each rollout, reducing the chance of overfitting to a single navigation pattern. If a plateau below 80 persists at the 250k-step checkpoint, the escalation path (a second gap-scan at double radius as obs[14]) should resolve it before the full 750k budget is exhausted.
