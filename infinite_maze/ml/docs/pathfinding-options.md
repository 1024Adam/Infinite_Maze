# Pathfinding Lookahead Options

**Context:** The Phase 3 agent has successfully learned bidirectional vertical navigation but repeatedly stalls at mean scores of 180–250. The root cause is **temporal depth**: the agent can see which direction a gap is (via `nearest_right_gap_offset`) but cannot see that reaching it requires 2–4 intermediate moves. It optimises for immediate step-reward rather than multi-step navigation sequences.

These three options address that limitation in increasing order of complexity.

---

## Option 1 — Wider Wall Grid

**What it does:** Expand the local wall grid from 4 columns × 5 rows to a larger coverage area (e.g. 10 columns × 11 rows), giving the network direct pixel-level visibility of ~220px right and ~110px above/below the player. The network can implicitly learn lookahead from this if given enough training data — it becomes a spatial pattern-matching problem rather than a temporal-planning problem.

**How it works:**
- Change `GRID_COLS: 4 → 10`, `GRID_ROWS: 5 → 11` in `ML_CONFIG`
- Observation shape grows from `(53,)` to `(13 + 10×11×2) = (233,)`
- No algorithm changes — `get_wall_grid()` in `features.py` already supports arbitrary dimensions
- **Must retrain from scratch** — obs shape change breaks all existing checkpoints

**Reward / algorithm changes:** None.

**Pros:**
- Minimal code change — one config edit and restart
- No handcrafted algorithm; the network discovers what the wider context means on its own

**Cons:**
- Obs vector grows 4.4×, slowing training (~30% more compute per step)
- Implicit lookahead is unreliable — the network may still fail to chain 4-step corridors even with full visibility, because PPO's 512-step rollout buffer averages over many situations
- Does not help with **pace awareness** — the agent still can't plan around the leftward drift

---

## Option 2 — BFS Path Features (Recommended)

**What it does:** At each step, run a fast breadth-first search from the player's current cell to find the next 3–5 moves that lead toward the nearest rightward opening. Encode that path as a small one-hot feature vector appended to the observation. The agent is given an explicit short-horizon plan; RL learns when to follow it and when to deviate (e.g. under pace pressure, or when a faster route opens up).

**How it works:**

1. Add `bfs_next_moves(player, lines, game, depth=5) → np.ndarray` to `features.py`
   - Build a cell-adjacency graph from current `Line` positions
   - Run BFS from `(player.getX(), player.getY())` toward the rightmost reachable X
   - Return the first `depth` actions as a one-hot encoded flat array: shape `(depth × 5,)` = `(25,)`
   - If no path found within `depth` steps, return all-zeros
2. Append to obs in `get_obs()` → new shape `(53 + 25) = (78,)`
3. Recompute every step (walls shift with pace, so the path can become stale)

**What the model still has to learn:**
- When to follow the BFS path vs deviate (e.g. pace forces a different escape route)
- How aggressively to navigate (the BFS path is the *shortest* path, not the *safest* under pace pressure)
- Adaptation when the path becomes stale mid-execution due to wall recycling
- Long-term survival: pacing, score maximisation, avoiding terminal states

**Pros:**
- Solves temporal depth directly — the agent is told "here's the next corridor"
- BFS on a 15×20 cell grid takes < 1 ms; negligible training overhead
- Obs shape increases modestly: `(53,) → (78,)`
- Existing checkpoints reusable with a small obs-shape adapter (load weights, extend input layer with zeros)
- The RL problem becomes pace management + path-following vs pure navigation, which is a much easier credit assignment problem for PPO

**Cons:**
- BFS path becomes stale on the tick a pace shift moves walls — need to handle the 1-step lag carefully
- Requires building a cell adjacency extractor from `Line` segments (~80 lines in `features.py`)
- Still model-free: path features are a richer input, not a planner — the agent must still generalise

**Reward changes:** None required. The existing `phase3_shaping` gap bonus can be simplified or removed once path features are active, since the BFS already encodes directional guidance.

---

## Option 3 — Monte Carlo Tree Search (MCTS)

**What it does:** At each step, use the environment as a forward simulator and run MCTS to evaluate candidate action sequences before committing. The trained PPO policy seeds the rollout distribution (as in AlphaZero); MCTS improves single-step action selection at inference time using tree-search lookahead.

**How it works:**
- Implement an MCTS planner that clones the environment state (a copy of `Game`, `Player`, `Lines`), rolls out N simulated episodes from each candidate action, and returns the action with the highest estimated value
- The PPO network provides the prior action probabilities and value estimate at each node
- At training time, the MCTS policy targets replace raw PPO targets (as in MuZero/AlphaZero)

**Pros:**
- Theoretically the strongest option — explicit lookahead with policy guidance
- Can plan around pace shifts explicitly (the simulator includes `PACE_SHIFT_INTERVAL` logic)

**Cons:**
- Major implementation effort: tree data structures, state cloning, visit count management, separate value network training
- Our environment is headless but not designed for fast state cloning — `Line` objects would need `__copy__` support
- At inference time, MCTS requires hundreds of simulator calls per step — reduces to ~5–10 fps even with a fast env, which is impractical for training at scale
- This is genuine overkill for a 2D maze game

---

## Recommendation: Option 2 — BFS Path Features

Option 2 addresses the root cause (temporal depth) with minimal cost and no architectural change to the RL algorithm. The BFS path directly answers the question the agent is currently unable to answer from its observation: *"which moves would get me through the next wall?"*

The critical distinction from "automated solving": the BFS provides the next 3–5 moves toward an opening, but the agent still has to:
- Decide whether to follow the path or take a different route under pace pressure
- Time its moves to avoid being pushed left while navigating vertically
- Generalise to the fact that walls shift and the plan goes stale mid-execution

Option 1 (wider grid) is a cheaper first attempt but unlikely to be sufficient alone — the implicit lookahead it enables requires far more training data to emerge. It could be combined with Option 2 at low cost.

Option 3 is ruled out by implementation complexity and inference-time performance requirements.

**Implementation order if Phase 3 stalls again:**
1. Implement `bfs_next_moves()` in `features.py` with `depth=5`
2. Extend `get_obs()` to append the 25 BFS features → new shape `(78,)`
3. Retrain from Phase 2 best (obs shape change requires fresh load with zero-padded input layer or full retrain)
4. Optionally remove `phase3_shaping` gap bonus once BFS features are confirmed working — it becomes redundant

---

*Last updated: 2026-04-01*
