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

## Option 2 — BFS Curriculum Reward (Recommended)

**What it does:** At each step, run a fast BFS from the player's current position to find the optimal next action toward the nearest rightward opening. Give a small reward bonus if the agent's chosen action matches the BFS-optimal action. This teaches genuine navigation skill — the agent is *rewarded for making the right move* rather than being shown what the right move is. The BFS signal is removed entirely at Phase 4+, at which point the agent relies solely on what it has learned.

**Why not BFS-as-observation (the original Option 2):** Encoding the BFS path directly into the observation causes **shortcut learning** — the agent converges on "action = decode(obs[53:78])" in a few thousand steps and never learns to navigate. Remove the features at inference time and the policy collapses entirely. The RL machinery becomes vestigial. The agent is a path-follower, not a navigator.

**How it works:**

1. Add `bfs_optimal_action(player, lines, game) → int` to `features.py`
   - Build a cell-adjacency graph from current `Line` positions (full visible 15×20 window)
   - **Phase 1 — find candidate gaps:** collect all cells from which RIGHT is unblocked, ordered by proximity
   - **Phase 2 — evaluate forward reachability:** for each candidate gap (nearest first), run a secondary BFS forward from that gap to measure how many additional rightward cells are reachable beyond it. Discard candidates whose forward reachability is below a minimum threshold (e.g. 2 cells) — these are pockets.
   - Select the nearest gap that passes the reachability threshold; return the first action on the path to it
   - Fall back to the nearest gap regardless of reachability if no gap passes the threshold (better than doing nothing)
   - Return `DO_NOTHING` only if no rightward gap exists in the entire visible window
2. In `environment.py` `step()`, after computing `blocked_flags`, call `bfs_optimal_action()`
3. In `rewards.py`, add `REWARD_BFS_MATCH: 0.05` to `ML_CONFIG`
4. Add the bonus to `compute_reward()` when `phase >= 3` and `action == bfs_optimal_action`
5. In Phase 4+, set `REWARD_BFS_MATCH: 0.0` — the curriculum is withdrawn and the agent navigates from learned skill alone

**No observation shape change.** Obs remains `(53,)`. All existing Phase 2 checkpoints are compatible.

**What the model still has to learn:**
- The underlying *pattern* that makes certain moves BFS-optimal — it receives the reward but must generalise the rule across all maze layouts
- Pace management: the BFS bonus is navigation-only; survival under acceleration is still purely RL
- When to deviate from the BFS-optimal move (e.g. pace forces a different escape route)
- Long-term survival: pacing, score maximisation, terminal avoidance

**Pros:**
- Agent learns genuine navigation skill — removing the reward at Phase 4 doesn't break the policy
- No obs shape change — compatible with existing checkpoints
- BFS takes < 1 ms on a 15×20 cell grid; negligible training overhead
- +0.05 bonus is large enough to be a meaningful gradient signal (5% of a successful RIGHT move) but small enough not to dominate pace-survival rewards
- Naturally fades: as the agent internalises navigation, the BFS bonus fires more often (correct moves increase) but contributes less *marginal* gradient signal — the curriculum self-withdraws

**Cons:**
- Credit assignment still operates one step at a time — the agent gets +0.05 for each correct step but still doesn't have explicit multi-step lookahead
- BFS path becomes stale on the tick a pace shift moves walls — the optimal action may change mid-corridor; the bonus needs to tolerate 1-step lag
- Two-phase search is slightly more expensive than single-target BFS (~2–3 ms worst case on a full 15×20 grid) — still negligible at 60 fps training
- Forward reachability threshold is a tunable parameter; if set too high it discards valid gaps, if too low it doesn't filter pockets effectively. Start at 2 cells and adjust if needed.

**Reward changes:** Add `REWARD_BFS_MATCH: 0.05` to `ML_CONFIG` in `config.py`. Apply in `compute_reward()` conditioned on `phase >= 3`.

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

## Recommendation: Option 2 — BFS Curriculum Reward

Option 2 teaches genuine navigation skill while keeping the RL problem intact. The key principle: **reward the right move without showing what it is**. The agent must learn the spatial pattern that makes a move BFS-optimal across thousands of diverse maze layouts — that generalisation is real navigational intelligence.

The critical distinction from shortcut learning: if you zero out `REWARD_BFS_MATCH` at the end of Phase 3 and run evaluation, the deterministic score should *not* collapse. If it does, the agent learned to respond to the bonus as a cue rather than internalising the underlying skill. That is the Phase 3 success gate before advancing to Phase 4.

Option 1 (wider grid) is worth combining — more spatial context helps the network recognise navigable corridor patterns faster. It can be enabled at the same time as Option 2 at no algorithm cost, though it requires retraining from scratch.

Option 3 is ruled out by implementation complexity and inference-time performance requirements.

**Implementation order if Phase 3 stalls again:**
1. Add `bfs_optimal_action()` to `features.py`
2. Add `REWARD_BFS_MATCH: 0.05` to `ML_CONFIG` in `config.py`
3. Apply bonus in `rewards.py` `compute_reward()` when `phase >= 3` and action matches BFS optimal
4. Retrain from Phase 2 best with `--phase 3 --ent-coef 0.05 --clip-range 0.1`
5. **Phase 3 exit gate:** zero out `REWARD_BFS_MATCH`, re-evaluate deterministically — mean score must hold at ≥ 250 to confirm genuine skill, not cue-following
6. Phase 4 proceeds with `REWARD_BFS_MATCH: 0.0` permanently

---

*Last updated: 2026-04-01 — Option 2 revised from BFS-as-observation to BFS-as-curriculum-reward to prevent shortcut learning. BFS design updated to two-phase search (candidate gaps → forward reachability filter) to avoid routing the agent into dead-end pockets.*
