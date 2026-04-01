"""Reward shaping for the Infinite Maze RL environment.

All functions are pure (stateless). Episode-level state such as
``prev_gap_offset`` and ``consecutive_blocked`` is owned by the environment and
passed in as arguments.

Reward weights are sourced exclusively from ``config.ML_CONFIG``; nothing is
hardcoded in this module.
"""

from ..utils.config import config

_MC = config.MOVEMENT_CONSTANTS
DO_NOTHING = _MC["DO_NOTHING"]
RIGHT      = _MC["RIGHT"]
LEFT       = _MC["LEFT"]
UP         = _MC["UP"]
DOWN       = _MC["DOWN"]

_ML = config.ML_CONFIG


def compute_reward(
    action: int,
    blocked_flags: dict,
    terminated: bool,
    game,
    bfs_action: int = -1,
    phase: int = 1,
) -> float:
    """Return the shaped reward for one environment step.

    Terminal state takes priority over all per-action rewards.

    Parameters
    ----------
    action : int
        The action taken this step (one of DO_NOTHING / RIGHT / LEFT / UP / DOWN).
    blocked_flags : dict
        Keys ``"right"``, ``"left"``, ``"up"``, ``"down"`` mapping to bools.
        Computed by ``features.is_blocked_*`` before the move is applied.
    terminated : bool
        True when ``player.getX() < game.X_MIN``.
    game : Game
        Live game instance; used for pace in Phase 4 urgency scaling.
        Not used by base rewards — kept in signature for consistency.
    bfs_action : int
        BFS-optimal action for this step, computed by ``features.bfs_optimal_action()``.
        Pass -1 (default) to skip the BFS curriculum bonus.
    phase : int
        Current training phase. BFS bonus only applies for phase >= 3.

    Returns
    -------
    float
        Shaped reward value.

    Cases
    -----
    terminal                 → REWARD_TERMINAL        (-10.0)
    RIGHT + not blocked      → REWARD_MOVE_RIGHT      (+1.0)
    RIGHT + blocked          → REWARD_MOVE_RIGHT_BLOCKED (-0.3)
    LEFT  + not blocked      → REWARD_MOVE_LEFT       (-1.5)
    LEFT  + blocked          → REWARD_MOVE_RIGHT_BLOCKED (-0.3)  # symmetric
    DO_NOTHING               → REWARD_DO_NOTHING      (-0.1)
    UP (free)                → REWARD_MOVE_VERTICAL_UP          (0.0)
    DOWN (free)              → REWARD_MOVE_VERTICAL_DOWN         (0.0)
    UP (blocked right)       → REWARD_VERTICAL_WHEN_BLOCKED_UP   (+0.1)  + phase3_shaping gap bonus
    DOWN (blocked right)     → REWARD_VERTICAL_WHEN_BLOCKED_DOWN (+0.1)  + phase3_shaping gap bonus
    + BFS match (phase >= 3) → +REWARD_BFS_MATCH (+0.05) when action == bfs_action
    """
    if terminated:
        return float(_ML["REWARD_TERMINAL"])

    if action == RIGHT:
        if blocked_flags.get("right", False):
            reward = float(_ML["REWARD_MOVE_RIGHT_BLOCKED"])
        else:
            reward = float(_ML["REWARD_MOVE_RIGHT"])

    elif action == LEFT:
        if blocked_flags.get("left", False):
            reward = float(_ML["REWARD_MOVE_RIGHT_BLOCKED"])
        else:
            reward = float(_ML["REWARD_MOVE_LEFT"])

    elif action == DO_NOTHING:
        reward = float(_ML["REWARD_DO_NOTHING"])

    else:
        # UP or DOWN — differentiated bonus to break DOWN-only bias
        if blocked_flags.get("right", False):
            key = "REWARD_VERTICAL_WHEN_BLOCKED_UP" if action == UP else "REWARD_VERTICAL_WHEN_BLOCKED_DOWN"
        else:
            key = "REWARD_MOVE_VERTICAL_UP" if action == UP else "REWARD_MOVE_VERTICAL_DOWN"
        reward = float(_ML[key])

    # BFS curriculum bonus (Phase 3+): small bonus when action matches BFS-optimal move
    if phase >= 3 and bfs_action >= 0 and action == bfs_action:
        reward += float(_ML["REWARD_BFS_MATCH"])

    return reward


def phase3_shaping(
    action: int,
    prev_gap_offset: float,
    new_gap_offset: float,
    prev_blocked_count: int,
    new_blocked_count: int,
    blocked_right: bool,
) -> float:
    """Return the Phase 3 directional path-finding bonus.

    This function is **Phase 3 only** and must not be called in Phases 4+.
    The environment enables/disables the call based on the active phase flag.

    Two bonuses (non-exclusive, additive):

    1. **Gap-approach bonus** (+0.3): fires when ``blocked_right=True``,
       the action is vertical (UP or DOWN), and the new gap offset is
       *closer to 0.5* than the previous one (i.e. the agent moved toward
       the nearest right-facing gap).

    2. **Escape bonus** (+0.05): fires whenever ``new_blocked_count`` is
       strictly less than ``prev_blocked_count``, rewarding the agent for
       breaking out of a stuck-right state.

    Parameters
    ----------
    action : int
    prev_gap_offset : float  Gap offset at the start of this step (from the previous obs).
    new_gap_offset  : float  Gap offset at the end of this step (from the new obs).
    prev_blocked_count : int  ``consecutive_blocked`` before this step.
    new_blocked_count  : int  ``consecutive_blocked`` after this step.
    blocked_right : bool  Whether RIGHT was blocked before the move.
    """
    bonus = 0.0

    # Gap-approach bonus
    if (
        blocked_right
        and action in (UP, DOWN)
        and abs(new_gap_offset - 0.5) < abs(prev_gap_offset - 0.5)
    ):
        bonus += 0.3

    # Escape bonus — agent reduced its stuck counter this step
    if new_blocked_count < prev_blocked_count:
        bonus += 0.05

    return bonus


def urgency_scaled_terminal(game) -> float:
    """Return the Phase 4 terminal penalty, scaled by current pace.

    Higher pace → heavier penalty, encouraging the agent to reach high
    scores before pace climbs into the unwinnable range.

    Returns ``REWARD_TERMINAL * (1 + pace / MAX_PACE)``.
    """
    scale = 1.0 + game.getPace() / float(_ML["MAX_PACE"])
    return float(_ML["REWARD_TERMINAL"]) * scale
