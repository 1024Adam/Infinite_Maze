"""Feature extraction for the Infinite Maze RL environment.

Provides headless-safe observation encoding, AABB collision checks
(replicated from core/engine.py — do not import from engine.py), wall distance
scanning, and the nearest-right-gap feature used in Phase 3 reward shaping.

All public functions are pure (stateless). Episode-level state such as
``consecutive_blocked`` is owned by the environment and passed in as a
parameter.
"""

import numpy as np

from ..utils.config import config

_MC = config.MOVEMENT_CONSTANTS
DO_NOTHING = _MC["DO_NOTHING"]
RIGHT      = _MC["RIGHT"]
LEFT       = _MC["LEFT"]
UP         = _MC["UP"]
DOWN       = _MC["DOWN"]

_ML = config.ML_CONFIG


# ---------------------------------------------------------------------------
# AABB collision checks — replicated from core/engine.py, no import
# ---------------------------------------------------------------------------

def is_blocked_right(player, lines) -> bool:
    """Return True if the player cannot move right by one speed unit."""
    px    = player.getX()
    py    = player.getY()
    pw    = player.getWidth()
    ph    = player.getHeight()
    speed = player.getSpeed()

    for line in lines:
        if line.getIsHorizontal():
            if (
                py <= line.getYStart()
                and py + ph >= line.getYStart()
                and px + pw + speed == line.getXStart()
            ):
                return True
        else:
            if (
                px + pw <= line.getXStart()
                and px + pw + speed >= line.getXStart()
                and (
                    (py >= line.getYStart() and py      <= line.getYEnd())
                    or (py + ph >= line.getYStart() and py + ph <= line.getYEnd())
                )
            ):
                return True
    return False


def is_blocked_left(player, lines) -> bool:
    """Return True if the player cannot move left by one speed unit."""
    px    = player.getX()
    py    = player.getY()
    ph    = player.getHeight()
    speed = player.getSpeed()

    for line in lines:
        if line.getIsHorizontal():
            if (
                py <= line.getYStart()
                and py + ph >= line.getYStart()
                and px - speed == line.getXEnd()
            ):
                return True
        else:
            if (
                px >= line.getXEnd()
                and px - speed <= line.getXEnd()
                and (
                    (py >= line.getYStart() and py      <= line.getYEnd())
                    or (py + ph >= line.getYStart() and py + ph <= line.getYEnd())
                )
            ):
                return True
    return False


def is_blocked_up(player, lines) -> bool:
    """Return True if the player cannot move up by one speed unit."""
    px    = player.getX()
    py    = player.getY()
    pw    = player.getWidth()
    speed = player.getSpeed()

    for line in lines:
        if line.getIsHorizontal():
            if (
                py >= line.getYStart()
                and py - speed <= line.getYStart()
                and (
                    (px >= line.getXStart() and px      <= line.getXEnd())
                    or (px + pw >= line.getXStart() and px + pw <= line.getXEnd())
                )
            ):
                return True
        else:
            if (
                px <= line.getXStart()
                and px + pw >= line.getXStart()
                and py - speed == line.getYEnd()
            ):
                return True
    return False


def is_blocked_down(player, lines) -> bool:
    """Return True if the player cannot move down by one speed unit."""
    px    = player.getX()
    py    = player.getY()
    pw    = player.getWidth()
    ph    = player.getHeight()
    speed = player.getSpeed()

    for line in lines:
        if line.getIsHorizontal():
            if (
                py + ph <= line.getYStart()
                and py + ph + speed >= line.getYStart()
                and (
                    (px >= line.getXStart() and px      <= line.getXEnd())
                    or (px + pw >= line.getXStart() and px + pw <= line.getXEnd())
                )
            ):
                return True
        else:
            if (
                px <= line.getXStart()
                and px + pw >= line.getXStart()
                and py + ph + speed == line.getYStart()
            ):
                return True
    return False


def is_blocked(player, lines, direction: int) -> bool:
    """Return True if the player is blocked in the given direction.

    Parameters
    ----------
    direction : int
        One of the ``config.MOVEMENT_CONSTANTS`` values (RIGHT, LEFT, UP, DOWN).
        DO_NOTHING always returns False.
    """
    if direction == RIGHT:
        return is_blocked_right(player, lines)
    if direction == LEFT:
        return is_blocked_left(player, lines)
    if direction == UP:
        return is_blocked_up(player, lines)
    if direction == DOWN:
        return is_blocked_down(player, lines)
    return False


# ---------------------------------------------------------------------------
# Wall distance scanning
# ---------------------------------------------------------------------------

def _wall_dist_right(player, lines) -> float:
    """Raw pixel distance to the nearest right-blocking wall, capped at MAX_WALL_SCAN_DIST."""
    px   = player.getX()
    py   = player.getY()
    pw   = player.getWidth()
    ph   = player.getHeight()
    edge = px + pw
    min_dist = float(_ML["MAX_WALL_SCAN_DIST"])

    for line in lines:
        dist = line.getXStart() - edge
        if dist < 0:
            continue
        if line.getIsHorizontal():
            if py <= line.getYStart() <= py + ph:
                min_dist = min(min_dist, dist)
        else:
            ys = line.getYStart()
            ye = line.getYEnd()
            if (
                (py >= ys and py <= ye)
                or (py + ph >= ys and py + ph <= ye)
            ):
                min_dist = min(min_dist, dist)
    return min_dist


def _wall_dist_left(player, lines) -> float:
    """Raw pixel distance to the nearest left-blocking wall, capped at MAX_WALL_SCAN_DIST."""
    px = player.getX()
    py = player.getY()
    ph = player.getHeight()
    min_dist = float(_ML["MAX_WALL_SCAN_DIST"])

    for line in lines:
        dist = px - line.getXEnd()
        if dist < 0:
            continue
        if line.getIsHorizontal():
            if py <= line.getYStart() <= py + ph:
                min_dist = min(min_dist, dist)
        else:
            ys = line.getYStart()
            ye = line.getYEnd()
            if (
                (py >= ys and py <= ye)
                or (py + ph >= ys and py + ph <= ye)
            ):
                min_dist = min(min_dist, dist)
    return min_dist


def _wall_dist_down(player, lines) -> float:
    """Raw pixel distance to the nearest downward-blocking wall, capped at MAX_WALL_SCAN_DIST."""
    px   = player.getX()
    py   = player.getY()
    pw   = player.getWidth()
    ph   = player.getHeight()
    edge = py + ph
    min_dist = float(_ML["MAX_WALL_SCAN_DIST"])

    for line in lines:
        if line.getIsHorizontal():
            dist = line.getYStart() - edge
            if dist < 0:
                continue
            xs = line.getXStart()
            xe = line.getXEnd()
            if (
                (px >= xs and px <= xe)
                or (px + pw >= xs and px + pw <= xe)
            ):
                min_dist = min(min_dist, dist)
        else:
            dist = line.getYStart() - edge
            if dist < 0:
                continue
            xs = line.getXStart()
            if px <= xs <= px + pw:
                min_dist = min(min_dist, dist)
    return min_dist


def _wall_dist_up(player, lines) -> float:
    """Raw pixel distance to the nearest upward-blocking wall, capped at MAX_WALL_SCAN_DIST."""
    px = player.getX()
    py = player.getY()
    pw = player.getWidth()
    min_dist = float(_ML["MAX_WALL_SCAN_DIST"])

    for line in lines:
        if line.getIsHorizontal():
            dist = py - line.getYStart()
            if dist < 0:
                continue
            xs = line.getXStart()
            xe = line.getXEnd()
            if (
                (px >= xs and px <= xe)
                or (px + pw >= xs and px + pw <= xe)
            ):
                min_dist = min(min_dist, dist)
        else:
            dist = py - line.getYEnd()
            if dist < 0:
                continue
            xs = line.getXStart()
            if px <= xs <= px + pw:
                min_dist = min(min_dist, dist)
    return min_dist


# ---------------------------------------------------------------------------
# Nearest right-gap feature
# ---------------------------------------------------------------------------

def _is_blocked_right_at_y(player, lines, y_candidate: float) -> bool:
    """RIGHT collision check with *y_candidate* substituted for player.getY().

    Player x, width, height, and speed are unchanged; only the y value is
    replaced. Used exclusively by ``nearest_right_gap_offset``.
    """
    px    = player.getX()
    pw    = player.getWidth()
    ph    = player.getHeight()
    speed = player.getSpeed()

    for line in lines:
        if line.getIsHorizontal():
            if (
                y_candidate <= line.getYStart()
                and y_candidate + ph >= line.getYStart()
                and int(px + pw + speed) == line.getXStart()
            ):
                return True
        else:
            if (
                px + pw <= line.getXStart()
                and px + pw + speed >= line.getXStart()
                and (
                    (y_candidate      >= line.getYStart() and y_candidate      <= line.getYEnd())
                    or (y_candidate + ph >= line.getYStart() and y_candidate + ph <= line.getYEnd())
                )
            ):
                return True
    return False


def nearest_right_gap_offset(player, lines, game) -> float:
    """Return a normalised [0.0, 1.0] offset to the nearest vertically-reachable
    right-facing gap relative to the player's current Y.

    Scan vertically within ``GAP_SCAN_RADIUS`` pixels at 5-px increments,
    alternating above and below, closest candidate first.

    Returns
    -------
    float
        0.5  → no gap found within scan radius, or gap is at the same Y.
        <0.5 → nearest gap is above the player.
        >0.5 → nearest gap is below the player.
    """
    radius = _ML["GAP_SCAN_RADIUS"]
    y_min  = game.Y_MIN
    y_max  = game.Y_MAX - player.getHeight()
    py     = player.getY()

    for offset in range(0, radius + 1, 5):
        if offset == 0:
            if y_min <= py <= y_max and not _is_blocked_right_at_y(player, lines, py):
                return 0.5
        else:
            y_above = py - offset
            y_below = py + offset
            if y_min <= y_above <= y_max and not _is_blocked_right_at_y(player, lines, y_above):
                raw = (y_above - py) / radius * 0.5 + 0.5
                return float(np.clip(raw, 0.0, 1.0))
            if y_min <= y_below <= y_max and not _is_blocked_right_at_y(player, lines, y_below):
                raw = (y_below - py) / radius * 0.5 + 0.5
                return float(np.clip(raw, 0.0, 1.0))

    return 0.5


# ---------------------------------------------------------------------------
# Main observation encoder
# ---------------------------------------------------------------------------

def get_obs(player, lines, game, consecutive_blocked: int = 0) -> np.ndarray:
    """Encode the current game state as a flat float32 array of shape (14,).

    All values are normalised to [0.0, 1.0].

    Parameters
    ----------
    player : Player
    lines : list[Line]
    game : Game
    consecutive_blocked : int
        Number of consecutive ticks the RIGHT action has been blocked.
        Tracked by the environment and passed in here; this module is stateless.
    """
    ml       = _ML
    max_scan = ml["MAX_WALL_SCAN_DIST"]
    x_range  = int(game.X_MAX) - game.X_MIN  # int() — X_MAX is WIDTH/2 (float)

    obs = np.array(
        [
            # [0]  player x
            np.clip(player.getX() / config.SCREEN_WIDTH, 0.0, 1.0),
            # [1]  player y
            np.clip(player.getY() / config.SCREEN_HEIGHT, 0.0, 1.0),
            # [2]  blocked right
            float(is_blocked_right(player, lines)),
            # [3]  blocked left
            float(is_blocked_left(player, lines)),
            # [4]  blocked up
            float(is_blocked_up(player, lines)),
            # [5]  blocked down
            float(is_blocked_down(player, lines)),
            # [6]  wall distance right
            np.clip(_wall_dist_right(player, lines) / max_scan, 0.0, 1.0),
            # [7]  wall distance left
            np.clip(_wall_dist_left(player, lines) / max_scan, 0.0, 1.0),
            # [8]  wall distance up
            np.clip(_wall_dist_up(player, lines) / max_scan, 0.0, 1.0),
            # [9]  wall distance down
            np.clip(_wall_dist_down(player, lines) / max_scan, 0.0, 1.0),
            # [10] pace
            np.clip(game.getPace() / ml["MAX_PACE"], 0.0, 1.0),
            # [11] distance from death boundary
            np.clip((player.getX() - game.X_MIN) / x_range, 0.0, 1.0),
            # [12] nearest right-gap vertical offset
            nearest_right_gap_offset(player, lines, game),
            # [13] consecutive ticks blocked right (normalised)
            np.clip(consecutive_blocked / ml["CONSECUTIVE_BLOCKED_CAP"], 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return obs
