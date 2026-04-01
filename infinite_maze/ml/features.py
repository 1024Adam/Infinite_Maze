"""Feature extraction for the Infinite Maze RL environment.

Provides headless-safe observation encoding, AABB collision checks
(replicated from core/engine.py — do not import from engine.py), wall distance
scanning, and a local wall-occupancy grid for maze lookahead.

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


def is_blocked_right_at_y(player, lines, y_candidate: float) -> bool:
    """RIGHT collision check with y_candidate substituted for player.getY().

    Keeps player.getX(), player.getWidth(), and all line positions fixed.
    Used by nearest_right_gap_offset to probe candidate Y positions.
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
                and px + pw + speed == line.getXStart()
            ):
                return True
        else:
            if (
                px + pw <= line.getXStart()
                and px + pw + speed >= line.getXStart()
                and (
                    (y_candidate >= line.getYStart() and y_candidate      <= line.getYEnd())
                    or (y_candidate + ph >= line.getYStart() and y_candidate + ph <= line.getYEnd())
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
# Nearest right-facing gap (Phase 3 shaping feature)
# ---------------------------------------------------------------------------

def nearest_right_gap_offset(player, lines, game) -> float:
    """Return the normalised vertical offset to the nearest Y where RIGHT is unblocked.

    Scans outward from the player's current Y in 5-pixel increments up to
    ``ML_CONFIG["GAP_SCAN_RADIUS"]`` pixels, alternating above and below.

    Returns
    -------
    float in [0.0, 1.0]
        0.5 — gap is at the current Y (or no gap found within radius).
        < 0.5 — nearest gap is above the player.
        > 0.5 — nearest gap is below the player.
    """
    py     = player.getY()
    ph     = player.getHeight()
    radius = _ML["GAP_SCAN_RADIUS"]
    y_min  = game.Y_MIN
    y_max  = game.Y_MAX - ph

    # Current Y already unblocked — gap is here
    if not is_blocked_right_at_y(player, lines, py):
        return 0.5

    # Scan outward, alternating up/down, in 5-pixel steps
    for dist in range(5, radius + 1, 5):
        for y_candidate in (py - dist, py + dist):
            if y_candidate < y_min or y_candidate > y_max:
                continue
            if not is_blocked_right_at_y(player, lines, y_candidate):
                offset = (y_candidate - py) / radius * 0.5 + 0.5
                return float(np.clip(offset, 0.0, 1.0))

    return 0.5  # no gap found within scan radius


# ---------------------------------------------------------------------------
# Local wall-occupancy grid
# ---------------------------------------------------------------------------

def get_wall_grid(player, lines) -> np.ndarray:
    """Return a flat binary float32 array encoding wall presence in a local grid.

    The grid covers ``GRID_COLS`` columns × ``GRID_ROWS`` rows centred on the
    player, scanning rightward from the player's right edge. For each cell:

      feature 0 (has_right_wall)  — 1.0 if any vertical wall exists within
                                    this cell's x extent that overlaps the
                                    cell's y range; 0.0 otherwise.
      feature 1 (has_bottom_wall) — 1.0 if any horizontal wall exists within
                                    this cell's y extent that overlaps the
                                    cell's x range; 0.0 otherwise.

    Cells are ordered col 0…COLS-1; within each col, row offsets −HALF…+HALF.

    Shape: (GRID_COLS * GRID_ROWS * 2,) — 40 features with defaults COLS=4, ROWS=5.

    Parameters
    ----------
    player : Player
    lines  : list[Line]
    """
    CELL      = config.MAZE_CELL_SIZE
    COLS      = _ML["GRID_COLS"]
    ROWS      = _ML["GRID_ROWS"]
    half_rows = ROWS // 2

    px = player.getX()
    py = player.getY()
    pw = player.getWidth()
    ph = player.getHeight()
    cy = py + ph // 2   # player centre y

    features = np.zeros(COLS * ROWS * 2, dtype=np.float32)
    idx = 0

    for col in range(COLS):
        left_edge  = px + pw + col * CELL
        right_edge = px + pw + (col + 1) * CELL

        for row_offset in range(-half_rows, half_rows + 1):
            top    = cy + row_offset * CELL - CELL // 2
            bottom = cy + row_offset * CELL + CELL // 2

            # has_right_wall: vertical wall with x in (left_edge, right_edge]
            #                 whose y-span overlaps [top, bottom]
            has_right = 0.0
            for line in lines:
                if not line.getIsHorizontal():
                    lx = line.getXStart()
                    if left_edge < lx <= right_edge:
                        if line.getYStart() <= bottom and line.getYEnd() >= top:
                            has_right = 1.0
                            break

            # has_bottom_wall: horizontal wall with y in (top, bottom]
            #                  whose x-span overlaps [left_edge, right_edge]
            has_bottom = 0.0
            for line in lines:
                if line.getIsHorizontal():
                    ly = line.getYStart()
                    if top < ly <= bottom:
                        if line.getXStart() <= right_edge and line.getXEnd() >= left_edge:
                            has_bottom = 1.0
                            break

            features[idx]     = has_right
            features[idx + 1] = has_bottom
            idx += 2

    return features


# ---------------------------------------------------------------------------
# Main observation encoder
# ---------------------------------------------------------------------------

def get_obs(player, lines, game, consecutive_blocked: int = 0) -> np.ndarray:
    """Encode the current game state as a flat float32 array of shape (53,).

    All values are normalised to [0.0, 1.0].

    Parameters
    ----------
    player : Player
    lines : list[Line]
    game : Game
    consecutive_blocked : int
        Number of consecutive ticks the RIGHT action has been blocked.
        Tracked by the environment and passed in here; this module is stateless.

    Layout
    ------
    [0]     player x (normalised)
    [1]     player y (normalised)
    [2]     blocked right
    [3]     blocked left
    [4]     blocked up
    [5]     blocked down
    [6]     wall distance right (normalised)
    [7]     wall distance left (normalised)
    [8]     wall distance up (normalised)
    [9]     wall distance down (normalised)
    [10]    pace (normalised)
    [11]    distance from death boundary (normalised)
    [12]    consecutive ticks blocked right (normalised)
    [13..52] local wall grid — GRID_COLS × GRID_ROWS × 2 binary features
    """
    ml       = _ML
    max_scan = ml["MAX_WALL_SCAN_DIST"]
    x_range  = int(game.X_MAX) - game.X_MIN  # int() — X_MAX is WIDTH/2 (float)

    scalars = np.array(
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
            # [12] consecutive ticks blocked right (normalised)
            np.clip(consecutive_blocked / ml["CONSECUTIVE_BLOCKED_CAP"], 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([scalars, get_wall_grid(player, lines)])
