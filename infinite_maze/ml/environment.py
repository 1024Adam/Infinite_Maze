"""Gymnasium environment wrapper for the Infinite Maze game.

Headless-safe: never calls pygame.display.set_mode(), pygame.display.flip(),
or any rendering method. Drives the game step-by-step through direct state
manipulation — the live engine loop (core/engine.py) is never imported or used.

Observation space: Box(53,) float32, all values in [0.0, 1.0].
Action space:      Discrete(5) mapping to config.MOVEMENT_CONSTANTS.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..core.game import Game
from ..entities.player import Player
from ..entities.maze import Line
from ..utils.config import config
from .features import (
    get_obs,
    is_blocked_right,
    is_blocked_left,
    is_blocked_up,
    is_blocked_down,
)
from .rewards import compute_reward

_ML = config.ML_CONFIG
_MC = config.MOVEMENT_CONSTANTS

DO_NOTHING = _MC["DO_NOTHING"]
RIGHT      = _MC["RIGHT"]
LEFT       = _MC["LEFT"]
UP         = _MC["UP"]
DOWN       = _MC["DOWN"]


class InfiniteMazeEnv(gym.Env):
    """Gymnasium environment wrapping the Infinite Maze game.

    Parameters
    ----------
    phase : int
        Training phase (1–5). Phase 3 activates path-finding reward shaping.
        All other phases use the base reward function from rewards.py.
    render_mode : None
        Rendering is not supported; value must be None.
    """

    metadata = {"render_modes": []}

    def __init__(self, phase: int = 1, render_mode=None):
        super().__init__()

        self.phase = phase

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(53,), dtype=np.float32
        )

        # Game objects — initialised on first reset()
        self._game   = None
        self._player = None
        self._lines  = None

        # Episode state
        self._tick_counter        = 0
        self._consecutive_blocked = 0

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Start a new episode.

        Parameters
        ----------
        seed : int | None
            RNG seed forwarded to gymnasium's base reset.
        options : dict | None
            Optional overrides:
            - ``"start_pace"`` (int): initial game pace. Used by Phase 4
              curriculum domain randomisation.
        """
        super().reset(seed=seed)

        self._game   = Game(headless=True)
        self._player = Player(
            config.PLAYER_START_X, config.PLAYER_START_Y, headless=True
        )
        self._lines  = Line.generateMaze(
            self._game, config.MAZE_ROWS, config.MAZE_COLS
        )

        self._tick_counter        = 0
        self._consecutive_blocked = 0

        if options is not None and "start_pace" in options:
            self._game.setPace(int(options["start_pace"]))

        obs  = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        """Advance the environment by one tick.

        Parameters
        ----------
        action : int
            Integer in [0, 4] mapping to config.MOVEMENT_CONSTANTS.

        Returns
        -------
        obs : np.ndarray  shape (53,) float32
        reward : float
        terminated : bool  True when player is pushed past X_MIN by pace
        truncated : bool   True when score >= EPISODE_SCORE_CAP
        info : dict        {'score', 'pace', 'tick'}
        """
        game   = self._game
        player = self._player
        lines  = self._lines

        # -- 1. Collision flags (computed before the move) --
        blocked_right = is_blocked_right(player, lines)
        blocked_left  = is_blocked_left(player, lines)
        blocked_up    = is_blocked_up(player, lines)
        blocked_down  = is_blocked_down(player, lines)
        blocked_flags = {
            "right": blocked_right,
            "left":  blocked_left,
            "up":    blocked_up,
            "down":  blocked_down,
        }

        # -- 2. Apply action --
        if action == RIGHT and not blocked_right:
            player.moveX(1)
            game.incrementScore()
        elif action == LEFT and not blocked_left:
            player.moveX(-1)
            game.decrementScore()
        elif action == UP and not blocked_up:
            player.moveY(-1)
        elif action == DOWN and not blocked_down:
            player.moveY(1)
        # DO_NOTHING: no move

        # -- 3. Tick counter --
        self._tick_counter += 1

        # -- 4. Pace shift (every PACE_SHIFT_INTERVAL ticks) --
        if self._tick_counter % _ML["PACE_SHIFT_INTERVAL"] == 0:
            shift = game.getPace()
            player.setX(player.getX() - shift)
            for line in lines:
                line.setXStart(line.getXStart() - shift)
                line.setXEnd(line.getXEnd() - shift)

        # -- 5. Position adjustments (every step, mirrors engine.py) --
        if player.getX() > int(game.X_MAX):
            player.setX(int(game.X_MAX))
            for line in lines:
                line.setXStart(line.getXStart() - player.getSpeed())
                line.setXEnd(line.getXEnd() - player.getSpeed())

        player.setY(max(player.getY(), game.Y_MIN))
        player.setY(min(player.getY(), game.Y_MAX))

        # -- 6. Line recycling (every step) --
        x_max = Line.getXMax(lines)
        for line in lines:
            start = line.getXStart()
            end   = line.getXEnd()
            if start < config.PLAYER_START_X:
                line.setXStart(x_max)
                if start == end:      # vertical line — xStart == xEnd
                    line.setXEnd(x_max)
                else:                 # horizontal line — span one cell
                    line.setXEnd(x_max + config.MAZE_CELL_SIZE)

        # -- 7. Pace update (tick-based, replaces real-time Clock) --
        if self._tick_counter % _ML["TICKS_PER_PACE_UPDATE"] == 0:
            game.setPace(game.getPace() + 1)

        # -- 8 & 9. Terminal / truncated --
        terminated = player.getX() < game.X_MIN
        truncated  = game.getScore() >= _ML["EPISODE_SCORE_CAP"]

        # -- 10. Consecutive-blocked counter update --
        if not blocked_right and action == RIGHT:
            new_blocked = 0
        elif blocked_right:
            new_blocked = self._consecutive_blocked + 1
        else:
            new_blocked = self._consecutive_blocked

        # -- 11. Reward --
        reward = float(compute_reward(action, blocked_flags, terminated, game))

        self._consecutive_blocked = new_blocked

        # -- 12. Build observation and return --
        obs  = self._get_obs()
        info = {
            "score": game.getScore(),
            "pace":  game.getPace(),
            "tick":  self._tick_counter,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # headless — no rendering supported

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return get_obs(
            self._player,
            self._lines,
            self._game,
            self._consecutive_blocked,
        )
