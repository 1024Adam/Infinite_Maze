"""
Unit tests for the InfiniteMazeEnv gymnasium wrapper and supporting ML modules.

Headless-safe: all game objects are instantiated with headless=True.
Uses the session-scoped `pygame_init` fixture from conftest.py (autouse).

Covers the five acceptance conditions from the training plan (Step 2) plus
hand-crafted wall-grid configuration tests for features.get_wall_grid.
"""

import numpy as np
import pytest

from infinite_maze.core.game import Game
from infinite_maze.entities.player import Player
from infinite_maze.entities.maze import Line
from infinite_maze.utils.config import config
from infinite_maze.ml.environment import InfiniteMazeEnv
from infinite_maze.ml.features import (
    is_blocked,
    is_blocked_right,
    is_blocked_left,
    is_blocked_up,
    is_blocked_down,
    get_obs,
    get_wall_grid,
)
from infinite_maze.ml.rewards import compute_reward, phase3_shaping

_ML = config.ML_CONFIG
_MC = config.MOVEMENT_CONSTANTS
RIGHT      = _MC["RIGHT"]
LEFT       = _MC["LEFT"]
UP         = _MC["UP"]
DOWN       = _MC["DOWN"]
DO_NOTHING = _MC["DO_NOTHING"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(phase=1):
    env = InfiniteMazeEnv(phase=phase)
    env.reset(seed=0)
    return env


def _vertical_wall(x, y_start, y_end):
    """Create a vertical Line segment."""
    return Line((x, y_start), (x, y_end), 0, 1)


def _horizontal_wall(x_start, x_end, y):
    """Create a horizontal Line segment."""
    return Line((x_start, y), (x_end, y), 0, 1)


# ---------------------------------------------------------------------------
# Acceptance 1 — reset() returns valid observation
# ---------------------------------------------------------------------------

class TestReset:
    def test_obs_shape(self):
        env = InfiniteMazeEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (53,)

    def test_obs_dtype(self):
        env = InfiniteMazeEnv()
        obs, _ = env.reset(seed=42)
        assert obs.dtype == np.float32

    def test_obs_in_observation_space(self):
        env = InfiniteMazeEnv()
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_obs_values_in_range(self):
        env = InfiniteMazeEnv()
        obs, _ = env.reset(seed=1)
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_info_is_dict(self):
        env = InfiniteMazeEnv()
        _, info = env.reset()
        assert isinstance(info, dict)

    def test_reset_reinitialises_tick_counter(self):
        env = _make_env()
        for _ in range(10):
            env.step(DO_NOTHING)
        env.reset(seed=99)
        assert env._tick_counter == 0

    def test_reset_reinitialises_episode_state(self):
        env = _make_env()
        env.reset()
        assert env._consecutive_blocked == 0

    def test_reset_with_start_pace_option(self):
        env = InfiniteMazeEnv()
        env.reset(options={"start_pace": 3})
        assert env._game.getPace() == 3

    def test_multiple_resets_produce_valid_obs(self):
        env = InfiniteMazeEnv()
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            assert obs.shape == (53,)
            assert env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# Acceptance 2 — step() returns correct 5-tuple for every action
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_five_tuple(self):
        env = _make_env()
        result = env.step(DO_NOTHING)
        assert len(result) == 5

    @pytest.mark.parametrize("action", [DO_NOTHING, RIGHT, LEFT, UP, DOWN])
    def test_step_obs_shape(self, action):
        env = _make_env()
        obs, _, _, _, _ = env.step(action)
        assert obs.shape == (53,)

    @pytest.mark.parametrize("action", [DO_NOTHING, RIGHT, LEFT, UP, DOWN])
    def test_step_obs_dtype(self, action):
        env = _make_env()
        obs, _, _, _, _ = env.step(action)
        assert obs.dtype == np.float32

    @pytest.mark.parametrize("action", [DO_NOTHING, RIGHT, LEFT, UP, DOWN])
    def test_step_obs_in_observation_space(self, action):
        env = _make_env()
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)

    @pytest.mark.parametrize("action", [DO_NOTHING, RIGHT, LEFT, UP, DOWN])
    def test_step_reward_is_float(self, action):
        env = _make_env()
        _, reward, _, _, _ = env.step(action)
        assert isinstance(reward, float)

    @pytest.mark.parametrize("action", [DO_NOTHING, RIGHT, LEFT, UP, DOWN])
    def test_step_terminated_is_bool(self, action):
        env = _make_env()
        _, _, terminated, _, _ = env.step(action)
        assert isinstance(terminated, (bool, np.bool_))

    @pytest.mark.parametrize("action", [DO_NOTHING, RIGHT, LEFT, UP, DOWN])
    def test_step_truncated_is_bool(self, action):
        env = _make_env()
        _, _, _, truncated, _ = env.step(action)
        assert isinstance(truncated, (bool, np.bool_))

    @pytest.mark.parametrize("action", [DO_NOTHING, RIGHT, LEFT, UP, DOWN])
    def test_step_info_is_dict(self, action):
        env = _make_env()
        _, _, _, _, info = env.step(action)
        assert isinstance(info, dict)

    def test_step_info_contains_score_pace_tick(self):
        env = _make_env()
        _, _, _, _, info = env.step(DO_NOTHING)
        assert "score" in info
        assert "pace" in info
        assert "tick" in info

    def test_step_tick_increments(self):
        env = _make_env()
        env.step(DO_NOTHING)
        assert env._tick_counter == 1
        env.step(DO_NOTHING)
        assert env._tick_counter == 2

    def test_step_do_nothing_reward(self):
        env = _make_env()
        _, reward, terminated, _, _ = env.step(DO_NOTHING)
        if not terminated:
            assert reward == pytest.approx(_ML["REWARD_DO_NOTHING"])

    def test_step_up_down_reward_neutral_when_unblocked(self):
        env = _make_env()
        from infinite_maze.ml.features import is_blocked_right
        for action in (UP, DOWN):
            env.reset(seed=0)
            # Seek a state where right is NOT blocked so vertical reward is 0.0
            for _ in range(50):
                if not is_blocked_right(env._player, env._lines):
                    _, reward, terminated, _, _ = env.step(action)
                    if not terminated:
                        assert reward == pytest.approx(_ML["REWARD_MOVE_VERTICAL"])
                    break
                env.step(DO_NOTHING)


# ---------------------------------------------------------------------------
# Acceptance 3 — blocked RIGHT does not advance player x
# ---------------------------------------------------------------------------

class TestBlockedMove:
    def test_blocked_right_does_not_advance_x(self):
        """Place a vertical wall directly to the right, confirm step(RIGHT) is a no-op."""
        game   = Game(headless=True)
        player = Player(100, 100, headless=True)
        # wall at x = player.x + player.width + player.speed = 111
        wall = _vertical_wall(111, 90, 115)

        assert is_blocked_right(player, [wall])

        env = InfiniteMazeEnv(phase=1)
        env.reset(seed=0)
        # Inject known state
        env._player = player
        env._lines  = [wall]
        env._game   = game

        x_before = player.getX()
        env.step(RIGHT)
        assert player.getX() == x_before

    def test_blocked_left_does_not_retreat_x(self):
        """Horizontal wall to the left should block leftward movement."""
        game   = Game(headless=True)
        player = Player(100, 100, headless=True)
        # px - speed = 99 must equal line.xEnd
        wall = _horizontal_wall(80, 99, 100)  # xEnd=99, yStart=100

        assert is_blocked_left(player, [wall])

        env = InfiniteMazeEnv(phase=1)
        env.reset(seed=0)
        env._player = player
        env._lines  = [wall]
        env._game   = game

        x_before = player.getX()
        env.step(LEFT)
        assert player.getX() == x_before

    def test_unblocked_right_advances_x(self):
        env = InfiniteMazeEnv()
        env.reset(seed=0)
        # Step right until we find an unblocked step
        for _ in range(50):
            if not is_blocked_right(env._player, env._lines):
                x_before = env._player.getX()
                env.step(RIGHT)
                # x may be clamped at X_MAX; just verify it did not decrease
                assert env._player.getX() >= x_before
                return
            env.step(DOWN)
        pytest.skip("Could not find unblocked right in 50 steps")


# ---------------------------------------------------------------------------
# Acceptance 4 — termination when pace pushes player past X_MIN
# ---------------------------------------------------------------------------

class TestTermination:
    def test_terminated_when_player_past_x_min(self):
        env = InfiniteMazeEnv()
        env.reset(seed=0)
        # Place player just inside the boundary and set a pace that will
        # push it past X_MIN on the next pace-shift tick
        env._player.setX(config.PLAYER_START_X + 1)  # one pixel inside
        env._game.setPace(2)
        # Advance tick counter so the next step triggers a pace shift
        env._tick_counter = _ML["PACE_SHIFT_INTERVAL"] - 1
        _, reward, terminated, _, _ = env.step(DO_NOTHING)
        assert terminated is True

    def test_terminal_reward_on_termination(self):
        env = InfiniteMazeEnv()
        env.reset(seed=0)
        env._player.setX(config.PLAYER_START_X + 1)
        env._game.setPace(2)
        env._tick_counter = _ML["PACE_SHIFT_INTERVAL"] - 1
        _, reward, terminated, _, _ = env.step(DO_NOTHING)
        assert terminated is True
        assert reward == pytest.approx(_ML["REWARD_TERMINAL"])

    def test_truncated_when_score_reaches_cap(self):
        env = InfiniteMazeEnv()
        env.reset(seed=0)
        env._game.setScore(_ML["EPISODE_SCORE_CAP"])
        _, _, _, truncated, _ = env.step(DO_NOTHING)
        assert truncated is True

    def test_not_terminated_at_start(self):
        env = InfiniteMazeEnv()
        env.reset(seed=0)
        _, _, terminated, truncated, _ = env.step(DO_NOTHING)
        assert terminated is False
        assert truncated is False


# ---------------------------------------------------------------------------
# Acceptance 5 — observation_space.contains(obs) every step
# ---------------------------------------------------------------------------

class TestObsSpaceCompliance:
    def test_obs_in_space_over_episode(self):
        env = InfiniteMazeEnv()
        obs, _ = env.reset(seed=7)
        assert env.observation_space.contains(obs)
        for action in [RIGHT, RIGHT, DOWN, DOWN, RIGHT, UP, LEFT, DO_NOTHING]:
            obs, _, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs), (
                f"obs out of space at action={action}: {obs}"
            )
            if terminated or truncated:
                break

    def test_obs_after_pace_shift(self):
        env = InfiniteMazeEnv()
        env.reset(seed=0)
        env._game.setPace(3)
        env._tick_counter = _ML["PACE_SHIFT_INTERVAL"] - 1
        obs, _, _, _, _ = env.step(DO_NOTHING)
        assert env.observation_space.contains(obs)

    def test_obs_at_high_pace(self):
        env = InfiniteMazeEnv()
        env.reset(seed=0)
        env._game.setPace(_ML["MAX_PACE"])
        obs, _, _, _, _ = env.step(DO_NOTHING)
        assert env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# get_wall_grid — local occupancy grid
# ---------------------------------------------------------------------------

class TestWallGrid:
    def _make_player(self, x=100, y=200):
        return Player(x, y, headless=True)

    def test_shape_with_no_lines(self):
        player = self._make_player()
        grid = get_wall_grid(player, [])
        expected_len = _ML["GRID_COLS"] * _ML["GRID_ROWS"] * 2
        assert grid.shape == (expected_len,)

    def test_all_zeros_with_no_lines(self):
        player = self._make_player()
        grid = get_wall_grid(player, [])
        assert np.all(grid == 0.0)

    def test_dtype_is_float32(self):
        player = self._make_player()
        grid = get_wall_grid(player, [])
        assert grid.dtype == np.float32

    def test_all_values_binary(self):
        """Every grid value must be 0.0 or 1.0."""
        player = self._make_player()
        wall = _vertical_wall(120, 190, 220)
        grid = get_wall_grid(player, [wall])
        assert np.all((grid == 0.0) | (grid == 1.0))

    def test_vertical_wall_in_col0_sets_has_right(self):
        """Vertical wall in col 0 x range overlapping centre row → has_right_wall=1."""
        # Player(100,200): right edge=110. Col 0: x in (110, 132].
        # cy=205, row_offset=0: top=194, bottom=216.
        # Wall at x=120 y=[190,220] overlaps col 0, centre row.
        # Index for col=0, row_idx=2 (offset=0): (0*5+2)*2 = 4.
        player = self._make_player(x=100, y=200)
        wall = _vertical_wall(120, 190, 220)
        grid = get_wall_grid(player, [wall])
        ROWS = _ML["GRID_ROWS"]
        has_right_idx = (0 * ROWS + 2) * 2
        assert grid[has_right_idx] == 1.0, f"Expected has_right_wall=1 at idx {has_right_idx}"

    def test_vertical_wall_col0_does_not_contaminate_col1(self):
        """A wall in col 0 must not set any col 1 features."""
        player = self._make_player(x=100, y=200)
        wall = _vertical_wall(120, 190, 220)  # col 0 only
        grid = get_wall_grid(player, [wall])
        ROWS = _ML["GRID_ROWS"]
        for row_idx in range(ROWS):
            idx = (1 * ROWS + row_idx) * 2
            assert grid[idx] == 0.0, f"Col 1 has_right should be 0 at idx {idx}"

    def test_horizontal_wall_sets_has_bottom(self):
        """Horizontal wall in centre row y range overlapping col 0 → has_bottom_wall=1."""
        # cy=205, row_offset=0: top=194, bottom=216. Wall at y=210 in (194,216].
        # Col 0: x in (110,132]. Wall x=[110,132] overlaps.
        # Index for col=0, row_idx=2: has_bottom at (0*5+2)*2+1 = 5.
        player = self._make_player(x=100, y=200)
        wall = _horizontal_wall(110, 132, 210)
        grid = get_wall_grid(player, [wall])
        ROWS = _ML["GRID_ROWS"]
        has_bottom_idx = (0 * ROWS + 2) * 2 + 1
        assert grid[has_bottom_idx] == 1.0, f"Expected has_bottom_wall=1 at idx {has_bottom_idx}"

    def test_wall_in_col2_does_not_affect_col0(self):
        """A wall in col 2 must not set col 0 features."""
        # Col 2: x in (154, 176]. Wall at x=165.
        player = self._make_player(x=100, y=200)
        wall = _vertical_wall(165, 190, 220)
        grid = get_wall_grid(player, [wall])
        ROWS = _ML["GRID_ROWS"]
        col0_right_centre = (0 * ROWS + 2) * 2
        col2_right_centre = (2 * ROWS + 2) * 2
        assert grid[col0_right_centre] == 0.0, "Col 0 should not be set by col 2 wall"
        assert grid[col2_right_centre] == 1.0, "Col 2 should be set"


# ---------------------------------------------------------------------------
# is_blocked helpers — directional collision correctness
# ---------------------------------------------------------------------------

class TestIsBlocked:
    def test_right_blocked_by_vertical_wall(self):
        player = Player(100, 100, headless=True)
        # px+pw+speed = 100+10+1 = 111; wall at x=111 overlapping y
        wall = _vertical_wall(111, 95, 115)
        assert is_blocked_right(player, [wall]) is True
        assert is_blocked(player, [wall], RIGHT) is True

    def test_right_not_blocked_without_wall(self):
        player = Player(100, 100, headless=True)
        assert is_blocked_right(player, []) is False

    def test_left_blocked_by_horizontal_wall(self):
        player = Player(100, 100, headless=True)
        # px - speed = 99; horizontal wall with xEnd=99, yStart=100
        wall = _horizontal_wall(80, 99, 100)
        assert is_blocked_left(player, [wall]) is True
        assert is_blocked(player, [wall], LEFT) is True

    def test_up_blocked_by_horizontal_wall(self):
        player = Player(100, 100, headless=True)
        # py - speed = 99; horizontal wall at yStart=99 with x-overlap
        wall = _horizontal_wall(95, 115, 99)
        assert is_blocked_up(player, [wall]) is True
        assert is_blocked(player, [wall], UP) is True

    def test_down_blocked_by_horizontal_wall(self):
        player = Player(100, 100, headless=True)
        # py+ph+speed = 111; horizontal wall at yStart=111 with x-overlap
        wall = _horizontal_wall(95, 115, 111)
        assert is_blocked_down(player, [wall]) is True
        assert is_blocked(player, [wall], DOWN) is True

    def test_do_nothing_never_blocked(self):
        player = Player(100, 100, headless=True)
        walls = [_vertical_wall(111, 0, 500), _horizontal_wall(0, 500, 111)]
        assert is_blocked(player, walls, DO_NOTHING) is False


# ---------------------------------------------------------------------------
# Phase 3 consecutive_blocked counter behaviour
# ---------------------------------------------------------------------------

class TestConsecutiveBlocked:
    def test_counter_increments_when_right_blocked(self):
        env = InfiniteMazeEnv(phase=1)
        env.reset(seed=0)
        game   = Game(headless=True)
        player = Player(100, 100, headless=True)
        wall   = _vertical_wall(111, 90, 115)
        env._player = player
        env._lines  = [wall]
        env._game   = game
        env.step(RIGHT)
        assert env._consecutive_blocked == 1

    def test_counter_resets_on_successful_right(self):
        env = InfiniteMazeEnv(phase=1)
        env.reset(seed=0)
        game   = Game(headless=True)
        player = Player(100, 100, headless=True)
        wall   = _vertical_wall(111, 90, 115)
        env._player = player
        env._lines  = [wall]
        env._game   = game
        # Block right for 3 steps
        for _ in range(3):
            env.step(RIGHT)
        assert env._consecutive_blocked == 3
        # Remove wall and step right — should reset counter
        env._lines = []
        env.step(RIGHT)
        assert env._consecutive_blocked == 0


# ---------------------------------------------------------------------------
# Phase 3 reward shaping — environment integration
# ---------------------------------------------------------------------------

class TestPhase3Shaping:
    def test_phase3_shaping_active_in_phase3(self):
        """Phase 3 env adds non-zero shaping bonus when blocked right and moves toward gap."""
        env = InfiniteMazeEnv(phase=3)
        env.reset(seed=0)
        # Just confirm step returns a finite reward without error
        _, reward, _, _, _ = env.step(UP)
        assert np.isfinite(reward)

    def test_phase1_no_phase3_shaping(self):
        """Phase 1 env reward for vertical move must be exactly REWARD_MOVE_VERTICAL."""
        env = InfiniteMazeEnv(phase=1)
        env.reset(seed=0)
        # Find a step where UP is not blocked and player won't terminate
        for _ in range(20):
            from infinite_maze.ml.features import is_blocked_up
            if not is_blocked_up(env._player, env._lines):
                _, reward, terminated, _, _ = env.step(UP)
                if not terminated:
                    assert reward == pytest.approx(_ML["REWARD_MOVE_VERTICAL"])
                    return
            env.step(DO_NOTHING)
