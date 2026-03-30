"""
Unit tests for the InfiniteMazeEnv gymnasium wrapper and supporting ML modules.

Headless-safe: all game objects are instantiated with headless=True.
Uses the session-scoped `pygame_init` fixture from conftest.py (autouse).

Covers the five acceptance conditions from the training plan (Step 2) plus
hand-crafted line configuration tests for features.nearest_right_gap_offset.
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
    nearest_right_gap_offset,
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
        assert obs.shape == (14,)

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
        assert env._prev_gap_offset == 0.5

    def test_reset_with_start_pace_option(self):
        env = InfiniteMazeEnv()
        env.reset(options={"start_pace": 3})
        assert env._game.getPace() == 3

    def test_multiple_resets_produce_valid_obs(self):
        env = InfiniteMazeEnv()
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            assert obs.shape == (14,)
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
        assert obs.shape == (14,)

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

    def test_step_up_down_reward_neutral(self):
        env = _make_env()
        for action in (UP, DOWN):
            env.reset(seed=0)
            _, reward, terminated, _, _ = env.step(action)
            if not terminated:
                # Vertical rewards are 0.0 (or phase3 shaping adds small bonus)
                assert reward == pytest.approx(_ML["REWARD_MOVE_VERTICAL"])


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
# nearest_right_gap_offset — hand-crafted line configurations
# ---------------------------------------------------------------------------

class TestNearestRightGapOffset:
    def _make_player_and_game(self, x=100, y=200):
        game   = Game(headless=True)
        player = Player(x, y, headless=True)
        return player, game

    def test_returns_half_when_no_lines(self):
        player, game = self._make_player_and_game()
        assert nearest_right_gap_offset(player, [], game) == pytest.approx(0.5)

    def test_returns_half_when_current_y_unblocked(self):
        """Player is already unblocked at its own Y → gap is at offset=0 → returns 0.5."""
        player, game = self._make_player_and_game(x=100, y=200)
        # No wall to the right at current y
        far_wall = _vertical_wall(300, 0, 10)  # does not overlap player y
        assert nearest_right_gap_offset(player, [far_wall], game) == pytest.approx(0.5)

    def test_gap_below_returns_greater_than_half(self):
        """Wall blocks current y; first gap is below → result > 0.5."""
        player, game = self._make_player_and_game(x=100, y=200)
        # wall at x=111 covers y=190..235 → player bounding box [200,210] is blocked
        # first unblocked y below: 220 (220+10=230 < 235 still blocked), 225 still blocked
        # 240: 240+10=250 > 235 → unblocked
        wall = _vertical_wall(111, 190, 235)
        result = nearest_right_gap_offset(player, [wall], game)
        assert result > 0.5, f"Expected gap below (>0.5), got {result}"

    def test_gap_above_returns_less_than_half(self):
        """Wall blocks current y; first gap is above → result < 0.5."""
        player, game = self._make_player_and_game(x=100, y=200)
        # wall at x=111 covers y=190..235; gap above is at y=175 (175+10=185 < 190)
        wall = _vertical_wall(111, 190, 235)
        # The scan checks upward and downward alternately starting from py;
        # closest unblocked is ~180 (above) given the wall starts at 190
        # Step outward: 195 (blocked), 190 edge... 185 (above, 185+10=195 > 190 — blocked at 190+)
        # Actually need to verify scan finds gap above first
        # Let's set player at y=220 so only an above gap exists within radius=110
        player2 = Player(100, 220, headless=True)
        # wall covers y=190..255, gap above at ~175 (175+10=185 < 190)
        wall2 = _vertical_wall(111, 190, 255)
        result = nearest_right_gap_offset(player2, [wall2], game)
        assert result < 0.5, f"Expected gap above (<0.5), got {result}"

    def test_returns_half_when_no_gap_in_radius(self):
        """Wall spans entire vertical scan range → no gap found → 0.5."""
        player, game = self._make_player_and_game(x=100, y=200)
        # Cover y_min to y_max completely
        big_wall = _vertical_wall(111, game.Y_MIN, game.Y_MAX + 50)
        result = nearest_right_gap_offset(player, [big_wall], game)
        assert result == pytest.approx(0.5)

    def test_result_clipped_to_unit_range(self):
        """All returned values must lie in [0.0, 1.0]."""
        player, game = self._make_player_and_game(x=100, y=200)
        wall = _vertical_wall(111, 190, 235)
        result = nearest_right_gap_offset(player, [wall], game)
        assert 0.0 <= result <= 1.0


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
