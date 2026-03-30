"""
Integration tests for the Infinite Maze ML training pipeline.

Acceptance condition: PPO trains for 1,000 timesteps without exception;
final mean reward is finite. Must complete in < 60 s.

stable-baselines3 is required. If not installed the entire module is skipped.
"""

import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3", reason="stable-baselines3 not installed")

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from infinite_maze.ml.environment import InfiniteMazeEnv
from infinite_maze.utils.config import config

_ML = config.ML_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(phase=1):
    def _factory():
        return InfiniteMazeEnv(phase=phase)
    return _factory


# ---------------------------------------------------------------------------
# Environment compliance from SB3's perspective
# ---------------------------------------------------------------------------

class TestEnvCompliance:
    def test_sb3_check_env_passes(self):
        """SB3's own env checker must pass (complements gymnasium check_env)."""
        env = InfiniteMazeEnv(phase=1)
        sb3_check_env(env, warn=True)

    def test_dummy_vec_env_wraps_cleanly(self):
        env = DummyVecEnv([_make_env(phase=1)])
        obs = env.reset()
        assert obs.shape == (1, 14)
        obs, rewards, dones, infos = env.step([0])
        assert obs.shape == (1, 14)
        env.close()

    def test_4_envs_dummy_vec(self):
        """Phase 3 uses n_envs=4; confirm DummyVecEnv handles it."""
        env = DummyVecEnv([_make_env(phase=3) for _ in range(4)])
        obs = env.reset()
        assert obs.shape == (4, 14)
        env.close()


# ---------------------------------------------------------------------------
# Short PPO training smoke tests (≤ 1 000 timesteps)
# ---------------------------------------------------------------------------

class TestPPOSmokeRun:
    """PPO training smoke tests — kept small to stay well under the 60-s CI limit."""

    SHORT_STEPS = 1_000

    def test_ppo_trains_without_exception(self):
        """PPO must complete 1,000 timesteps without raising any exception."""
        env = DummyVecEnv([_make_env(phase=1)])
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            verbose=0,
        )
        model.learn(total_timesteps=self.SHORT_STEPS)
        env.close()

    def test_ppo_mean_reward_is_finite(self):
        """After training, evaluate_policy must return a finite mean reward."""
        train_env = DummyVecEnv([_make_env(phase=1)])
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            verbose=0,
        )
        model.learn(total_timesteps=self.SHORT_STEPS)

        eval_env = InfiniteMazeEnv(phase=1)
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=3, deterministic=True
        )
        assert np.isfinite(mean_reward), f"mean_reward is not finite: {mean_reward}"
        eval_env.close()
        train_env.close()

    def test_ppo_model_can_predict(self):
        """Trained model must produce valid actions for all 5 action IDs."""
        env = DummyVecEnv([_make_env(phase=1)])
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            verbose=0,
        )
        model.learn(total_timesteps=self.SHORT_STEPS)

        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1,)
        assert 0 <= int(action[0]) <= 4
        env.close()

    def test_ppo_save_and_load(self, tmp_path):
        """Model must save and reload without error; predictions are unchanged."""
        env = DummyVecEnv([_make_env(phase=1)])
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            verbose=0,
        )
        model.learn(total_timesteps=self.SHORT_STEPS)

        save_path = str(tmp_path / "test_model")
        model.save(save_path)

        loaded = PPO.load(save_path, env=env)
        obs = env.reset()
        orig_action, _ = model.predict(obs, deterministic=True)
        load_action, _ = loaded.predict(obs, deterministic=True)
        assert int(orig_action[0]) == int(load_action[0])
        env.close()


# ---------------------------------------------------------------------------
# Phase 3 multi-env smoke
# ---------------------------------------------------------------------------

class TestPhase3MultiEnv:
    def test_phase3_ppo_4_envs(self):
        """Phase 3 with n_envs=4 DummyVecEnv must train 1k steps without error."""
        env = DummyVecEnv([_make_env(phase=3) for _ in range(4)])
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            verbose=0,
        )
        model.learn(total_timesteps=1_000)
        env.close()


# ---------------------------------------------------------------------------
# Episode lifecycle integration
# ---------------------------------------------------------------------------

class TestEpisodeLifecycle:
    def test_random_policy_completes_episodes(self):
        """Random policy must complete at least one full episode in 500 steps."""
        env = InfiniteMazeEnv(phase=1)
        env.reset(seed=0)
        episodes_done = 0
        for _ in range(500):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
                episodes_done += 1
                if episodes_done >= 1:
                    break
        assert episodes_done >= 1, "Random policy should terminate at least one episode in 500 steps"

    def test_pace_increments_during_episode(self):
        """Pace must increment at least once during a 400-step episode."""
        env = InfiniteMazeEnv(phase=1)
        env.reset(seed=0)
        initial_pace = env._game.getPace()
        for _ in range(_ML["TICKS_PER_PACE_UPDATE"] + 10):
            _, _, terminated, truncated, _ = env.step(0)  # DO_NOTHING
            if terminated or truncated:
                env.reset(seed=0)
        final_pace = env._game.getPace()
        assert final_pace > initial_pace or env._tick_counter == 0, (
            "Pace should increment after TICKS_PER_PACE_UPDATE steps"
        )

    def test_score_increments_on_right_move(self):
        """Score must increase when an unblocked RIGHT move is taken."""
        env = InfiniteMazeEnv(phase=1)
        env.reset(seed=0)
        from infinite_maze.ml.features import is_blocked_right
        for _ in range(100):
            if not is_blocked_right(env._player, env._lines):
                score_before = env._game.getScore()
                env.step(1)  # RIGHT
                assert env._game.getScore() == score_before + 1
                return
            env.step(4)  # DOWN to find a gap
        pytest.skip("Could not find unblocked right in 100 steps")
