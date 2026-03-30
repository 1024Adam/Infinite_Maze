"""Evaluation script for a saved Infinite Maze RL agent.

Usage
-----
python -m infinite_maze.ml.evaluate --model checkpoints/ppo_maze_final_1000_steps.zip --episodes 3

# Reproducible evaluation with a fixed seed
python -m infinite_maze.ml.evaluate --model checkpoints/ppo_maze_final_1000_steps.zip --episodes 10 --seed 42

# Override phase for reward-shaping context
python -m infinite_maze.ml.evaluate --model checkpoints/ppo_maze_final_1000_steps.zip --episodes 5 --phase 2
"""

import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from stable_baselines3 import PPO

from .environment import InfiniteMazeEnv


def evaluate(model_path: str, n_episodes: int, phase: int, seed: int | None) -> list[int]:
    """Run evaluation episodes and return per-episode scores.

    Parameters
    ----------
    model_path : str   Path to the saved .zip model file.
    n_episodes : int   Number of evaluation episodes to run.
    phase      : int   Environment phase (controls reward shaping).
    seed       : int | None  If provided, passed to env.reset() for reproducibility.

    Returns
    -------
    list[int]  Per-episode final scores.
    """
    env = InfiniteMazeEnv(phase=phase)
    model = PPO.load(model_path, env=env)

    scores = []
    for ep in range(n_episodes):
        reset_kwargs = {"seed": seed + ep} if seed is not None else {}
        obs, _ = env.reset(**reset_kwargs)
        episode_score = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            episode_score = info["score"]

        scores.append(episode_score)
        print(f"  Episode {ep + 1:>3}: score = {episode_score}")

    env.close()
    return scores


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a saved Infinite Maze PPO agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",    type=str, required=True,
                   help="Path to a saved model .zip file.")
    p.add_argument("--episodes", type=int, default=10,
                   help="Number of evaluation episodes to run.")
    p.add_argument("--phase",    type=int, default=1, choices=[0, 1, 2, 3, 4, 5],
                   help="Environment phase for reward shaping context.")
    p.add_argument("--seed",     type=int, default=None,
                   help="Base RNG seed for reproducible evaluation. "
                        "Episode i uses seed+i.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    print(f"Model : {args.model}")
    print(f"Phase : {args.phase}  Episodes: {args.episodes}"
          + (f"  Seed: {args.seed}" if args.seed is not None else ""))
    print()

    scores = evaluate(args.model, args.episodes, args.phase, args.seed)

    mean_score = sum(scores) / len(scores)
    print()
    print(f"Mean score : {mean_score:.1f}")
    print(f"Min / Max  : {min(scores)} / {max(scores)}")


if __name__ == "__main__":
    main()
