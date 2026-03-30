"""Behavioural diagnostic for a trained Infinite Maze agent.

Reports action distribution statistics that reveal whether the agent
has learned genuine maze navigation or is stuck in the degenerate
'spam RIGHT' policy.

Usage
-----
python -m infinite_maze.ml.diagnose --model checkpoints/ppo_maze_final_10000_steps.zip

# More steps for a stable estimate
python -m infinite_maze.ml.diagnose --model checkpoints/ppo_maze_final_10000_steps.zip --steps 5000

# Reproducible run
python -m infinite_maze.ml.diagnose --model checkpoints/ppo_maze_final_10000_steps.zip --seed 42
"""

import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from stable_baselines3 import PPO

from .environment import InfiniteMazeEnv
from .features import is_blocked_right
from ..utils.config import config

_MC = config.MOVEMENT_CONSTANTS
DO_NOTHING = _MC["DO_NOTHING"]
RIGHT      = _MC["RIGHT"]
LEFT       = _MC["LEFT"]
UP         = _MC["UP"]
DOWN       = _MC["DOWN"]

ACTION_NAMES = {DO_NOTHING: "DO_NOTHING", RIGHT: "RIGHT", LEFT: "LEFT", UP: "UP", DOWN: "DOWN"}


def diagnose(model_path: str, n_steps: int, phase: int, seed: int | None) -> dict:
    env = InfiniteMazeEnv(phase=phase)
    model = PPO.load(model_path, env=env)

    reset_kwargs = {"seed": seed} if seed is not None else {}
    obs, _ = env.reset(**reset_kwargs)

    # Counters
    action_counts   = {a: 0 for a in range(5)}
    episodes        = 0
    scores          = []
    episode_score   = 0

    # Blocked-right breakdown
    blocked_right_total     = 0
    blocked_right_vertical  = 0   # UP or DOWN while blocked right
    blocked_right_right     = 0   # RIGHT again while blocked right
    blocked_right_nothing   = 0   # DO_NOTHING while blocked right

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        action_counts[action] += 1

        blocked = is_blocked_right(env._player, env._lines)
        if blocked:
            blocked_right_total += 1
            if action in (UP, DOWN):
                blocked_right_vertical += 1
            elif action == RIGHT:
                blocked_right_right += 1
            elif action == DO_NOTHING:
                blocked_right_nothing += 1

        obs, _, terminated, truncated, info = env.step(action)
        episode_score = info["score"]

        if terminated or truncated:
            scores.append(episode_score)
            episodes += 1
            reset_kwargs2 = {"seed": seed + episodes} if seed is not None else {}
            obs, _ = env.reset(**reset_kwargs2)

    env.close()

    total_actions = sum(action_counts.values())
    mean_score    = sum(scores) / len(scores) if scores else 0.0

    return {
        "steps":                  n_steps,
        "episodes":               episodes,
        "mean_score":             mean_score,
        "min_score":              min(scores) if scores else 0,
        "max_score":              max(scores) if scores else 0,
        "action_counts":          action_counts,
        "total_actions":          total_actions,
        "blocked_right_total":    blocked_right_total,
        "blocked_right_vertical": blocked_right_vertical,
        "blocked_right_right":    blocked_right_right,
        "blocked_right_nothing":  blocked_right_nothing,
    }


def _print_report(result: dict, model_path: str) -> None:
    r             = result
    total         = r["total_actions"]
    blocked_total = r["blocked_right_total"]

    print(f"Model   : {model_path}")
    print(f"Steps   : {r['steps']}  |  Episodes completed: {r['episodes']}")
    print()

    print("── Episode scores ─────────────────────────────────")
    print(f"  Mean : {r['mean_score']:.1f}")
    print(f"  Min  : {r['min_score']}   Max : {r['max_score']}")
    print()

    print("── Action distribution (all steps) ────────────────")
    for action, name in ACTION_NAMES.items():
        count = r["action_counts"][action]
        pct   = count / total * 100 if total else 0
        bar   = "█" * int(pct / 2)
        print(f"  {name:<12} {count:>6}  ({pct:5.1f}%)  {bar}")
    print()

    print("── Behaviour when RIGHT is blocked ─────────────────")
    if blocked_total == 0:
        print("  (no blocked-right steps recorded)")
    else:
        def _row(label, count):
            pct = count / blocked_total * 100
            bar = "█" * int(pct / 2)
            print(f"  {label:<24} {count:>5}  ({pct:5.1f}%)  {bar}")

        _row("Vertical (UP or DOWN)", r["blocked_right_vertical"])
        _row("RIGHT again (degenerate)", r["blocked_right_right"])
        _row("DO_NOTHING",              r["blocked_right_nothing"])
        _row("LEFT",
             blocked_total
             - r["blocked_right_vertical"]
             - r["blocked_right_right"]
             - r["blocked_right_nothing"])
        print()

        v_pct = r["blocked_right_vertical"] / blocked_total * 100
        if v_pct < 5:
            verdict = "⚠  Degenerate policy — spamming RIGHT, no vertical navigation"
        elif v_pct < 20:
            verdict = "↗  Early navigation emerging — Phase 3 shaping needed"
        else:
            verdict = "✓  Vertical navigation established — Phase 3 target met"
        print(f"  Verdict: {verdict}  ({v_pct:.1f}% vertical when blocked)")

    print()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Behavioural diagnostic for a trained Infinite Maze agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",  type=str, required=True,
                   help="Path to a saved .zip model file.")
    p.add_argument("--steps",  type=int, default=2000,
                   help="Number of environment steps to sample.")
    p.add_argument("--phase",  type=int, default=1, choices=[0, 1, 2, 3, 4, 5],
                   help="Environment phase.")
    p.add_argument("--seed",   type=int, default=None,
                   help="RNG seed for reproducibility.")
    return p.parse_args(argv)


def main(argv=None):
    args   = _parse_args(argv)
    result = diagnose(args.model, args.steps, args.phase, args.seed)
    _print_report(result, args.model)


if __name__ == "__main__":
    main()
