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

# Add sequence pattern analysis and per-episode action traces
python -m infinite_maze.ml.diagnose --model checkpoints/ppo_maze_final_10000_steps.zip \
    --steps 5000 --top-k 8 --trace-episodes 2 --trace-steps 120
"""

import argparse
import os
from collections import Counter

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


def _top_ngrams(actions: list[int], n: int, top_k: int) -> list[tuple[tuple[int, ...], int]]:
    if n <= 0 or len(actions) < n:
        return []
    counts = Counter(tuple(actions[i:i + n]) for i in range(len(actions) - n + 1))
    return counts.most_common(top_k)


def _max_run_lengths(actions: list[int]) -> dict[int, int]:
    max_runs = {a: 0 for a in range(5)}
    if not actions:
        return max_runs

    current_action = actions[0]
    current_len = 1
    for action in actions[1:]:
        if action == current_action:
            current_len += 1
        else:
            max_runs[current_action] = max(max_runs[current_action], current_len)
            current_action = action
            current_len = 1
    max_runs[current_action] = max(max_runs[current_action], current_len)
    return max_runs


def _alternating_loops(actions: list[int], min_len: int = 6) -> list[tuple[int, int, int]]:
    loops = []
    i = 0
    n = len(actions)

    while i + 3 < n:
        a = actions[i]
        b = actions[i + 1]
        if a == b:
            i += 1
            continue
        if actions[i + 2] != a or actions[i + 3] != b:
            i += 1
            continue

        j = i + 2
        while j < n and actions[j] == (a if (j - i) % 2 == 0 else b):
            j += 1

        loop_len = j - i
        if loop_len >= min_len:
            loops.append((a, b, loop_len))
            i = j
        else:
            i += 1

    return loops


def diagnose(
    model_path: str,
    n_steps: int,
    phase: int,
    seed: int | None,
    device: str,
    top_k: int,
    trace_episodes: int,
    trace_steps: int,
) -> dict:
    env = InfiniteMazeEnv(phase=phase)
    model = PPO.load(model_path, env=env, device=device)

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

    # Sequence diagnostics
    action_sequence = []
    transition_counts = Counter()
    prev_action = None

    # Optional action traces for the first N episodes
    traces = []
    current_trace = []
    trace_enabled = trace_episodes > 0 and trace_steps > 0

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        action_counts[action] += 1
        action_sequence.append(action)

        if prev_action is not None:
            transition_counts[(prev_action, action)] += 1
        prev_action = action

        if trace_enabled and len(traces) < trace_episodes and len(current_trace) < trace_steps:
            current_trace.append(action)

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

            if trace_enabled and len(traces) < trace_episodes:
                traces.append(current_trace)
                current_trace = []

            reset_kwargs2 = {"seed": seed + episodes} if seed is not None else {}
            obs, _ = env.reset(**reset_kwargs2)

    if trace_enabled and len(traces) < trace_episodes and current_trace:
        traces.append(current_trace)

    env.close()

    total_actions = sum(action_counts.values())
    mean_score    = sum(scores) / len(scores) if scores else 0.0

    top_bigrams = _top_ngrams(action_sequence, 2, top_k)
    top_trigrams = _top_ngrams(action_sequence, 3, top_k)
    max_runs = _max_run_lengths(action_sequence)
    top_transitions = transition_counts.most_common(top_k)

    loop_counts = Counter((a, b) for a, b, _ in _alternating_loops(action_sequence))
    longest_loops = {}
    for a, b, loop_len in _alternating_loops(action_sequence):
        longest_loops[(a, b)] = max(longest_loops.get((a, b), 0), loop_len)

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
        "top_bigrams":            top_bigrams,
        "top_trigrams":           top_trigrams,
        "max_runs":               max_runs,
        "top_transitions":        top_transitions,
        "loop_counts":            dict(loop_counts),
        "loop_longest":           longest_loops,
        "traces":                 traces,
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
    print("── Sequence patterns ────────────────────────────────")

    if r["top_bigrams"]:
        print("  Top 2-action patterns:")
        for gram, count in r["top_bigrams"]:
            pattern = " -> ".join(ACTION_NAMES[a] for a in gram)
            print(f"    {pattern:<28} {count:>6}")

    if r["top_trigrams"]:
        print("  Top 3-action patterns:")
        for gram, count in r["top_trigrams"]:
            pattern = " -> ".join(ACTION_NAMES[a] for a in gram)
            print(f"    {pattern:<28} {count:>6}")

    print("  Max consecutive runs:")
    for action, name in ACTION_NAMES.items():
        print(f"    {name:<12} {r['max_runs'][action]:>4}")

    if r["top_transitions"]:
        print("  Top transitions:")
        for (a, b), count in r["top_transitions"]:
            print(f"    {ACTION_NAMES[a]:<12} -> {ACTION_NAMES[b]:<12} {count:>6}")

    if r["loop_counts"]:
        print("  Alternating loop habits (A -> B -> A -> B ...):")
        loops = sorted(
            r["loop_counts"].items(),
            key=lambda item: (item[1], r["loop_longest"].get(item[0], 0)),
            reverse=True,
        )
        for (a, b), count in loops[:5]:
            longest = r["loop_longest"].get((a, b), 0)
            print(
                f"    {ACTION_NAMES[a]:<12} <-> {ACTION_NAMES[b]:<12} "
                f"occurrences={count:>4}  longest={longest:>3}"
            )

    if r["traces"]:
        print()
        print("── Action traces (first steps per episode) ─────────")
        for idx, trace in enumerate(r["traces"], start=1):
            actions = " -> ".join(ACTION_NAMES[a] for a in trace)
            print(f"  Episode {idx:>2} ({len(trace)} steps): {actions}")

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
    p.add_argument("--device", type=str, default="cpu",
                   help="Torch device for PPO load/predict (cpu, cuda, or auto).")
    p.add_argument("--top-k",  type=int, default=5,
                   help="How many top sequence patterns/transitions to print.")
    p.add_argument("--trace-episodes", type=int, default=0,
                   help="Number of episode traces to print (0 disables traces).")
    p.add_argument("--trace-steps", type=int, default=0,
                   help="Max steps per traced episode (0 disables traces).")
    return p.parse_args(argv)


def main(argv=None):
    args   = _parse_args(argv)
    result = diagnose(
        args.model,
        args.steps,
        args.phase,
        args.seed,
        args.device,
        args.top_k,
        args.trace_episodes,
        args.trace_steps,
    )
    _print_report(result, args.model)


if __name__ == "__main__":
    main()
