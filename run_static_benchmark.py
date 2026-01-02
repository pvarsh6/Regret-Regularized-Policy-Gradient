"""run the static-opponent benchmark."""

from __future__ import annotations

import os
import sys


# allow `import benchmark` from `src/`
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from benchmark import (  # noqa: E402
    SEEDS,
    N_ROUNDS,
    get_algorithm_configs,
    resolve_eval_window,
    run_full_benchmark_multi_seed,
)


# static opponents (3)
STATIC_OPPONENTS = ["deterministic", "biased", "uniform"]

# eval window for windowed summaries in results/report
EVAL_WINDOW_K = 1000
EVAL_WINDOW_FRAC = None  # if set, overrides EVAL_WINDOW_K


if __name__ == "__main__":
    # matchups per seed = |games| * |opponents| * |algorithms|
    expected = 2 * len(STATIC_OPPONENTS) * len(get_algorithm_configs())
    print(f"Expected matchups per seed (static): {expected}")
    resolved_k = resolve_eval_window(N_ROUNDS, EVAL_WINDOW_K, EVAL_WINDOW_FRAC)
    print(f"Rounds={N_ROUNDS} | Seeds={len(SEEDS)} | Opponents={STATIC_OPPONENTS} | EvalWindowK={resolved_k}")
    outdir = run_full_benchmark_multi_seed(
        seeds=SEEDS,
        n_rounds=N_ROUNDS,
        opponents=STATIC_OPPONENTS,
        games=["matching_pennies", "rps"],
        algorithm_configs=get_algorithm_configs(),
        eval_window_k=EVAL_WINDOW_K,
        eval_window_frac=EVAL_WINDOW_FRAC,
    )
    print(f"\nWrote static benchmark artifacts to: {outdir}")


