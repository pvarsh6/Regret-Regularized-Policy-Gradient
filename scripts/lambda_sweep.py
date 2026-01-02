"""lambda_R sweep to select BEST_LAMBDA_R for the paper."""

# pyright: reportMissingImports=false

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# allow imports from ../src
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from benchmark import set_seed  # noqa: E402
from games import MatchingPennies  # noqa: E402
from opponents import EXP3Opponent, MultiplicativeWeightsOpponent, UniformRandomOpponent  # noqa: E402
from policy_gradient import PolicyGradient, run_policy_gradient  # noqa: E402


# sweep config (fast proxy run)

SWEEP_LAMBDAS = [0.0001, 0.001, 0.075, 0.1]
SWEEP_SEEDS = list[int](range(10))  # 10 seeds
SWEEP_ROUNDS = 5000  # Only 5k steps
SWEEP_OPPONENTS = ["mw", "exp3"]  # adaptive opponents (selection)
SWEEP_GAMES = ["matching_pennies"]  # Just one game

# eval window (use either absolute k or fractional window)
EVAL_WINDOW_K = 1000
EVAL_WINDOW_FRAC: Optional[float] = 0.2  # if set, overrides EVAL_WINDOW_K

# model/training defaults
PG_DEFAULTS = {
    "hidden_size": 64,
    "lr": 1e-2,
    "entropy_coef": 0.01,
    "rm_type": "plus",
}


@dataclass(frozen=True)
class SweepRow:
    lam: float
    seed: int
    opponent: str
    eval_k: int
    ext_avg_lastk: float
    ext_cum_slope_lastk: float
    entropy_lastk: float
    policy_step_l1_lastk: float
    policy_var_lastk: float
    # Optional curve storage (for plotting). Only populated for some opponents.
    avg_regret_curve: Optional[np.ndarray] = None


def _make_game(game_id: str):
    game_id = str(game_id).lower()
    if game_id != "matching_pennies":
        raise ValueError(f"lambda sweep expects only matching_pennies, got {game_id!r}")
    return MatchingPennies()


def _make_opponent(opponent_id: str, game):
    opponent_id = str(opponent_id).lower()
    if opponent_id == "uniform":
        return UniformRandomOpponent(game)
    if opponent_id == "mw":
        # mirror benchmark opponent defaults
        return MultiplicativeWeightsOpponent(game, eta=0.1, role="col")
    if opponent_id == "exp3":
        # mirror benchmark opponent defaults
        return EXP3Opponent(game, eta=0.1, gamma=0.1, role="col")
    raise ValueError(f"Unknown opponent_id: {opponent_id!r}")


def _eval_window_k() -> int:
    if EVAL_WINDOW_FRAC is not None:
        k = int(float(EVAL_WINDOW_FRAC) * int(SWEEP_ROUNDS))
    else:
        k = int(EVAL_WINDOW_K)
    k = int(min(max(1, k), int(SWEEP_ROUNDS)))
    return k


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def _linear_slope(y: np.ndarray) -> float:
    """slope of y vs time index via a linear fit."""
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    if n <= 1:
        return 0.0
    t = np.arange(n, dtype=float)
    # polyfit returns [slope, intercept] for deg=1
    slope = float(np.polyfit(t, y, deg=1)[0])
    return slope


def _entropy_from_policy(policy: np.ndarray) -> np.ndarray:
    p = np.asarray(policy, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    return -np.sum(p * np.log(p), axis=-1)


def _compute_metrics(results: Dict, k: int) -> Dict[str, float]:
    # external regret (running avg)
    avg_regrets = np.asarray(results["average_regrets"], dtype=float)
    ext_avg_lastk = _safe_mean(avg_regrets[-k:])

    # cumulative regret slope (diagnostic)
    cum_regrets = np.asarray(results["cumulative_regrets"], dtype=float)
    ext_cum_slope_lastk = _linear_slope(cum_regrets[-k:])

    # entropy (runner if present; else compute from policies)
    entropies = results.get("entropies", None)
    if entropies is not None:
        ent = np.asarray(entropies, dtype=float)
        entropy_lastk = _safe_mean(ent[-k:])
    else:
        policies = np.asarray(results["policies"], dtype=float)
        entropy_lastk = _safe_mean(_entropy_from_policy(policies[-k:]))

    # policy stability diagnostics
    policies = np.asarray(results["policies"], dtype=float)
    lastk_p = policies[-k:]
    if lastk_p.shape[0] <= 1:
        policy_step_l1_lastk = 0.0
    else:
        diffs = np.diff(lastk_p, axis=0)
        policy_step_l1_lastk = _safe_mean(np.sum(np.abs(diffs), axis=1))

    # variance of policy over time (higher can indicate oscillation)
    policy_var_lastk = float(np.mean(np.var(lastk_p, axis=0)))

    return {
        "ext_avg_lastk": float(ext_avg_lastk),
        "ext_cum_slope_lastk": float(ext_cum_slope_lastk),
        "entropy_lastk": float(entropy_lastk),
        "policy_step_l1_lastk": float(policy_step_l1_lastk),
        "policy_var_lastk": float(policy_var_lastk),
    }


def run_sweep() -> Tuple[List[SweepRow], float]:
    rows: List[SweepRow] = []
    k = _eval_window_k()
    adaptive_opponents = [o for o in SWEEP_OPPONENTS if o in {"mw", "exp3"}]
    if not adaptive_opponents:
        raise ValueError("SWEEP_OPPONENTS must include at least one adaptive opponent ('mw' or 'exp3').")

    for game_id in SWEEP_GAMES:
        game = _make_game(game_id)

        for lam in SWEEP_LAMBDAS:
            for opp_id in SWEEP_OPPONENTS:
                for seed in SWEEP_SEEDS:
                    set_seed(seed)

                    opponent = _make_opponent(opp_id, game)

                    algo = PolicyGradient(
                        game,
                        hidden_size=int(PG_DEFAULTS["hidden_size"]),
                        lr=float(PG_DEFAULTS["lr"]),
                        entropy_coef=float(PG_DEFAULTS["entropy_coef"]),
                        regret_matching_coef=float(lam),
                        rm_type=str(PG_DEFAULTS["rm_type"]),
                        role="row",
                        name=f"PG-RM+_{lam}",
                    )

                    results = run_policy_gradient(algo, opponent, SWEEP_ROUNDS, verbose=False, algorithm_role="row")
                    m = _compute_metrics(results, k=k)

                    # Store curve only for MW (for optional plotting)
                    curve = None
                    if str(opp_id).lower() == "mw":
                        curve = np.asarray(results["average_regrets"], dtype=float)

                    rows.append(SweepRow(
                        lam=float(lam),
                        seed=int(seed),
                        opponent=str(opp_id),
                        eval_k=int(k),
                        ext_avg_lastk=float(m["ext_avg_lastk"]),
                        ext_cum_slope_lastk=float(m["ext_cum_slope_lastk"]),
                        entropy_lastk=float(m["entropy_lastk"]),
                        policy_step_l1_lastk=float(m["policy_step_l1_lastk"]),
                        policy_var_lastk=float(m["policy_var_lastk"]),
                        avg_regret_curve=curve,
                    ))

    # selection: minimize worst-case adaptive external regret (last k)
    summary = summarize(rows, adaptive_opponents=adaptive_opponents)
    best_lam = select_best_lambda(summary)
    return rows, float(best_lam)


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return 0.0, 0.0
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return mean, std


def _stderr(vals: List[float]) -> float:
    a = np.asarray(vals, dtype=float)
    if a.size <= 1:
        return 0.0
    return float(a.std(ddof=1) / np.sqrt(a.size))


def summarize(rows: List[SweepRow], adaptive_opponents: List[str]) -> Dict[float, Dict[str, float]]:
    """aggregate per lambda, including an adaptive worst-case metric over seeds."""
    out: Dict[float, Dict[str, float]] = {}
    seeds = sorted(set(int(r.seed) for r in rows))

    for lam in SWEEP_LAMBDAS:
        rs_lam = [r for r in rows if abs(r.lam - float(lam)) < 1e-12]
        if not rs_lam:
            continue

        # per-opponent external regret mean ± stderr (across seeds)
        per_opp: Dict[str, Dict[str, float]] = {}
        for opp in SWEEP_OPPONENTS:
            rs = [r for r in rs_lam if r.opponent == opp]
            ext_vals = [r.ext_avg_lastk for r in rs]
            ext_m, _ = _mean_std(ext_vals)
            ext_se = _stderr(ext_vals)

            per_opp[opp] = {
                "ext_mean": float(ext_m),
                "ext_stderr": float(ext_se),
                # stability means (across seeds)
                "policy_step_l1_mean": float(np.mean([r.policy_step_l1_lastk for r in rs])) if rs else 0.0,
                "entropy_mean": float(np.mean([r.entropy_lastk for r in rs])) if rs else 0.0,
                "slope_mean": float(np.mean([r.ext_cum_slope_lastk for r in rs])) if rs else 0.0,
            }

        # adaptive worst-case per seed (vs max-of-means)
        worst_ext_by_seed: List[float] = []
        worst_policy_step_by_seed: List[float] = []
        worst_slope_by_seed: List[float] = []
        worst_entropy_by_seed: List[float] = []

        for s in seeds:
            rs_seed = [r for r in rs_lam if int(r.seed) == int(s)]
            rs_ad = [r for r in rs_seed if r.opponent in set(adaptive_opponents)]
            if not rs_ad:
                continue

            worst_ext_by_seed.append(max(r.ext_avg_lastk for r in rs_ad))
            worst_policy_step_by_seed.append(max(r.policy_step_l1_lastk for r in rs_ad))
            worst_slope_by_seed.append(max(r.ext_cum_slope_lastk for r in rs_ad))
            worst_entropy_by_seed.append(min(r.entropy_lastk for r in rs_ad))

        out[float(lam)] = {
            "eval_k": float(rs_lam[0].eval_k),
            # primary selection metric
            "worst_adaptive_ext_mean": float(np.mean(worst_ext_by_seed)) if worst_ext_by_seed else 0.0,
            "worst_adaptive_ext_stderr": float(_stderr(worst_ext_by_seed)),
            # tie-breakers
            "worst_adaptive_policy_step_l1_mean": float(np.mean(worst_policy_step_by_seed)) if worst_policy_step_by_seed else 0.0,
            "worst_adaptive_entropy_mean": float(np.mean(worst_entropy_by_seed)) if worst_entropy_by_seed else 0.0,
            "worst_adaptive_slope_mean": float(np.mean(worst_slope_by_seed)) if worst_slope_by_seed else 0.0,
        }

        # flatten per-opponent ext metrics for printing
        for opp, d in per_opp.items():
            out[float(lam)][f"ext_mean_{opp}"] = float(d["ext_mean"])
            out[float(lam)][f"ext_stderr_{opp}"] = float(d["ext_stderr"])

        # also provide mean stability metrics over adaptive opponents (informational)
        ad_rs = [r for r in rs_lam if r.opponent in set(adaptive_opponents)]
        out[float(lam)]["policy_step_l1_mean_adaptive"] = float(np.mean([r.policy_step_l1_lastk for r in ad_rs])) if ad_rs else 0.0
        out[float(lam)]["entropy_mean_adaptive"] = float(np.mean([r.entropy_lastk for r in ad_rs])) if ad_rs else 0.0

    return out


def select_best_lambda(summary: Dict[float, Dict[str, float]]) -> float:
    def key(lam: float):
        s = summary[lam]
        return (
            float(s["worst_adaptive_ext_mean"]),                # PRIMARY: lower worst-case adaptive ext regret
            float(s["worst_adaptive_policy_step_l1_mean"]),     # TIE 1: more stable (lower L1 step)
            -float(s["worst_adaptive_entropy_mean"]),           # TIE 2: avoid collapse (higher entropy)
            float(s["worst_adaptive_slope_mean"]),              # TIE 3: flatter cum regret slope
        )

    return float(min(summary.keys(), key=key))


def print_summary_table(rows: List[SweepRow], best_lam: float) -> None:
    adaptive_opponents = [o for o in SWEEP_OPPONENTS if o in {"mw", "exp3"}]
    summary = summarize(rows, adaptive_opponents=adaptive_opponents)
    k = int(_eval_window_k())

    print("\nLAMBDA SWEEP SUMMARY")
    print("-" * 80)
    print(f"lambdas={SWEEP_LAMBDAS}")
    print(f"seeds={SWEEP_SEEDS}")
    print(f"rounds={SWEEP_ROUNDS}")
    print(f"eval_window_k={k} (EVAL_WINDOW_FRAC={EVAL_WINDOW_FRAC}, EVAL_WINDOW_K={EVAL_WINDOW_K})")
    print(f"games={SWEEP_GAMES}")
    print(f"opponents={SWEEP_OPPONENTS}")
    print(f"adaptive_opponents_used_for_selection={adaptive_opponents}")
    print("")

    header = (
        "lambda_R | worst_adaptive_ext_avg_lastK (±stderr) | "
        "MW ext_avg_lastK (±stderr) | EXP3 ext_avg_lastK (±stderr) | "
        "policy_step_L1_lastK(worst) | entropy_lastK(worst) | ext_cum_slope_lastK(worst)"
    )
    print(header)
    print("-" * len(header))

    for lam in SWEEP_LAMBDAS:
        a = summary[float(lam)]
        tag = "  <== BEST" if abs(float(lam) - float(best_lam)) < 1e-12 else ""

        def fmt_ext(opp: str) -> str:
            if f"ext_mean_{opp}" not in a:
                return "n/a"
            return f"{a[f'ext_mean_{opp}']:.6f} ± {a[f'ext_stderr_{opp}']:.6f}"

        print(
            f"{lam:<7} | "
            f"{a['worst_adaptive_ext_mean']:.6f} ± {a['worst_adaptive_ext_stderr']:.6f} | "
            f"{fmt_ext('mw'):<23} | "
            f"{fmt_ext('exp3'):<25} | "
            f"{a['worst_adaptive_policy_step_l1_mean']:.6f} | "
            f"{a['worst_adaptive_entropy_mean']:.6f} | "
            f"{a['worst_adaptive_slope_mean']:.6f}"
            f"{tag}"
        )

    print("\nRecommended BEST_LAMBDA_R =", best_lam)
    print("Selection rule:")
    print("  PRIMARY: minimize worst_adaptive_ext_avg_lastK (over adaptive opponents, averaged over seeds)")
    print("  TIEBREAKERS: lower policy_step_L1_lastK (more stable), higher entropy_lastK, lower ext_cum_slope_lastK")


def maybe_plot_top3_mw_curves(rows: List[SweepRow], best_lam: float) -> None:
    # optional plotting: mean ± stderr of avg external regret vs MW for top 3 lambdas
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    adaptive_opponents = [o for o in SWEEP_OPPONENTS if o in {"mw", "exp3"}]
    summary = summarize(rows, adaptive_opponents=adaptive_opponents)
    top_lams = sorted(summary.keys(), key=lambda lam: summary[lam]["worst_adaptive_ext_mean"])[:3]

    # collect MW curves for these lambdas
    outdir = os.path.join(ROOT, "out")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "lambda_sweep_regret_curves.png")

    plt.figure(figsize=(9, 5))
    x = np.arange(1, int(SWEEP_ROUNDS) + 1)

    for lam in top_lams:
        curves = [
            np.asarray(r.avg_regret_curve, dtype=float)
            for r in rows
            if abs(r.lam - float(lam)) < 1e-12 and r.opponent == "mw" and r.avg_regret_curve is not None
        ]
        if not curves:
            continue
        A = np.stack(curves, axis=0)  # (n_seeds, T)
        mean = A.mean(axis=0)
        stderr = A.std(axis=0, ddof=1) / np.sqrt(A.shape[0]) if A.shape[0] > 1 else np.zeros_like(mean)
        plt.plot(x, mean, label=f"λ={lam}")
        plt.fill_between(x, mean - stderr, mean + stderr, alpha=0.2)

    plt.xlabel("round")
    plt.ylabel("average external regret")
    plt.title(f"lambda sweep: avg external regret vs MW (top 3 λ by worst-case adaptive ext)\n(best λ={best_lam})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


if __name__ == "__main__":
    rows, best = run_sweep()
    print_summary_table(rows, best_lam=best)
    maybe_plot_top3_mw_curves(rows, best_lam=best)


