from __future__ import annotations

import csv
import json
import os
import random
import time
import traceback
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from games import MatchingPennies, RockPaperScissors, RepeatedGame
from opponents import (
    UniformRandomOpponent, 
    DeterministicOpponent, 
    FixedBiasedOpponent,
    MultiplicativeWeightsOpponent,
    EXP3Opponent
)
from classical_algorithms import MultiplicativeWeights, EXP3, run_algorithm
from policy_gradient import PolicyGradient, run_policy_gradient

try:
    import torch
except Exception:
    torch = None


# experiment config
SEEDS = list(range(20))
N_ROUNDS = 10000
OPPONENTS = ["deterministic", "biased", "uniform", "mw", "exp3"]
GAMES = ["matching_pennies", "rps"]

# eval window defaults (for steady-state summaries)
EVAL_WINDOW_K_DEFAULT = 2000
EVAL_WINDOW_FRAC_DEFAULT: float | None = 0.2

# best-performing regret-matching coefficient from lambda sweep
BEST_LAM = "0.1"
BEST_LAMBDA_R = float(BEST_LAM)


def get_algorithm_configs() -> List[Tuple[str, str, Dict[str, Any]]]:
    """return list of (name, family, kwargs)."""
    configs: List[Tuple[str, str, Dict[str, Any]]] = []

    # baselines
    configs.append(("MW", "classical", {"type": "mw"}))
    configs.append(("EXP3", "classical", {"type": "exp3"}))

    # pg vanilla
    configs.append(("PG-Vanilla", "pg", {
        "regret_matching_coef": 0.0,
        "entropy_coef": 0.01
    }))

    # pg-rm+ sweep
    for lambda_r in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        configs.append((f"PG-RM+ {lambda_r}", "pg", {
            "regret_matching_coef": lambda_r,
            "rm_type": "plus",
            "entropy_coef": 0.01
        }))

    # variants (using BEST_LAMBDA_R)
    configs.append(("PG-RM-NoClip", "pg", {
        "regret_matching_coef": BEST_LAMBDA_R,
        "rm_type": "no_clip",
        "entropy_coef": 0.01
    }))

    configs.append(("PG-SoftmaxRM 0.5", "pg", {
        "regret_matching_coef": BEST_LAMBDA_R,
        "rm_type": "softmax",
        "rm_tau": 0.5,
        "entropy_coef": 0.01
    }))

    configs.append(("PG-SoftmaxRM 1.0", "pg", {
        "regret_matching_coef": BEST_LAMBDA_R,
        "rm_type": "softmax",
        "rm_tau": 1.0,
        "entropy_coef": 0.01
    }))

    # discounted regret variants
    configs.append(("PG-DiscountRM+ 0.95", "pg", {
        "regret_matching_coef": BEST_LAMBDA_R,
        "rm_type": "plus",
        "rm_discount": 0.95,
        "entropy_coef": 0.01
    }))

    configs.append(("PG-DiscountRM+ 0.99", "pg", {
        "regret_matching_coef": BEST_LAMBDA_R,
        "rm_type": "plus",
        "rm_discount": 0.99,
        "entropy_coef": 0.01
    }))

    # progressive schedule variants
    configs.append(("PG-Progressive-Linear", "pg", {
        "regret_matching_coef": BEST_LAMBDA_R,
        "rm_type": "plus",
        "rm_schedule": "linear_decay",
        "entropy_coef": 0.01
    }))

    configs.append(("PG-Progressive-Exp", "pg", {
        "regret_matching_coef": BEST_LAMBDA_R,
        "rm_type": "plus",
        "rm_schedule": "exponential_decay",
        "rm_schedule_alpha": 3.0,
        "entropy_coef": 0.01
    }))

    # ablations: rm vs entropy
    configs.append(("PG-RM+only", "pg", {
        "regret_matching_coef": BEST_LAMBDA_R,
        "rm_type": "plus",
        "entropy_coef": 0.0
    }))

    configs.append(("PG-Entropyonly", "pg", {
        "regret_matching_coef": 0.0,
        "entropy_coef": 0.1  # higher since it's the only regularizer
    }))

    return configs


# plot/report ordering (new algos appear after these in sorted order)
OPPONENT_ORDER = ["Deterministic", "Biased", "Uniform Random", "MW Opponent", "EXP3 Opponent"]
ALGORITHM_ORDER = ["MW", "EXP3", "PG-Vanilla", f"PG-RM+ {BEST_LAM}"]
RAW_CSV_FIELDNAMES = [
    "seed",
    "game",
    "opponent_group",
    "opponent",
    "algorithm",
    "role",
    "n_rounds",
    "instant_regret_avg",
    "external_regret_avg",
    "external_regret_pos_avg",
    "external_regret_cum_last",
    "exploitability",
    "time_avg_exploitability",
    "entropy",
    "final_policy",
    # instant regret downsample (compact, replot-friendly)
    "instant_regret_ds_stride",
    "instant_regret_per_round_ds",
    # windowed summaries (last-K)
    "eval_window_k",
    "external_regret_avg_lastk",
    "instant_regret_avg_lastk",
    "exploitability_lastk",
    "time_avg_exploitability_lastk",
    "entropy_lastk",
    # curve artifact linkage (scalar-only, replot-friendly)
    "curves_npz_path",
    "curve_seed_index",
    "curve_algo_index",
]

# plot config

PLOT_EXTERNAL_REGRET_CURVES = True
PLOT_INSTANT_REGRET_CURVES = True

# regret-curve uncertainty bands (off by default)
SHADE_REGRET_BANDS = False
BAND_MODE = "sem"  # "sem" or "ci95"
CI95_Z = 1.96
REGRET_BAND_EPS = 1e-12

# plot opts
SAVEFIG_BBOX_TIGHT = True
SAVEFIG_PAD_INCHES = 0.05

# bar annotations
ANNOTATE_BAR_VALUES = True
ANNOTATE_BAR_VALUES_FONTSIZE = 7

# er curve scale/zoom
ER_AVG_USE_LOG_SCALE = True
ER_AVG_LOG_MODE = "symlog"    # "log" or "symlog"
ER_AVG_SYMLOG_LINTHRESH = 1e-6
ER_TAIL_INSET = False         # tail zoom inset (effective for long runs)
ER_TAIL_INSET_FRAC = 0.2      # show last 30% of rounds
ER_TAIL_INSET_LOC = "lower right"
ER_TAIL_INSET_SIZE = "30%"

# curve artifact persistence (avoid multi-gb csvs on long runs)
SAVE_CURVES = True
CURVES_FORMAT = "npz"          # currently only "npz" supported
CURVES_DOWNSAMPLE = 1          # 1 = full, 10 = every 10 rounds (always includes last round)
CURVES_FLOAT_DTYPE = "float32" # stored dtype inside npz
CURVES_PACKING = "per_matchup" # "per_matchup" (preferred) or "per_run"
SAVE_INSTANT_REGRET_TS = False # optional; off by default (can be huge)

# baselines only appear in dedicated comparison plots
PLOT_BASELINES = ["MW", "EXP3", "PG-Vanilla"]

instant_regret_ds_stride = 25

# paper groups
PAPER_ALGO_GROUPS: Dict[str, List[str]] = {
    # full landscape (includes baselines)
    "baseline_comparison": [f"PG-RM+ {BEST_LAM}", "MW", "EXP3", "PG-Vanilla"],

    # core ablation
    "ablation_components": [
        f"PG-RM+ {BEST_LAM}",  # both (full method)
        "PG-Vanilla",
        "PG-Entropyonly",      # just entropy
        "PG-RM+only"           # just RM+ regularizer
    ],
}

# diagnostic groups (no baselines; include the anchor pg-rm+ {BEST_LAM})
DIAGNOSTIC_ALGO_GROUPS: Dict[str, List[str]] = {
    "rmplus_lambda_sweep": [
        "PG-RM+ 0.05",
        f"PG-RM+ {BEST_LAM}",  # anchor
        "PG-RM+ 0.2",
        "PG-RM+ 0.5",
        "PG-RM+ 1.0",
        "PG-RM+ 2.0"
    ],
    "functional_form": [
        f"PG-RM+ {BEST_LAM}",
        "PG-RM-NoClip",
        "PG-SoftmaxRM 0.5",
        "PG-SoftmaxRM 1.0"
    ],
    "discounting": [
        f"PG-RM+ {BEST_LAM}",
        "PG-DiscountRM+ 0.95",
        "PG-DiscountRM+ 0.99"
    ],
    "schedules": [
        f"PG-RM+ {BEST_LAM}",
        "PG-Progressive-Linear",
        "PG-Progressive-Exp"
    ],
}

# plot everything, but keep groups explicit (paper + diagnostic)
ALGO_GROUPS: Dict[str, List[str]] = {**PAPER_ALGO_GROUPS, **DIAGNOSTIC_ALGO_GROUPS}


def _dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _is_paper_group(group_name: str) -> bool:
    return str(group_name) in set(PAPER_ALGO_GROUPS.keys())


def _band_scale(mode: str, std: np.ndarray, n: int) -> np.ndarray:
    """return band half-width (sem or 95% ci)."""
    n_eff = max(1, int(n))
    sem = std / np.sqrt(float(n_eff))
    m = str(mode).lower()
    if m in {"ci95", "95", "95ci"}:
        return CI95_Z * sem
    if m in {"sem", "stderr"}:
        return sem
    raise ValueError(f"Unknown band mode: {mode!r}")


def _clip_regret_band_lower(lo: np.ndarray, use_log_scale: bool) -> np.ndarray:
    """clip lower band to nonnegative (or eps on log scale)."""
    if use_log_scale:
        return np.maximum(lo, float(REGRET_BAND_EPS))
    return np.maximum(lo, 0.0)


def legend_right(ax, fig, fontsize: str = "x-small") -> None:
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=fontsize, frameon=False)
    # rely on tight bbox during save (avoid forcing a big right margin)


def _savefig(path: str, fig=None, dpi: int = 200) -> None:
    """save a figure with consistent bbox/padding."""
    if fig is None:
        fig = plt.gcf()
    if SAVEFIG_BBOX_TIGHT:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=SAVEFIG_PAD_INCHES)
    else:
        fig.savefig(path, dpi=dpi)


def _game_acronym(game_name: str) -> str:
    n = str(game_name).lower()
    if "matching" in n:
        return "MP"
    if "rock" in n or "rps" in n:
        return "RPS"
    return "".join([c for c in game_name if c.isalnum()])[:8]


def _metric_acronym(metric: str) -> str:
    return {
        "instant_regret_avg": "IR",
        "external_regret_pos_avg": "ER+",
        "external_regret_avg": "ER",
        "exploitability": "EXP",
        "time_avg_exploitability": "TAEXP",
        "entropy": "H",
        "instant_regret_avg_lastk": "IRk",
        "external_regret_avg_lastk": "ERk",
        "exploitability_lastk": "EXP_k",
        "time_avg_exploitability_lastk": "TAEXP_k",
        "entropy_lastk": "Hk",
    }.get(metric, metric.replace("_", "")[:10])


def _opponent_acronym(opponent_group: str) -> str:
    """short label for filenames/titles."""
    m = {
        "Deterministic": "Det",
        "Uniform Random": "Uni",
        "Biased": "Bias",
        "MW Opponent": "MW",
        "EXP3 Opponent": "EXP3",
    }
    if opponent_group in m:
        return m[opponent_group]
    # fallback: alnum, short
    s = "".join(c for c in str(opponent_group) if c.isalnum())
    return s[:8] if s else "Opp"


def _assert_plot_group_coverage(algorithms_present: List[str]) -> None:
    """fail fast if plot grouping misses any algorithms."""
    plotted = set(PLOT_BASELINES)
    for _, algos in ALGO_GROUPS.items():
        plotted.update(algos)

    present = set(algorithms_present)
    missing = sorted(present - plotted)
    extra = sorted(plotted - present)
    if missing:
        raise AssertionError(
            "Plot grouping is missing algorithms (won't appear in grouped curve plots): "
            + ", ".join(missing)
        )
    # extra entries are ok (e.g. if configs change), but warn loudly
    if extra:
        warnings.warn("Plot grouping contains algorithms not present in this run: " + ", ".join(extra))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def print_section_header(title: str):
    print(f"\n{title}")
    print("-" * len(title))


def print_algorithm_config(algo_name: str, params: Dict):
    print(f"{algo_name}:")
    for key, value in params.items():
        print(f"  {key}: {value}")


def _instant_regret_per_round(game: RepeatedGame, actions: np.ndarray, opponent_actions: np.ndarray, role: str) -> np.ndarray:
    actions = np.asarray(actions, dtype=int)
    opponent_actions = np.asarray(opponent_actions, dtype=int)
    if actions.shape != opponent_actions.shape:
        raise ValueError("actions and opponent_actions must have same shape")
    if actions.size == 0:
        return np.array([], dtype=float)

    if role == "row":
        realized = game.payoff_matrix[actions, opponent_actions]
        best = game.payoff_matrix[:, opponent_actions].max(axis=0)
        return best - realized

    if role == "col":
        realized = -game.payoff_matrix[opponent_actions, actions]
        best = (-game.payoff_matrix[opponent_actions, :]).max(axis=1)
        return best - realized

    raise ValueError(f"Unknown role: {role!r}")


def resolve_eval_window(n_rounds: int, k: int | None, frac: float | None) -> int:
    """resolve last-k evaluation window size."""
    n_rounds = int(n_rounds)
    if n_rounds <= 0:
        return 1

    if frac is not None:
        k_resolved = int(float(frac) * float(n_rounds))
    elif k is not None:
        k_resolved = int(k)
    else:
        k_resolved = int(EVAL_WINDOW_K_DEFAULT)

    # clamp
    if k_resolved > n_rounds:
        warnings.warn(f"eval_window_k ({k_resolved}) > n_rounds ({n_rounds}); clamping to {n_rounds}")
    return max(1, min(int(k_resolved), int(n_rounds)))


def _safe_last_k_mean(arr: Any, k: int, name: str) -> float:
    if arr is None:
        warnings.warn(f"Missing array for {name}; recording NaN")
        return float("nan")
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        warnings.warn(f"Empty array for {name}; recording NaN")
        return float("nan")
    kk = int(max(1, min(int(k), int(a.size))))
    return float(np.mean(a[-kk:]))


def extract_metrics(
    results: Dict,
    n_rounds: int,
    game: RepeatedGame,
    algo_role: str,
    eval_window_k: int | None = None,
    eval_window_frac: float | None = None,
) -> Dict[str, Any]:
    final_policy = results['policies'][-1]
    final_exploit = results['exploitabilities'][-1]
    final_time_avg_exploit = results.get("time_avg_exploitabilities", results["exploitabilities"])[-1]
    final_entropy = -np.sum(final_policy * np.log(final_policy + 1e-10))

    actions = np.asarray(results["actions"], dtype=int)
    opp_actions = np.asarray(results["opponent_actions"], dtype=int)

    external_cum = game.compute_regret(actions, opp_actions, role=algo_role)
    external_avg = external_cum / max(1, int(n_rounds))
    external_avg_pos = max(0.0, external_avg)

    instant_cum = game.compute_instant_regret(actions, opp_actions, role=algo_role)
    instant_avg = instant_cum / max(1, int(n_rounds))

    inst_per_round = _instant_regret_per_round(game, actions, opp_actions, role=algo_role)
    avg_external_running = np.asarray(results["average_regrets"], dtype=float)
    exploitability_ts = np.asarray(results["exploitabilities"], dtype=float)
    time_avg_exploitability_ts = np.asarray(results.get("time_avg_exploitabilities", exploitability_ts), dtype=float)
    entropies_ts = results.get("entropies", None)
    if entropies_ts is None:
        # fall back to entropy computed from policy history
        policies_ts = np.asarray(results.get("policies", []), dtype=float)
        if policies_ts.size == 0:
            entropies_ts = None
        else:
            entropies_ts = -np.sum(policies_ts * np.log(policies_ts + 1e-10), axis=1)

    if algo_role == "row":
        rewards = game.payoff_matrix[actions, opp_actions]
    elif algo_role == "col":
        rewards = -game.payoff_matrix[opp_actions, actions]
    else:
        raise ValueError(f"Unknown role: {algo_role!r}")

    # last-100 windows
    last_100 = min(100, n_rounds)
    avg_external_running_last_100 = float(np.mean(avg_external_running[-last_100:])) if avg_external_running.size else 0.0
    avg_instant_last_100 = float(np.mean(inst_per_round[-last_100:])) if inst_per_round.size else 0.0
    avg_exploit_last_100 = float(np.mean(exploitability_ts[-last_100:])) if exploitability_ts.size else 0.0
    avg_time_avg_exploit_last_100 = float(np.mean(time_avg_exploitability_ts[-last_100:])) if time_avg_exploitability_ts.size else 0.0

    # sanity checks
    if inst_per_round.size and float(inst_per_round.min()) < -1e-8:
        raise ValueError(f"instant regret per round should be >= 0; min={inst_per_round.min()}")
    if inst_per_round.shape != (n_rounds,):
        raise ValueError(f"instant regret ts length mismatch: {inst_per_round.shape} vs ({n_rounds},)")
    if avg_external_running.shape != (n_rounds,):
        raise ValueError(f"avg external regret ts length mismatch: {avg_external_running.shape} vs ({n_rounds},)")
    if exploitability_ts.shape != (n_rounds,):
        raise ValueError(f"exploitability ts length mismatch: {exploitability_ts.shape} vs ({n_rounds},)")
    if time_avg_exploitability_ts.shape != (n_rounds,):
        raise ValueError(f"time-avg exploitability ts length mismatch: {time_avg_exploitability_ts.shape} vs ({n_rounds},)")
    if rewards.shape != (n_rounds,):
        raise ValueError(f"reward ts length mismatch: {rewards.shape} vs ({n_rounds},)")

    scalars = np.array([external_avg, external_avg_pos, instant_avg, final_exploit, final_time_avg_exploit, final_entropy], dtype=float)
    if not np.isfinite(scalars).all():
        raise ValueError(f"Non-finite metrics detected: {scalars}")

    # windowed summaries (last-k) for steady-state tables
    resolved_k = resolve_eval_window(n_rounds=n_rounds, k=eval_window_k, frac=eval_window_frac)
    external_avg_lastk = _safe_last_k_mean(avg_external_running, resolved_k, "average_regrets")
    instant_avg_lastk = _safe_last_k_mean(results.get("instant_regrets_average", None), resolved_k, "instant_regrets_average")
    exploit_lastk = _safe_last_k_mean(exploitability_ts, resolved_k, "exploitabilities")
    time_avg_exploit_lastk = _safe_last_k_mean(time_avg_exploitability_ts, resolved_k, "time_avg_exploitabilities")
    entropy_lastk = _safe_last_k_mean(entropies_ts, resolved_k, "entropies")

    return {
        # scalars
        "external_avg": float(external_avg),
        "external_avg_pos": float(external_avg_pos),
        "instant_avg": float(instant_avg),
        "final_exploit": float(final_exploit),
        "final_time_avg_exploit": float(final_time_avg_exploit),
        "final_entropy": float(final_entropy),
        # windowed (last-K)
        "eval_window_k": int(resolved_k),
        "external_avg_lastk": float(external_avg_lastk),
        "instant_avg_lastk": float(instant_avg_lastk),
        "exploit_lastk": float(exploit_lastk),
        "time_avg_exploit_lastk": float(time_avg_exploit_lastk),
        "entropy_lastk": float(entropy_lastk),
        "avg_external_running_last_100": float(avg_external_running_last_100),
        "avg_instant_last_100": float(avg_instant_last_100),
        "avg_exploit_last_100": float(avg_exploit_last_100),
        "avg_time_avg_exploit_last_100": float(avg_time_avg_exploit_last_100),
        # arrays
        "inst_per_round": inst_per_round.astype(float),
        "avg_external_running": avg_external_running.astype(float),
        "exploitability_ts": exploitability_ts.astype(float),
        "time_avg_exploitability_ts": time_avg_exploitability_ts.astype(float),
        "rewards": np.asarray(rewards, dtype=float),
        "actions": actions,
        "opp_actions": opp_actions,
        # policy as json-friendly
        "final_policy": np.asarray(final_policy, dtype=float).tolist(),
    }


def print_metrics(results: Dict, n_rounds: int, game: RepeatedGame, algo_role: str):
    m = extract_metrics(results, n_rounds, game, algo_role, eval_window_k=None, eval_window_frac=None)
    final_policy = np.asarray(m["final_policy"], dtype=float)

    print(f"Final Policy:        {np.array2string(final_policy, precision=4, suppress_small=True)}")
    print(f"Avg External Regret (best fixed; can be negative): {m['external_avg']:.6f}")
    print(f"Avg External Regret (clipped at 0):               {m['external_avg_pos']:.6f}")
    print(f"Avg Instant Regret (diagnostic; best per-round; >= 0): {m['instant_avg']:.6f}")
    print(f"Exploitability (last-iterate):                         {m['final_exploit']:.6f}")
    print(f"Exploitability (time-avg policy):                      {m['final_time_avg_exploit']:.6f}")
    print(f"Policy Entropy:      {m['final_entropy']:.6f}")
    print(f"Eval Window K:       {m['eval_window_k']}")
    print(f"External Regret+ (avg lastK):                         {m['external_avg_lastk']:.6f}")
    print(f"Instant Regret (avg lastK):                           {m['instant_avg_lastk']:.6f}")
    print(f"Time-Avg Exploitability (avg lastK):                  {m['time_avg_exploit_lastk']:.6f}")
    print(f"Entropy (avg lastK):                                  {m['entropy_lastk']:.6f}")

    last_100 = min(100, n_rounds)
    print(f"Avg External Regret (running avg; last {last_100}): {m['avg_external_running_last_100']:.6f}")
    print(f"Avg Instant Regret (last {last_100} rounds):        {m['avg_instant_last_100']:.6f}")
    print(f"Avg Exploit (last {last_100}):                      {m['avg_exploit_last_100']:.6f}")
    print(f"Avg Time-Avg Exploit (last {last_100}):             {m['avg_time_avg_exploit_last_100']:.6f}")


def run_matchup(game: RepeatedGame, 
                algo, 
                opponent,
                n_rounds: int,
                algo_role: str = "row",
                algo_params: Dict = None,
                opp_params: Dict = None,
                verbose: bool = False,
                seed: Optional[int] = None,
                collect: bool = True,
                log_fn: Optional[Callable[[str], None]] = None,
                verbose_headers: bool = True,
                opponent_group: Optional[str] = None,
                eval_window_k: int | None = None,
                eval_window_frac: float | None = None):

    def log(msg: str = "") -> None:
        if log_fn is not None:
            log_fn(msg)
        else:
            print(msg)

    # prefer the descriptive `algo.name` for policy-gradient variants
    algorithm_name = (
        "MW" if isinstance(algo, MultiplicativeWeights)
        else "EXP3" if isinstance(algo, EXP3)
        else getattr(algo, "name", "PolicyGradient") if isinstance(algo, PolicyGradient)
        else algo.__class__.__name__
    )
    opponent_group = opponent_group or opponent.name

    if verbose_headers:
        print_section_header(f"{game.name}: {algo.__class__.__name__} vs {opponent.name}")
        if algo_params:
            print_algorithm_config("Algorithm", algo_params)
        if opp_params:
            print_algorithm_config("Opponent", opp_params)
        print(f"Role: {algo_role}")
        print(f"Rounds: {n_rounds}")
        print()
    
    if isinstance(algo, (MultiplicativeWeights, EXP3)):
        results = run_algorithm(algo, opponent, n_rounds, verbose=verbose, algorithm_role=algo_role)
    elif isinstance(algo, PolicyGradient):
        results = run_policy_gradient(algo, opponent, n_rounds, verbose=verbose, algorithm_role=algo_role)
    else:
        raise ValueError(f"Unknown algorithm type: {type(algo)}")

    metrics = extract_metrics(
        results,
        n_rounds,
        game=game,
        algo_role=algo_role,
        eval_window_k=eval_window_k,
        eval_window_frac=eval_window_frac,
    )

    record = {
        "seed": seed,
        "game": game.name,
        "opponent": opponent.name,
        "opponent_group": opponent_group,
        "algorithm": algorithm_name,
        "role": algo_role,
        "n_rounds": int(n_rounds),
        # final scalars
        "external_regret_avg": metrics["external_avg"],
        "external_regret_pos_avg": metrics["external_avg_pos"],
        "instant_regret_avg": metrics["instant_avg"],
        "exploitability": metrics["final_exploit"],
        "time_avg_exploitability": metrics["final_time_avg_exploit"],
        "entropy": metrics["final_entropy"],
        "final_policy": metrics["final_policy"],
        # instant regret downsample (CSV-friendly; full-res kept as ndarray for plots/aggregation)
        "instant_regret_ds_stride": int(instant_regret_ds_stride),
        # windowed summaries
        "eval_window_k": metrics["eval_window_k"],
        "external_regret_avg_lastk": metrics["external_avg_lastk"],
        "instant_regret_avg_lastk": metrics["instant_avg_lastk"],
        "exploitability_lastk": metrics["exploit_lastk"],
        "time_avg_exploitability_lastk": metrics["time_avg_exploit_lastk"],
        "entropy_lastk": metrics["entropy_lastk"],
        # time series (store only what we plot/aggregate)
        # keep cumulative external regret in memory (plots/aggregation; optional .npz export)
        "cumulative_regret_ts": np.asarray(results.get("cumulative_regrets", []), dtype=float),
    }

    # convenience scalar: final cumulative external regret (last value)
    try:
        record["external_regret_cum_last"] = float(record["cumulative_regret_ts"][-1]) if record["cumulative_regret_ts"].size else float("nan")
    except Exception:
        record["external_regret_cum_last"] = float("nan")

    # instant-regret time series (full-res in memory; csv gets a downsample)
    if PLOT_INSTANT_REGRET_CURVES or SAVE_INSTANT_REGRET_TS:
        inst_full = np.asarray(metrics["inst_per_round"], dtype=float)
        record["instant_regret_per_round"] = inst_full

        # downsample for csv replotting (always include last round)
        stride = max(1, int(instant_regret_ds_stride))
        idx = np.arange(0, int(n_rounds), stride, dtype=int)
        if idx.size == 0 or idx[-1] != (int(n_rounds) - 1):
            idx = np.concatenate([idx, np.array([int(n_rounds) - 1], dtype=int)])
        inst_ds = inst_full[idx] if inst_full.size else np.array([], dtype=float)
        record["instant_regret_per_round_ds"] = json.dumps([float(x) for x in inst_ds.tolist()])
    else:
        record["instant_regret_per_round_ds"] = json.dumps([])

    if verbose_headers and log_fn is None:
        # preserve interactive printing behavior
        print_metrics(results, n_rounds, game=game, algo_role=algo_role)
    elif log_fn is not None:
        # compact log line for reports
        log(
            f"{game.name} | {opponent_group} | {algorithm_name} | "
            f"inst_avg(diagnostic)={record['instant_regret_avg']:.6f} "
            f"ext_avg={record['external_regret_avg']:.6f} "
            f"ext_pos_avg={record['external_regret_pos_avg']:.6f} "
            f"exploit={record['exploitability']:.6f} "
            f"time_avg_exploit={record['time_avg_exploitability']:.6f} "
            f"entropy={record['entropy']:.6f}"
        )

    return record if collect else None


def _make_game(game_id: str) -> RepeatedGame:
    game_id = str(game_id).lower()
    if game_id in {"matching_pennies", "mp"}:
        return MatchingPennies()
    if game_id in {"rps", "rock_paper_scissors"}:
        return RockPaperScissors()
    raise ValueError(f"Unknown game: {game_id!r}")


def _make_opponent(opponent_id: str, game: RepeatedGame) -> Tuple[Any, str, Dict[str, Any]]:
    """return (opponent, opponent_group, opp_params)."""
    opponent_id = str(opponent_id).lower()

    if opponent_id == "deterministic":
        det_action = 0
        opp = DeterministicOpponent(game, det_action)
        return opp, "Deterministic", {"action": game.action_names[det_action]}

    if opponent_id == "uniform":
        opp = UniformRandomOpponent(game)
        return opp, "Uniform Random", {}

    if opponent_id == "biased":
        if game.n_actions == 2:
            probs = np.array([0.7, 0.3])
        else:
            probs = np.array([0.5, 0.3, 0.2])
        opp = FixedBiasedOpponent(game, probs)
        return opp, "Biased", {"probabilities": probs}

    if opponent_id == "mw":
        eta = 0.1
        opp = MultiplicativeWeightsOpponent(game, eta=eta, role="col")
        return opp, "MW Opponent", {"eta": eta}

    if opponent_id == "exp3":
        eta = 0.1
        gamma = 0.1
        opp = EXP3Opponent(game, eta=eta, gamma=gamma, role="col")
        return opp, "EXP3 Opponent", {"eta": eta, "gamma": gamma}

    raise ValueError(f"Unknown opponent: {opponent_id!r}")


def _make_algorithm(algo_name: str, family: str, game: RepeatedGame, algo_kwargs: Dict[str, Any], role: str = "row"):
    family = str(family).lower()
    algo_kwargs = dict(algo_kwargs or {})

    if family == "classical":
        algo_type = str(algo_kwargs.get("type", "")).lower()
        if algo_type == "mw":
            eta = float(algo_kwargs.get("eta", 0.1))
            return MultiplicativeWeights(game, eta=eta, role=role), {"eta": eta}
        if algo_type == "exp3":
            eta = float(algo_kwargs.get("eta", 0.1))
            gamma = float(algo_kwargs.get("gamma", 0.1))
            return EXP3(game, eta=eta, gamma=gamma, role=role), {"eta": eta, "gamma": gamma}
        raise ValueError(f"Unknown classical algorithm type: {algo_type!r}")

    if family == "pg":
        # defaults match prior benchmark settings unless overridden
        hidden_size = int(algo_kwargs.pop("hidden_size", 64))
        lr = float(algo_kwargs.pop("lr", 1e-2))
        pg = PolicyGradient(
            game,
            hidden_size=hidden_size,
            lr=lr,
            role=role,
            name=algo_name,  # ensure exact descriptive names in outputs
            **algo_kwargs,
        )
        params = {"hidden_size": hidden_size, "lr": lr, **algo_kwargs}
        return pg, params

    raise ValueError(f"Unknown algorithm family: {family!r}")


def benchmark_vs_deterministic(
    game: RepeatedGame,
    records: List[Dict[str, Any]],
    n_rounds: int = 3000,
    seed: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    verbose_headers: bool = True,
):
    if verbose_headers:
        print_section_header(f"BENCHMARK: {game.name} vs Deterministic Opponent")
    
    det_action = 0
    opponent_params = {"action": game.action_names[det_action]}
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    det = DeterministicOpponent(game, det_action)
    records.append(run_matchup(
        game, mw, det, n_rounds, "row",
        {"eta": 0.1}, opponent_params,
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="Deterministic",
    ))
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    det = DeterministicOpponent(game, det_action)
    records.append(run_matchup(
        game, exp3, det, n_rounds, "row",
        {"eta": 0.1, "gamma": 0.1}, opponent_params,
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="Deterministic",
    ))
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    det = DeterministicOpponent(game, det_action)
    records.append(run_matchup(
        game, pg, det, n_rounds, "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01}, opponent_params,
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="Deterministic",
    ))


def benchmark_vs_uniform(
    game: RepeatedGame,
    records: List[Dict[str, Any]],
    n_rounds: int = 3000,
    seed: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    verbose_headers: bool = True,
):
    if verbose_headers:
        print_section_header(f"BENCHMARK: {game.name} vs Uniform Random")
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    uniform = UniformRandomOpponent(game)
    records.append(run_matchup(
        game, mw, uniform, n_rounds, "row", {"eta": 0.1}, {},
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="Uniform Random",
    ))
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    uniform = UniformRandomOpponent(game)
    records.append(run_matchup(
        game, exp3, uniform, n_rounds, "row",
        {"eta": 0.1, "gamma": 0.1}, {},
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="Uniform Random",
    ))
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    uniform = UniformRandomOpponent(game)
    records.append(run_matchup(
        game, pg, uniform, n_rounds, "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01}, {},
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="Uniform Random",
    ))


def benchmark_vs_biased(
    game: RepeatedGame,
    records: List[Dict[str, Any]],
    n_rounds: int = 3000,
    seed: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    verbose_headers: bool = True,
):
    if verbose_headers:
        print_section_header(f"BENCHMARK: {game.name} vs Biased Opponent")
    
    if game.n_actions == 2:
        probs = np.array([0.7, 0.3])
    else:
        probs = np.array([0.5, 0.3, 0.2])
    
    opponent_params = {"probabilities": probs}
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    biased = FixedBiasedOpponent(game, probs)
    records.append(run_matchup(
        game, mw, biased, n_rounds, "row", {"eta": 0.1}, opponent_params,
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="Biased",
    ))
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    biased = FixedBiasedOpponent(game, probs)
    records.append(run_matchup(
        game, exp3, biased, n_rounds, "row",
        {"eta": 0.1, "gamma": 0.1}, opponent_params,
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="Biased",
    ))
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    biased = FixedBiasedOpponent(game, probs)
    records.append(run_matchup(
        game, pg, biased, n_rounds, "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01}, opponent_params,
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="Biased",
    ))


def benchmark_vs_mw(
    game: RepeatedGame,
    records: List[Dict[str, Any]],
    n_rounds: int = 3000,
    seed: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    verbose_headers: bool = True,
):
    if verbose_headers:
        print_section_header(f"BENCHMARK: {game.name} vs MW Opponent")
    
    opponent_params = {"eta": 0.1}
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    mw_opp = MultiplicativeWeightsOpponent(game, eta=0.1, role="col")
    records.append(run_matchup(
        game, mw, mw_opp, n_rounds, "row", {"eta": 0.1}, opponent_params,
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="MW Opponent",
    ))
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    mw_opp = MultiplicativeWeightsOpponent(game, eta=0.1, role="col")
    records.append(run_matchup(
        game, exp3, mw_opp, n_rounds, "row",
        {"eta": 0.1, "gamma": 0.1}, opponent_params,
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="MW Opponent",
    ))
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    mw_opp = MultiplicativeWeightsOpponent(game, eta=0.1, role="col")
    records.append(run_matchup(
        game, pg, mw_opp, n_rounds, "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01}, opponent_params,
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="MW Opponent",
    ))


def benchmark_vs_exp3(
    game: RepeatedGame,
    records: List[Dict[str, Any]],
    n_rounds: int = 3000,
    seed: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    verbose_headers: bool = True,
):
    if verbose_headers:
        print_section_header(f"BENCHMARK: {game.name} vs EXP3 Opponent")
    
    opponent_params = {"eta": 0.1, "gamma": 0.1}
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    exp3_opp = EXP3Opponent(game, eta=0.1, gamma=0.1, role="col")
    records.append(run_matchup(
        game, mw, exp3_opp, n_rounds, "row", {"eta": 0.1}, opponent_params,
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="EXP3 Opponent",
    ))
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    exp3_opp = EXP3Opponent(game, eta=0.1, gamma=0.1, role="col")
    records.append(run_matchup(
        game, exp3, exp3_opp, n_rounds, "row",
        {"eta": 0.1, "gamma": 0.1}, opponent_params,
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="EXP3 Opponent",
    ))
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    exp3_opp = EXP3Opponent(game, eta=0.1, gamma=0.1, role="col")
    records.append(run_matchup(
        game, pg, exp3_opp, n_rounds, "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01}, opponent_params,
        seed=seed, collect=True, log_fn=log_fn, verbose_headers=verbose_headers,
        opponent_group="EXP3 Opponent",
    ))


def run_full_benchmark(
    records: List[Dict[str, Any]],
    n_rounds: int = 3000,
    seed: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    verbose_headers: bool = True,
    opponents: Optional[List[str]] = None,
    games: Optional[List[str]] = None,
    algorithm_configs: Optional[List[Tuple[str, str, Dict[str, Any]]]] = None,
    eval_window_k: int | None = None,
    eval_window_frac: float | None = None,
):
    opponents = list(OPPONENTS if opponents is None else opponents)
    games = list(GAMES if games is None else games)
    algorithm_configs = list(get_algorithm_configs() if algorithm_configs is None else algorithm_configs)

    # sanity checks for the suite
    algo_names = [n for (n, _, _) in algorithm_configs]
    if len(algorithm_configs) < 3:
        raise ValueError(f"Expected at least baseline+PG configs, got {len(algorithm_configs)}: {algo_names}")
    if len(set(algo_names)) != len(algo_names):
        raise ValueError(f"Duplicate algorithm names detected: {algo_names}")

    for game_id in games:
        game = _make_game(game_id)
        if verbose_headers:
            print("\n" + "=" * 80)
            print(f"GAME: {game.name}")
            print("=" * 80)

        for opponent_id in opponents:
            opponent, opponent_group, opp_params = _make_opponent(opponent_id, game)
            for (algo_name, family, algo_kwargs) in algorithm_configs:
                algo, algo_params = _make_algorithm(algo_name, family, game, algo_kwargs, role="row")
                records.append(run_matchup(
                    game,
                    algo,
                    opponent,
                    n_rounds,
                    "row",
                    algo_params=algo_params,
                    opp_params=opp_params,
                    verbose=False,
                    seed=seed,
                    collect=True,
                    log_fn=log_fn,
                    verbose_headers=verbose_headers,
                    opponent_group=opponent_group,
                    eval_window_k=eval_window_k,
                    eval_window_frac=eval_window_frac,
                ))
                # reset opponent between runs to avoid leakage across algorithms
                if hasattr(opponent, "reset"):
                    opponent.reset()


def _strip_arrays_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in record.items():
        if isinstance(v, np.ndarray):
            # don't write time-series arrays into csv (too large); curves go to .npz
            continue
        if k == "final_policy":
            out[k] = repr(v)
        else:
            out[k] = v
    return out


def write_results_raw_csv(records: List[Dict[str, Any]], outpath: str) -> None:
    rows = [_strip_arrays_for_csv(r) for r in records]
    if not rows:
        return
    # include any extra keys so newly-added metrics don't get dropped
    extras = sorted({k for row in rows for k in row.keys()} - set(RAW_CSV_FIELDNAMES))
    fieldnames = list(RAW_CSV_FIELDNAMES) + extras
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_curves_npz(path: str):
    """load curve artifacts from an .npz file."""
    data = np.load(path, allow_pickle=False)
    return data["cum_regret"], data["seeds"], data["algorithms"], data["indices"]


def _curves_indices(n_rounds: int, downsample: int) -> np.ndarray:
    n_rounds = int(n_rounds)
    downsample = int(downsample)
    if n_rounds <= 0:
        return np.array([], dtype=int)
    if downsample <= 1:
        return np.arange(n_rounds, dtype=int)
    idx = np.arange(0, n_rounds, downsample, dtype=int)
    if idx.size == 0 or idx[-1] != (n_rounds - 1):
        idx = np.concatenate([idx, np.array([n_rounds - 1], dtype=int)])
    return idx


def write_curves_npz_per_matchup(
    records: List[Dict[str, Any]],
    outdir: str,
    seeds: List[int],
    algorithms: List[str],
    n_rounds: int,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """write per-(game, opponent_group) curve .npz files and return metadata."""
    if CURVES_FORMAT != "npz":
        raise ValueError(f"Unsupported CURVES_FORMAT: {CURVES_FORMAT!r}")
    if CURVES_PACKING not in {"per_matchup", "per_run"}:
        raise ValueError(f"Unsupported CURVES_PACKING: {CURVES_PACKING!r}")

    curves_dir = os.path.join(outdir, "curves")
    os.makedirs(curves_dir, exist_ok=True)

    idx = _curves_indices(n_rounds=n_rounds, downsample=CURVES_DOWNSAMPLE)
    n_points = int(idx.size)

    dtype = np.float32 if str(CURVES_FLOAT_DTYPE).lower() == "float32" else np.float64

    seed_to_i = {int(s): i for i, s in enumerate(seeds)}
    algo_to_j = {a: j for j, a in enumerate(algorithms)}

    matchups = sorted({(r["game"], r["opponent_group"]) for r in records})
    meta: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # pre-create unicode arrays to avoid object dtype (keeps allow_pickle=False safe)
    max_algo_len = max([len(a) for a in algorithms], default=1)
    algos_arr = np.asarray(algorithms, dtype=f"<U{max_algo_len}")
    seeds_arr = np.asarray([int(s) for s in seeds], dtype=int)

    for (game_name, opp_group) in matchups:
        cum = np.full((len(seeds), len(algorithms), n_points), np.nan, dtype=dtype)

        for r in records:
            if r["game"] != game_name or r["opponent_group"] != opp_group:
                continue
            si = seed_to_i.get(int(r["seed"]), None)
            aj = algo_to_j.get(r["algorithm"], None)
            if si is None or aj is None:
                continue
            ts = r.get("cumulative_regret_ts", None)
            if not isinstance(ts, np.ndarray) or ts.size == 0:
                continue
            if ts.shape[0] != n_rounds:
                raise ValueError(
                    f"n_rounds mismatch in cumulative regret ts for {(game_name, opp_group, r['algorithm'])}: {ts.shape}"
                )
            cum[si, aj, :] = np.asarray(ts, dtype=float)[idx].astype(dtype, copy=False)

        fname = f"curves_{_game_acronym(game_name)}_{_opponent_acronym(opp_group)}.npz"
        fname = fname.replace("(", "").replace(")", "").replace("+", "p")
        path = os.path.join(curves_dir, fname)
        np.savez_compressed(
            path,
            cum_regret=cum,
            seeds=seeds_arr,
            algorithms=algos_arr,
            indices=idx.astype(int),
        )
        meta[(game_name, opp_group)] = {"path": path, "seeds": seeds, "algorithms": algorithms, "indices": idx}

    return meta


def aggregate_results(records: List[Dict[str, Any]], n_rounds: int) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str], Dict[str, np.ndarray]]]:
    # summary rows: one per (game, opponent_group, algorithm)
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in records:
        key = (r["game"], r["opponent_group"], r["algorithm"])
        groups.setdefault(key, []).append(r)

    # metrics for results_summary.csv (explicit list + auto-add numeric scalars)
    metrics = [
        "instant_regret_avg",
        "external_regret_avg",
        "external_regret_pos_avg",
        "external_regret_cum_last",
        "exploitability",
        "time_avg_exploitability",
        "entropy",
        # windowed
        "instant_regret_avg_lastk",
        "external_regret_avg_lastk",
        "exploitability_lastk",
        "time_avg_exploitability_lastk",
        "entropy_lastk",
    ]

    excluded = {
        # identifiers / non-aggregates
        "seed", "game", "opponent", "opponent_group", "algorithm", "role", "final_policy",
        # time series
        "instant_regret_per_round", "average_regret_ts", "cumulative_regret_ts",
    }

    def _is_numeric_scalar(v: Any) -> bool:
        if isinstance(v, (bool,)):
            return False
        if isinstance(v, np.ndarray):
            return False
        return isinstance(v, (int, float, np.integer, np.floating))

    if records:
        common_keys = set.intersection(*[set(r.keys()) for r in records])
        auto_metrics = sorted(
            k for k in common_keys
            if k not in excluded and k not in set(metrics) and _is_numeric_scalar(records[0].get(k))
        )
        metrics = metrics + auto_metrics
    summary_rows: List[Dict[str, Any]] = []
    ts_aggs: Dict[Tuple[str, str, str], Dict[str, np.ndarray]] = {}

    for (game, opp_group, algo), rs in groups.items():
        row: Dict[str, Any] = {"game": game, "opponent_group": opp_group, "algorithm": algo, "n": len(rs)}
        for m in metrics:
            vals = np.asarray([float(x[m]) for x in rs], dtype=float)
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(rs) > 1 else 0.0
        summary_rows.append(row)

        ddof = 1 if len(rs) > 1 else 0

        # instant regret time-series aggregates (diagnostic / optional)
        if all(("instant_regret_per_round" in x) for x in rs):
            inst = np.stack([np.asarray(x["instant_regret_per_round"], dtype=float) for x in rs], axis=0)
            if inst.shape[1] != n_rounds:
                raise ValueError(f"n_rounds mismatch in instant regret ts for {(game, opp_group, algo)}: {inst.shape}")
            cum_inst = np.cumsum(inst, axis=1)
            avg_cum_inst = cum_inst / (np.arange(n_rounds, dtype=float) + 1.0)[None, :]
        else:
            inst = np.zeros((len(rs), n_rounds), dtype=float)
            cum_inst = np.zeros((len(rs), n_rounds), dtype=float)
            avg_cum_inst = np.zeros((len(rs), n_rounds), dtype=float)

        # external regret time-series aggregates (primary)
        cum_reg = np.stack([np.asarray(x["cumulative_regret_ts"], dtype=float) for x in rs], axis=0)
        if cum_reg.shape[1] != n_rounds:
            raise ValueError(f"n_rounds mismatch in cumulative regret ts for {(game, opp_group, algo)}: {cum_reg.shape}")
        avg_reg = cum_reg / (np.arange(n_rounds, dtype=float) + 1.0)[None, :]

        ts_aggs[(game, opp_group, algo)] = {
            # instant regret
            "inst_per_round_mean": inst.mean(axis=0),
            "inst_per_round_std": inst.std(axis=0, ddof=ddof) if ddof == 1 else np.zeros(n_rounds, dtype=float),
            "cum_inst_mean": cum_inst.mean(axis=0),
            "cum_inst_std": cum_inst.std(axis=0, ddof=ddof) if ddof == 1 else np.zeros(n_rounds, dtype=float),
            "avg_cum_inst_mean": avg_cum_inst.mean(axis=0),
            "avg_cum_inst_std": avg_cum_inst.std(axis=0, ddof=ddof) if ddof == 1 else np.zeros(n_rounds, dtype=float),
            # external regret
            "avg_regret_mean": avg_reg.mean(axis=0),
            "avg_regret_std": avg_reg.std(axis=0, ddof=ddof) if ddof == 1 else np.zeros(n_rounds, dtype=float),
            "cum_regret_mean": cum_reg.mean(axis=0),
            "cum_regret_std": cum_reg.std(axis=0, ddof=ddof) if ddof == 1 else np.zeros(n_rounds, dtype=float),
            "n": np.array([len(rs)], dtype=int),
        }

    summary_rows.sort(key=lambda r: (r["game"], OPPONENT_ORDER.index(r["opponent_group"]) if r["opponent_group"] in OPPONENT_ORDER else 999, ALGORITHM_ORDER.index(r["algorithm"]) if r["algorithm"] in ALGORITHM_ORDER else 999))
    return summary_rows, ts_aggs


def write_results_summary_csv(summary_rows: List[Dict[str, Any]], outpath: str) -> None:
    if not summary_rows:
        return
    fieldnames = sorted(summary_rows[0].keys())
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

def plot_bar_with_error(summary_rows: List[Dict[str, Any]], game_name: str, metric: str, outpath: str) -> None:
    rows = [r for r in summary_rows if r["game"] == game_name]
    if not rows:
        return

    # rows are grouped by (opponent_group, algorithm)
    means = {(r["opponent_group"], r["algorithm"]): r[f"{metric}_mean"] for r in rows}
    stds = {(r["opponent_group"], r["algorithm"]): r[f"{metric}_std"] for r in rows}
    ns = {(r["opponent_group"], r["algorithm"]): int(r.get("n", 1)) for r in rows}

    opponents = [o for o in OPPONENT_ORDER if any(r["opponent_group"] == o for r in rows)]
    algos_known = [a for a in ALGORITHM_ORDER if any(r["algorithm"] == a for r in rows)]
    algos_other = sorted({r["algorithm"] for r in rows if r["algorithm"] not in set(ALGORITHM_ORDER)})
    algos = algos_known + algos_other

    x = np.arange(len(opponents))
    width = 0.25 if len(algos) <= 3 else 0.8 / max(1, len(algos))

    fig, ax = plt.subplots(figsize=(max(10, len(opponents) * 1.8), 5.0))

    import matplotlib.transforms as mtransforms
    # transform for vertical bar labels
    text_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    BAR_LABEL_Y = -0.10  # slightly below x-axis ticks

    for i, algo in enumerate(algos):
        y = [means.get((opp, algo), np.nan) for opp in opponents]
        # 95% CI half-width: 1.96 * std / sqrt(n)
        e = []
        for opp in opponents:
            std = float(stds.get((opp, algo), 0.0))
            n = int(ns.get((opp, algo), 1))
            e.append(float(_band_scale("ci95", np.asarray(std, dtype=float), n)))

        bars = ax.bar(
            x + (i - (len(algos) - 1) / 2) * width,
            y,
            width,
            yerr=e,
            capsize=3,
            label=algo,
            zorder=2,
        )

        for rect, mean, ci in zip(bars, y, e):
            if mean is None or not np.isfinite(mean) or ci is None or not np.isfinite(ci):
                continue

            label = f"{mean:.3g} ± {ci:.2g}"

            ax.text(
                rect.get_x() + rect.get_width() / 2,
                BAR_LABEL_Y,
                label,
                ha="center",
                va="top",
                fontsize=ANNOTATE_BAR_VALUES_FONTSIZE,
                rotation=90,
                transform=text_transform,
                clip_on=False,
                zorder=10,
            )

    # horizontal opponent labels
    ax.set_xticks(x)
    ax.set_xticklabels(opponents, rotation=0, ha="center", fontsize=9)

    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{game_name}: {metric} (mean ± 95% CI over seeds)")

    # room for bar labels below x-axis
    fig.subplots_adjust(bottom=0.25)

    legend_right(ax, fig, fontsize="x-small")
    _savefig(outpath, fig=fig, dpi=200)
    plt.close(fig)

def plot_regret_curves(
    ts_aggs: Dict[Tuple[str, str, str], Dict[str, np.ndarray]],
    game_name: str,
    opponent_group: str,
    outpath: str,
    use_avg: bool = True,
) -> None:
    """legacy instant-regret curve plot (diagnostic)."""
    if not PLOT_INSTANT_REGRET_CURVES:
        return

    keys = [(g, o, a) for (g, o, a) in ts_aggs.keys() if g == game_name and o == opponent_group]
    if not keys:
        return
    n_rounds = len(ts_aggs[keys[0]]["avg_cum_inst_mean"] if use_avg else ts_aggs[keys[0]]["cum_inst_mean"])
    x = np.arange(1, n_rounds + 1)

    fig = plt.figure(figsize=(10, 5))
    algos_present = sorted({a for (g, o, a) in keys})
    algos_known = [a for a in ALGORITHM_ORDER if a in algos_present]
    algos_other = [a for a in algos_present if a not in set(ALGORITHM_ORDER)]
    for algo in algos_known + algos_other:
        k = (game_name, opponent_group, algo)
        if use_avg:
            mean = ts_aggs[k]["avg_cum_inst_mean"]
            std = ts_aggs[k]["avg_cum_inst_std"]
            ylabel = "avg cumulative instant regret"
        else:
            mean = ts_aggs[k]["cum_inst_mean"]
            std = ts_aggs[k]["cum_inst_std"]
            ylabel = "cumulative instant regret"
        plt.plot(x, mean, label=algo)
        if SHADE_REGRET_BANDS:
            n = int(ts_aggs[k].get("n", np.array([1]))[0])
            band = _band_scale(BAND_MODE, std, n)
            lo = _clip_regret_band_lower(mean - band, use_log_scale=False)
            hi = mean + band
            plt.fill_between(x, lo, hi, alpha=0.18)

    plt.xlabel("round")
    plt.ylabel(ylabel)
    shaded = "none" if not SHADE_REGRET_BANDS else ("95% CI" if str(BAND_MODE).lower() in {"ci95", "95", "95ci"} else "SEM")
    plt.title(f"{game_name}: {opponent_group} ({'avg' if use_avg else 'cum'}) | shaded: {shaded}")
    legend_right(plt.gca(), fig, fontsize="x-small")
    _savefig(outpath, fig=fig, dpi=200)
    plt.close()


def plot_instant_regret_curves_grouped(
    ts_aggs: Dict[Tuple[str, str, str], Dict[str, np.ndarray]],
    game_name: str,
    opponent_group: str,
    outdir: str,
    use_avg: bool = True,
) -> None:
    """instant-regret curves, grouped to match external-regret plotting."""
    if not PLOT_INSTANT_REGRET_CURVES:
        return

    keys_all = [(g, o, a) for (g, o, a) in ts_aggs.keys() if g == game_name and o == opponent_group]
    if not keys_all:
        return
    algos_present = sorted({a for (_, _, a) in keys_all})

    mean_key = "avg_cum_inst_mean" if use_avg else "cum_inst_mean"
    std_key = "avg_cum_inst_std" if use_avg else "cum_inst_std"
    ylabel = "avg cumulative instant regret" if use_avg else "cumulative instant regret"
    metric_tag = "IRavg" if use_avg else "IRcum"

    n_rounds = len(ts_aggs[keys_all[0]][mean_key])
    x = np.arange(1, n_rounds + 1)

    def plot_group(group_name: str, algos: List[str]) -> None:
        # baselines only appear in the dedicated baseline_comparison plots
        if group_name == "baseline_comparison":
            wanted = _dedupe_keep_order(algos + [b for b in PLOT_BASELINES if b not in set(algos)])
        else:
            wanted = _dedupe_keep_order(algos)
        wanted = [a for a in wanted if a in set(algos_present)]
        if not wanted:
            return
        band_label = "none" if not SHADE_REGRET_BANDS else ("95% CI" if str(BAND_MODE).lower() in {"ci95", "95", "95ci"} else "SEM")

        fig = plt.figure(figsize=(11, 5))
        ax = plt.gca()
        for algo in wanted:
            k = (game_name, opponent_group, algo)
            mean = ts_aggs[k][mean_key]
            std = ts_aggs[k][std_key]
            n = int(ts_aggs[k].get("n", np.array([1]))[0])
            ax.plot(x, mean, label=algo)
            if SHADE_REGRET_BANDS:
                band = _band_scale(BAND_MODE, std, n)
                lo = _clip_regret_band_lower(mean - band, use_log_scale=False)
                hi = mean + band
                ax.fill_between(x, lo, hi, alpha=0.18)

        ax.set_xlabel("round")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{_game_acronym(game_name)} vs {_opponent_acronym(opponent_group)} | {metric_tag} curve | {group_name} | shaded: {band_label}"
        )
        legend_right(ax, fig, fontsize="x-small")

        os.makedirs(outdir, exist_ok=True)
        fname = f"curve__{_game_acronym(game_name)}__{_opponent_acronym(opponent_group)}__{metric_tag}__{group_name}.png"
        fname = fname.replace("(", "").replace(")", "").replace("+", "p")
        _savefig(os.path.join(outdir, fname), fig=fig, dpi=200)
        plt.close(fig)

    for group_name, algos in ALGO_GROUPS.items():
        plot_group(group_name, algos)


def plot_external_regret_curves_grouped(
    ts_aggs: Dict[Tuple[str, str, str], Dict[str, np.ndarray]],
    game_name: str,
    opponent_group: str,
    outdir: str,
    use_cumulative: bool = False,
) -> None:
    """external-regret curves (avg or cumulative), grouped for readability."""
    if not PLOT_EXTERNAL_REGRET_CURVES:
        return

    # collect all algorithms available for this matchup
    keys_all = [(g, o, a) for (g, o, a) in ts_aggs.keys() if g == game_name and o == opponent_group]
    if not keys_all:
        return
    algos_present = sorted({a for (_, _, a) in keys_all})
    mean_key = "cum_regret_mean" if use_cumulative else "avg_regret_mean"
    std_key = "cum_regret_std" if use_cumulative else "avg_regret_std"
    ylabel = "cumulative external regret" if use_cumulative else "avg external regret"
    metric_tag = "ERcum" if use_cumulative else "ER"

    n_rounds = len(ts_aggs[keys_all[0]][mean_key])
    x = np.arange(1, n_rounds + 1)

    def plot_group(group_name: str, algos: List[str]) -> None:
        # baselines only appear in the dedicated baseline_comparison plots
        if group_name == "baseline_comparison":
            wanted = _dedupe_keep_order(algos + [b for b in PLOT_BASELINES if b not in set(algos)])
        else:
            wanted = _dedupe_keep_order(algos)
        wanted = [a for a in wanted if a in set(algos_present)]
        if not wanted:
            return

        band_label = "none" if not SHADE_REGRET_BANDS else ("95% CI" if str(BAND_MODE).lower() in {"ci95", "95", "95ci"} else "SEM")

        fig = plt.figure(figsize=(11, 5))
        ax = plt.gca()
        for algo in wanted:
            k = (game_name, opponent_group, algo)
            mean = ts_aggs[k][mean_key]
            std = ts_aggs[k][std_key]
            n = int(ts_aggs[k].get("n", np.array([1]))[0])
            ax.plot(x, mean, label=algo)
            if SHADE_REGRET_BANDS:
                band = _band_scale(BAND_MODE, std, n)
                use_log = (not use_cumulative) and bool(ER_AVG_USE_LOG_SCALE)
                lo = _clip_regret_band_lower(mean - band, use_log_scale=use_log)
                hi = mean + band
                ax.fill_between(x, lo, hi, alpha=0.18)

        ax.set_xlabel("round")
        ax.set_ylabel(ylabel)
        if (not use_cumulative) and ER_AVG_USE_LOG_SCALE:
            if ER_AVG_LOG_MODE == "log":
                ax.set_yscale("log")
            else:
                ax.set_yscale("symlog", linthresh=ER_AVG_SYMLOG_LINTHRESH)

        if ER_TAIL_INSET:
            start = int(max(0, np.floor((1.0 - ER_TAIL_INSET_FRAC) * float(n_rounds))))
            start = min(start, n_rounds - 1)
            axins = inset_axes(ax, width=ER_TAIL_INSET_SIZE, height=ER_TAIL_INSET_SIZE, loc=ER_TAIL_INSET_LOC, borderpad=1.5)
            for algo in wanted:
                k = (game_name, opponent_group, algo)
                mean = ts_aggs[k][mean_key]
                axins.plot(x[start:], mean[start:])
            axins.set_xlim(x[start], x[-1])
            if (not use_cumulative) and ER_AVG_USE_LOG_SCALE:
                if ER_AVG_LOG_MODE == "log":
                    axins.set_yscale("log")
                else:
                    axins.set_yscale("symlog", linthresh=ER_AVG_SYMLOG_LINTHRESH)
            axins.tick_params(labelsize=7)

        ax.set_title(
            f"{_game_acronym(game_name)} vs {_opponent_acronym(opponent_group)} | {metric_tag} curve | {group_name} | shaded: {band_label}"
        )
        legend_right(ax, fig, fontsize="x-small")

        os.makedirs(outdir, exist_ok=True)
        fname = f"curve__{_game_acronym(game_name)}__{_opponent_acronym(opponent_group)}__{metric_tag}__{group_name}.png"
        fname = fname.replace("(", "").replace(")", "").replace("+", "p")
        _savefig(os.path.join(outdir, fname), fig=fig, dpi=200)
        plt.close(fig)

    for group_name, algos in ALGO_GROUPS.items():
        plot_group(group_name, algos)


def plot_external_regret_curves_grouped_from_npz(
    npz_path: str,
    game_name: str,
    opponent_group: str,
    outdir: str,
    use_cumulative: bool = False,
) -> None:
    """load per-matchup curves from .npz and generate grouped plots."""
    if not PLOT_EXTERNAL_REGRET_CURVES:
        return

    cum, _seeds, algorithms, indices = load_curves_npz(npz_path)
    algorithms_list = [str(a) for a in algorithms.tolist()]
    x = np.asarray(indices, dtype=int) + 1  # rounds are 1-indexed in plots

    # derive avg external regret on the stored indices if requested
    if use_cumulative:
        y = np.asarray(cum, dtype=float)
        ylabel = "cumulative external regret"
        metric_tag = "ERcum"
    else:
        denom = (np.asarray(indices, dtype=float) + 1.0)[None, None, :]
        y = np.asarray(cum, dtype=float) / denom
        ylabel = "avg external regret"
        metric_tag = "ER"

    # aggregate over seeds: mean and stderr (ignore NaNs for missing runs)
    mean_by_algo = {}
    sem_by_algo = {}
    for j, algo in enumerate(algorithms_list):
        series = y[:, j, :]  # (n_seeds, n_points)
        mean = np.nanmean(series, axis=0)
        # stderr with NaN-aware n
        n_eff = np.sum(~np.isnan(series), axis=0).astype(float)
        std = np.nanstd(series, axis=0, ddof=1)
        sem = np.where(n_eff > 0, std / np.sqrt(np.maximum(1.0, n_eff)), 0.0)
        mean_by_algo[algo] = mean
        sem_by_algo[algo] = sem

    algos_present = [a for a in algorithms_list]

    def plot_group(group_name: str, algos: List[str]) -> None:
        # baselines only appear in the dedicated baseline_comparison plots
        if group_name == "baseline_comparison":
            wanted = _dedupe_keep_order(algos + [b for b in PLOT_BASELINES if b not in set(algos)])
        else:
            wanted = _dedupe_keep_order(algos)
        wanted = [a for a in wanted if a in set(algos_present)]
        if not wanted:
            return

        band_label = "none" if not SHADE_REGRET_BANDS else ("95% CI" if str(BAND_MODE).lower() in {"ci95", "95", "95ci"} else "SEM")
        z = CI95_Z if str(BAND_MODE).lower() in {"ci95", "95", "95ci"} else 1.0

        fig = plt.figure(figsize=(11, 5))
        ax = plt.gca()
        for algo in wanted:
            mean = mean_by_algo[algo]
            band = z * sem_by_algo[algo]
            ax.plot(x, mean, label=algo)
            if SHADE_REGRET_BANDS:
                use_log = (not use_cumulative) and bool(ER_AVG_USE_LOG_SCALE)
                lo = _clip_regret_band_lower(mean - band, use_log_scale=use_log)
                hi = mean + band
                ax.fill_between(x, lo, hi, alpha=0.18)

        ax.set_xlabel("round")
        ax.set_ylabel(ylabel)
        if (not use_cumulative) and ER_AVG_USE_LOG_SCALE:
            if ER_AVG_LOG_MODE == "log":
                ax.set_yscale("log")
            else:
                ax.set_yscale("symlog", linthresh=ER_AVG_SYMLOG_LINTHRESH)

        if ER_TAIL_INSET:
            # indices are 0-based; tail is based on rounds, then filtered to stored indices
            start_round = int(max(0, np.floor((1.0 - ER_TAIL_INSET_FRAC) * float(int(indices[-1]) + 1))))
            mask = (np.asarray(indices, dtype=int) >= start_round)
            if np.any(mask):
                axins = inset_axes(ax, width=ER_TAIL_INSET_SIZE, height=ER_TAIL_INSET_SIZE, loc=ER_TAIL_INSET_LOC)
                for algo in wanted:
                    mean = mean_by_algo[algo]
                    axins.plot(x[mask], mean[mask])
                axins.set_xlim(x[mask][0], x[mask][-1])
                if (not use_cumulative) and ER_AVG_USE_LOG_SCALE:
                    if ER_AVG_LOG_MODE == "log":
                        axins.set_yscale("log")
                    else:
                        axins.set_yscale("symlog", linthresh=ER_AVG_SYMLOG_LINTHRESH)
                axins.tick_params(labelsize=7)

        ax.set_title(
            f"{_game_acronym(game_name)} vs {_opponent_acronym(opponent_group)} | {metric_tag} curve | {group_name} | shaded: {band_label}"
        )
        legend_right(ax, fig, fontsize="x-small")

        os.makedirs(outdir, exist_ok=True)
        fname = f"curve__{_game_acronym(game_name)}__{_opponent_acronym(opponent_group)}__{metric_tag}__{group_name}.png"
        fname = fname.replace("(", "").replace(")", "").replace("+", "p")
        _savefig(os.path.join(outdir, fname), fig=fig, dpi=200)
        plt.close(fig)

    for group_name, algos in ALGO_GROUPS.items():
        plot_group(group_name, algos)


def write_report_txt(
    report_path: str,
    seeds: List[int],
    n_rounds: int,
    summary_rows: List[Dict[str, Any]],
) -> None:
    def fmt_ci95(mean: float, std: float, n: int) -> str:
        ci = float(_band_scale("ci95", np.asarray(float(std), dtype=float), int(n)))
        return f"{float(mean):.4f} ± {ci:.4f}"

    # index summary for easy lookup
    idx: Dict[Tuple[str, str, str], Dict[str, Any]] = {(r["game"], r["opponent_group"], r["algorithm"]): r for r in summary_rows}
    games = sorted(set(r["game"] for r in summary_rows))
    all_algorithms = sorted(set(r["algorithm"] for r in summary_rows))
    algos_known = [a for a in ALGORITHM_ORDER if a in set(all_algorithms)]
    algos_other = [a for a in all_algorithms if a not in set(ALGORITHM_ORDER)]
    algorithms_for_tables = algos_known + algos_other

    with open(report_path, "a") as f:
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SUMMARY (mean ± 95% CI over seeds)\n")
        f.write("=" * 80 + "\n")
        f.write("notes: primary metric is External Regret (best fixed action in hindsight).\n")
        f.write("       Instant regret is diagnostic only; equilibrium convergence is reflected by time-avg exploitability.\n")
        f.write(f"seeds={seeds}\n")
        f.write(f"n_rounds={n_rounds}\n")
        # tables use last-k windowed summaries; curves are stored in raw records
        f.write(f"algorithms={algorithms_for_tables}\n")
        f.write(f"opponents={OPPONENT_ORDER}\n\n")

        for game in games:
            f.write(f"## {game}\n\n")
            f.write("| Opponent | Algorithm | External Regret+ (avg lastK) | Instant Regret (avg lastK) | Time-Avg Exploitability (avg lastK) | Exploitability (last) | Entropy |\n")
            f.write("|:--|:--|--:|--:|--:|--:|--:|\n")
            for opp in OPPONENT_ORDER:
                for algo in algorithms_for_tables:
                    r = idx.get((game, opp, algo))
                    if r is None:
                        continue
                    n = int(r.get("n", 1))
                    extp_lastk = fmt_ci95(r["external_regret_avg_lastk_mean"], r["external_regret_avg_lastk_std"], n)
                    inst_lastk = fmt_ci95(r["instant_regret_avg_lastk_mean"], r["instant_regret_avg_lastk_std"], n)
                    ta_expl_lastk = fmt_ci95(r["time_avg_exploitability_lastk_mean"], r["time_avg_exploitability_lastk_std"], n)
                    expl_last = fmt_ci95(r["exploitability_mean"], r["exploitability_std"], n)
                    ent = fmt_ci95(r["entropy_mean"], r["entropy_std"], n)
                    f.write(f"| {opp} | {algo} | {extp_lastk} | {inst_lastk} | {ta_expl_lastk} | {expl_last} | {ent} |\n")
            f.write("\n")


def run_full_benchmark_multi_seed(
    seeds: List[int],
    n_rounds: int = 3000,
    outdir: Optional[str] = None,
    opponents: Optional[List[str]] = None,
    games: Optional[List[str]] = None,
    algorithm_configs: Optional[List[Tuple[str, str, Dict[str, Any]]]] = None,
    eval_window_k: int | None = None,
    eval_window_frac: float | None = None,
) -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if outdir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join(base_dir, "runs", f"benchmark_{ts}")
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    report_path = os.path.join(outdir, "benchmark_report.txt")
    records: List[Dict[str, Any]] = []

    with open(report_path, "w") as f:
        def write_line(s: str = "") -> None:
            f.write(s + "\n")
            f.flush()

        opponents = list(OPPONENTS if opponents is None else opponents)
        games = list(GAMES if games is None else games)
        algorithm_configs = list(get_algorithm_configs() if algorithm_configs is None else algorithm_configs)
        # default: use global frac unless caller overrides k/frac
        frac_to_use = EVAL_WINDOW_FRAC_DEFAULT if (eval_window_k is None and eval_window_frac is None) else eval_window_frac
        resolved_k = resolve_eval_window(n_rounds=n_rounds, k=eval_window_k, frac=frac_to_use)

        write_line("BENCHMARK RUN")
        write_line(f"started_at={time.strftime('%Y-%m-%d %H:%M:%S')}")
        write_line(f"n_rounds={n_rounds}")
        write_line(f"seeds={seeds}")
        write_line(f"games={games}")
        write_line(f"opponents={opponents}")
        write_line(f"n_algorithms={len(algorithm_configs)}")
        write_line(f"eval_window_k_resolved={resolved_k} (eval_window_k={eval_window_k}, eval_window_frac={eval_window_frac}, default_frac={EVAL_WINDOW_FRAC_DEFAULT})")
        write_line("")

        for seed in seeds:
            print(f"[seed {seed}] running...")
            write_line("=" * 80)
            write_line(f"seed={seed}")
            write_line("=" * 80)
            try:
                set_seed(seed)
                run_full_benchmark(
                    records,
                    n_rounds=n_rounds,
                    seed=seed,
                    log_fn=write_line,
                    verbose_headers=False,
                    opponents=opponents,
                    games=games,
                    algorithm_configs=algorithm_configs,
                    eval_window_k=resolved_k,
                    eval_window_frac=None,
                )
            except Exception as e:
                write_line(f"ERROR seed={seed}: {e!r}")
                write_line(traceback.format_exc())
                write_line("continuing...\n")
                continue

    # save artifacts (curves first; records still have arrays)
    if SAVE_CURVES:
        algo_names = [n for (n, _, _) in algorithm_configs]
        curves_meta = write_curves_npz_per_matchup(
            records=records,
            outdir=outdir,
            seeds=seeds,
            algorithms=algo_names,
            n_rounds=n_rounds,
        )
        # enrich raw records with scalar linkage fields for easy joining later
        seed_to_i = {int(s): i for i, s in enumerate(seeds)}
        algo_to_j = {a: j for j, a in enumerate(algo_names)}
        for r in records:
            m = curves_meta.get((r["game"], r["opponent_group"]))
            if m is None:
                continue
            r["curves_npz_path"] = m["path"]
            r["curve_seed_index"] = int(seed_to_i.get(int(r["seed"]), -1))
            r["curve_algo_index"] = int(algo_to_j.get(r["algorithm"], -1))

    write_results_raw_csv(records, os.path.join(outdir, "results_raw.csv"))
    summary_rows, ts_aggs = aggregate_results(records, n_rounds=n_rounds)
    write_results_summary_csv(summary_rows, os.path.join(outdir, "results_summary.csv"))

    # plots: bars per game
    for game_name in sorted(set(r["game"] for r in summary_rows)):
        for metric in ["instant_regret_avg", "exploitability", "time_avg_exploitability", "external_regret_pos_avg"]:
            outpath = os.path.join(plots_dir, f"bar__{_game_acronym(game_name)}__{_metric_acronym(metric)}.png")
            plot_bar_with_error(summary_rows, game_name, metric, outpath)

    # plots: curves per (game, opponent)
    # sanity: ensure every algorithm is covered by at least one grouped curve plot
    all_algorithms_present = sorted(set(r["algorithm"] for r in summary_rows))
    _assert_plot_group_coverage(all_algorithms_present)

    for game_name in sorted(set(r["game"] for r in summary_rows)):
        for opp in OPPONENT_ORDER:
            # external regret curves (primary)
            plot_external_regret_curves_grouped(ts_aggs, game_name, opp, plots_dir, use_cumulative=False)
            plot_external_regret_curves_grouped(ts_aggs, game_name, opp, plots_dir, use_cumulative=True)
            # instant regret curves (optional diagnostics)
            if PLOT_INSTANT_REGRET_CURVES:
                plot_instant_regret_curves_grouped(ts_aggs, game_name, opp, plots_dir, use_avg=True)

    write_report_txt(report_path, seeds=seeds, n_rounds=n_rounds, summary_rows=summary_rows)
    return outdir


if __name__ == "__main__":
    outdir = run_full_benchmark_multi_seed(seeds=SEEDS, n_rounds=N_ROUNDS)
    print(f"\nWrote benchmark artifacts to: {outdir}")