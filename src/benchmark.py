from __future__ import annotations

import csv
import os
import random
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
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


OPPONENT_ORDER = ["Deterministic", "Uniform Random", "Biased", "MW Opponent", "EXP3 Opponent"]
ALGORITHM_ORDER = ["MultiplicativeWeights", "EXP3", "PolicyGradient"]
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
    "exploitability",
    "entropy",
    "final_policy",
]


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


def extract_metrics(results: Dict, n_rounds: int, game: RepeatedGame, algo_role: str) -> Dict[str, Any]:
    final_policy = results['policies'][-1]
    final_exploit = results['exploitabilities'][-1]
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

    # sanity checks
    if inst_per_round.size and float(inst_per_round.min()) < -1e-8:
        raise ValueError(f"instant regret per round should be >= 0; min={inst_per_round.min()}")
    if inst_per_round.shape != (n_rounds,):
        raise ValueError(f"instant regret ts length mismatch: {inst_per_round.shape} vs ({n_rounds},)")
    if avg_external_running.shape != (n_rounds,):
        raise ValueError(f"avg external regret ts length mismatch: {avg_external_running.shape} vs ({n_rounds},)")
    if exploitability_ts.shape != (n_rounds,):
        raise ValueError(f"exploitability ts length mismatch: {exploitability_ts.shape} vs ({n_rounds},)")
    if rewards.shape != (n_rounds,):
        raise ValueError(f"reward ts length mismatch: {rewards.shape} vs ({n_rounds},)")

    scalars = np.array([external_avg, external_avg_pos, instant_avg, final_exploit, final_entropy], dtype=float)
    if not np.isfinite(scalars).all():
        raise ValueError(f"Non-finite metrics detected: {scalars}")

    return {
        # scalars
        "external_avg": float(external_avg),
        "external_avg_pos": float(external_avg_pos),
        "instant_avg": float(instant_avg),
        "final_exploit": float(final_exploit),
        "final_entropy": float(final_entropy),
        "avg_external_running_last_100": float(avg_external_running_last_100),
        "avg_instant_last_100": float(avg_instant_last_100),
        "avg_exploit_last_100": float(avg_exploit_last_100),
        # arrays
        "inst_per_round": inst_per_round.astype(float),
        "avg_external_running": avg_external_running.astype(float),
        "exploitability_ts": exploitability_ts.astype(float),
        "rewards": np.asarray(rewards, dtype=float),
        "actions": actions,
        "opp_actions": opp_actions,
        # policy as json-friendly
        "final_policy": np.asarray(final_policy, dtype=float).tolist(),
    }


def print_metrics(results: Dict, n_rounds: int, game: RepeatedGame, algo_role: str):
    m = extract_metrics(results, n_rounds, game, algo_role)
    final_policy = np.asarray(m["final_policy"], dtype=float)

    print(f"Final Policy:        {np.array2string(final_policy, precision=4, suppress_small=True)}")
    print(f"Avg External Regret (best fixed; can be negative): {m['external_avg']:.6f}")
    print(f"Avg External Regret (clipped at 0):               {m['external_avg_pos']:.6f}")
    print(f"Avg Instant Regret (best per-round; >= 0):        {m['instant_avg']:.6f}")
    print(f"Exploitability:      {m['final_exploit']:.6f}")
    print(f"Policy Entropy:      {m['final_entropy']:.6f}")

    last_100 = min(100, n_rounds)
    print(f"Avg External Regret (running avg; last {last_100}): {m['avg_external_running_last_100']:.6f}")
    print(f"Avg Instant Regret (last {last_100} rounds):        {m['avg_instant_last_100']:.6f}")
    print(f"Avg Exploit (last {last_100}):                      {m['avg_exploit_last_100']:.6f}")


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
                opponent_group: Optional[str] = None):

    def log(msg: str = "") -> None:
        if log_fn is not None:
            log_fn(msg)
        else:
            print(msg)

    algorithm_name = (
        "MultiplicativeWeights" if isinstance(algo, MultiplicativeWeights)
        else "EXP3" if isinstance(algo, EXP3)
        else "PolicyGradient" if isinstance(algo, PolicyGradient)
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

    metrics = extract_metrics(results, n_rounds, game=game, algo_role=algo_role)

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
        "entropy": metrics["final_entropy"],
        "final_policy": metrics["final_policy"],
        # time series (store only what we plot/aggregate)
        "instant_regret_per_round": metrics["inst_per_round"],
    }

    if verbose_headers and log_fn is None:
        # preserve current interactive printing behavior
        print_metrics(results, n_rounds, game=game, algo_role=algo_role)
    elif log_fn is not None:
        # compact log line for reports
        log(
            f"{game.name} | {opponent_group} | {algorithm_name} | "
            f"inst_avg={record['instant_regret_avg']:.6f} "
            f"ext_avg={record['external_regret_avg']:.6f} "
            f"ext_pos_avg={record['external_regret_pos_avg']:.6f} "
            f"exploit={record['exploitability']:.6f} "
            f"entropy={record['entropy']:.6f}"
        )

    return record if collect else None


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
):
    games = [MatchingPennies(), RockPaperScissors()]
    
    for game in games:
        if verbose_headers:
            print("\n" + "=" * 80)
            print(f"GAME: {game.name}")
            print("=" * 80)
        
        benchmark_vs_deterministic(game, records, n_rounds, seed=seed, log_fn=log_fn, verbose_headers=verbose_headers)
        benchmark_vs_uniform(game, records, n_rounds, seed=seed, log_fn=log_fn, verbose_headers=verbose_headers)
        benchmark_vs_biased(game, records, n_rounds, seed=seed, log_fn=log_fn, verbose_headers=verbose_headers)
        benchmark_vs_mw(game, records, n_rounds, seed=seed, log_fn=log_fn, verbose_headers=verbose_headers)
        benchmark_vs_exp3(game, records, n_rounds, seed=seed, log_fn=log_fn, verbose_headers=verbose_headers)


def _strip_arrays_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in record.items():
        if isinstance(v, np.ndarray):
            continue
        if k in {"instant_regret_per_round"}:
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
    fieldnames = RAW_CSV_FIELDNAMES
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def aggregate_results(records: List[Dict[str, Any]], n_rounds: int) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str], Dict[str, np.ndarray]]]:
    # summary rows: one per (game, opponent_group, algorithm)
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in records:
        key = (r["game"], r["opponent_group"], r["algorithm"])
        groups.setdefault(key, []).append(r)

    metrics = ["instant_regret_avg", "external_regret_avg", "external_regret_pos_avg", "exploitability", "entropy"]
    summary_rows: List[Dict[str, Any]] = []
    ts_aggs: Dict[Tuple[str, str, str], Dict[str, np.ndarray]] = {}

    for (game, opp_group, algo), rs in groups.items():
        row: Dict[str, Any] = {"game": game, "opponent_group": opp_group, "algorithm": algo, "n": len(rs)}
        for m in metrics:
            vals = np.asarray([float(x[m]) for x in rs], dtype=float)
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(rs) > 1 else 0.0
        summary_rows.append(row)

        inst = np.stack([np.asarray(x["instant_regret_per_round"], dtype=float) for x in rs], axis=0)
        if inst.shape[1] != n_rounds:
            raise ValueError(f"n_rounds mismatch in instant regret ts for {(game, opp_group, algo)}: {inst.shape}")
        cum = np.cumsum(inst, axis=1)
        avg_cum = cum / (np.arange(n_rounds, dtype=float) + 1.0)[None, :]
        ddof = 1 if inst.shape[0] > 1 else 0

        ts_aggs[(game, opp_group, algo)] = {
            "inst_per_round_mean": inst.mean(axis=0),
            "inst_per_round_std": inst.std(axis=0, ddof=ddof) if ddof == 1 else np.zeros(n_rounds, dtype=float),
            "cum_inst_mean": cum.mean(axis=0),
            "cum_inst_std": cum.std(axis=0, ddof=ddof) if ddof == 1 else np.zeros(n_rounds, dtype=float),
            "avg_cum_inst_mean": avg_cum.mean(axis=0),
            "avg_cum_inst_std": avg_cum.std(axis=0, ddof=ddof) if ddof == 1 else np.zeros(n_rounds, dtype=float),
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

    opponents = [o for o in OPPONENT_ORDER if any(r["opponent_group"] == o for r in rows)]
    algos = [a for a in ALGORITHM_ORDER if any(r["algorithm"] == a for r in rows)]

    x = np.arange(len(opponents))
    width = 0.25 if len(algos) <= 3 else 0.8 / max(1, len(algos))

    plt.figure(figsize=(max(8, len(opponents) * 1.6), 4.5))
    for i, algo in enumerate(algos):
        y = [means.get((opp, algo), np.nan) for opp in opponents]
        e = [stds.get((opp, algo), 0.0) for opp in opponents]
        plt.bar(x + (i - (len(algos) - 1) / 2) * width, y, width, yerr=e, capsize=3, label=algo)

    plt.xticks(x, opponents, rotation=20, ha="right")
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"{game_name}: {metric} (mean±std over seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_regret_curves(ts_aggs: Dict[Tuple[str, str, str], Dict[str, np.ndarray]], game_name: str, opponent_group: str, outpath: str, use_avg: bool = True) -> None:
    keys = [(g, o, a) for (g, o, a) in ts_aggs.keys() if g == game_name and o == opponent_group]
    if not keys:
        return
    n_rounds = len(ts_aggs[keys[0]]["avg_cum_inst_mean"] if use_avg else ts_aggs[keys[0]]["cum_inst_mean"])
    x = np.arange(1, n_rounds + 1)

    plt.figure(figsize=(8, 4.5))
    for algo in ALGORITHM_ORDER:
        k = (game_name, opponent_group, algo)
        if k not in ts_aggs:
            continue
        if use_avg:
            mean = ts_aggs[k]["avg_cum_inst_mean"]
            std = ts_aggs[k]["avg_cum_inst_std"]
            ylabel = "avg cumulative instant regret"
        else:
            mean = ts_aggs[k]["cum_inst_mean"]
            std = ts_aggs[k]["cum_inst_std"]
            ylabel = "cumulative instant regret"
        plt.plot(x, mean, label=algo)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("round")
    plt.ylabel(ylabel)
    plt.title(f"{game_name}: {opponent_group} ({'avg' if use_avg else 'cum'})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def write_report_txt(
    report_path: str,
    seeds: List[int],
    n_rounds: int,
    summary_rows: List[Dict[str, Any]],
) -> None:
    def fmt(ms: Tuple[float, float]) -> str:
        m, s = ms
        return f"{m:.4f} ± {s:.4f}"

    # index summary for easy lookup
    idx: Dict[Tuple[str, str, str], Dict[str, Any]] = {(r["game"], r["opponent_group"], r["algorithm"]): r for r in summary_rows}
    games = sorted(set(r["game"] for r in summary_rows))

    with open(report_path, "a") as f:
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SUMMARY (mean ± std over seeds)\n")
        f.write("=" * 80 + "\n")
        f.write(f"seeds={seeds}\n")
        f.write(f"n_rounds={n_rounds}\n")
        f.write(f"algorithms={ALGORITHM_ORDER}\n")
        f.write(f"opponents={OPPONENT_ORDER}\n\n")

        for game in games:
            f.write(f"## {game}\n\n")
            f.write("| Opponent | Algorithm | Instant Regret (avg) | Exploitability | Entropy | External Regret+ (avg) |\n")
            f.write("|:--|:--|--:|--:|--:|--:|\n")
            for opp in OPPONENT_ORDER:
                for algo in ALGORITHM_ORDER:
                    r = idx.get((game, opp, algo))
                    if r is None:
                        continue
                    inst = fmt((r["instant_regret_avg_mean"], r["instant_regret_avg_std"]))
                    expl = fmt((r["exploitability_mean"], r["exploitability_std"]))
                    ent = fmt((r["entropy_mean"], r["entropy_std"]))
                    extp = fmt((r["external_regret_pos_avg_mean"], r["external_regret_pos_avg_std"]))
                    f.write(f"| {opp} | {algo} | {inst} | {expl} | {ent} | {extp} |\n")
            f.write("\n")


def run_full_benchmark_multi_seed(seeds: List[int], n_rounds: int = 3000, outdir: Optional[str] = None) -> str:
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

        write_line("BENCHMARK RUN")
        write_line(f"started_at={time.strftime('%Y-%m-%d %H:%M:%S')}")
        write_line(f"n_rounds={n_rounds}")
        write_line(f"seeds={seeds}")
        write_line("")

        for seed in seeds:
            print(f"[seed {seed}] running...")
            write_line("=" * 80)
            write_line(f"seed={seed}")
            write_line("=" * 80)
            try:
                set_seed(seed)
                run_full_benchmark(records, n_rounds=n_rounds, seed=seed, log_fn=write_line, verbose_headers=False)
            except Exception as e:
                write_line(f"ERROR seed={seed}: {e!r}")
                write_line(traceback.format_exc())
                write_line("continuing...\n")
                continue

    # save raw + aggregated artifacts
    write_results_raw_csv(records, os.path.join(outdir, "results_raw.csv"))
    summary_rows, ts_aggs = aggregate_results(records, n_rounds=n_rounds)
    write_results_summary_csv(summary_rows, os.path.join(outdir, "results_summary.csv"))

    # plots: bars per game
    for game_name in sorted(set(r["game"] for r in summary_rows)):
        plot_bar_with_error(summary_rows, game_name, "instant_regret_avg", os.path.join(plots_dir, f"bar__{game_name}__instant_regret_avg.png"))
        plot_bar_with_error(summary_rows, game_name, "exploitability", os.path.join(plots_dir, f"bar__{game_name}__exploitability.png"))
        plot_bar_with_error(summary_rows, game_name, "external_regret_pos_avg", os.path.join(plots_dir, f"bar__{game_name}__external_regret_pos_avg.png"))

    # plots: curves per (game, opponent)
    for game_name in sorted(set(r["game"] for r in summary_rows)):
        for opp in OPPONENT_ORDER:
            outpath = os.path.join(plots_dir, f"curve__{game_name}__{opp}__avg_cum_inst_regret.png".replace(" ", "_").replace("(", "").replace(")", ""))
            plot_regret_curves(ts_aggs, game_name, opp, outpath, use_avg=True)

    write_report_txt(report_path, seeds=seeds, n_rounds=n_rounds, summary_rows=summary_rows)
    return outdir


if __name__ == "__main__":
    outdir = run_full_benchmark_multi_seed(seeds=list(range(30)), n_rounds=10000)
    print(f"\nWrote benchmark artifacts to: {outdir}")