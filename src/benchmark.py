import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from pathlib import Path
import re
from games import MatchingPennies, RockPaperScissors, RepeatedGame
from opponents import (
    UniformRandomOpponent, 
    DeterministicOpponent, 
    FixedBiasedOpponent,
    MultiplicativeWeightsOpponent,
    EXP3Opponent
)
from classical_algorithms import MultiplicativeWeights, EXP3, run_algorithm
from policy_gradient import PolicyGradient, PolicyGradientNoRegret, run_policy_gradient


def print_section_header(title: str):
    print(f"\n{title}")
    print("-" * len(title))


def print_algorithm_config(algo_name: str, params: Dict):
    print(f"{algo_name}:")
    for key, value in params.items():
        print(f"  {key}: {value}")


def print_metrics(results: Dict, n_rounds: int):
    final_policy = results['policies'][-1]
    final_avg_regret = results['average_regrets'][-1]
    final_exploit = results['exploitabilities'][-1]
    final_entropy = -np.sum(final_policy * np.log(final_policy + 1e-10))
    
    print(f"Final Policy:        {np.array2string(final_policy, precision=4, suppress_small=True)}")
    print(f"Average Regret:      {final_avg_regret:.6f}")
    print(f"Exploitability:      {final_exploit:.6f}")
    print(f"Policy Entropy:      {final_entropy:.6f}")
    
    last_100 = min(100, n_rounds)
    avg_regret_last_100 = np.mean(results['average_regrets'][-last_100:])
    avg_exploit_last_100 = np.mean(results['exploitabilities'][-last_100:])
    print(f"Avg Regret (last {last_100}): {avg_regret_last_100:.6f}")
    print(f"Avg Exploit (last {last_100}): {avg_exploit_last_100:.6f}")


def _sanitize_filename(s: str) -> str:
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]", "", s)
    return s


def _short_name_for_path(name: str) -> str:
    s = str(name).strip()
    s_lower = s.lower()

    if "multiplicative weights" in s_lower:
        return "mw"
    if "pg_no_regret" in s_lower:
        return "pg_no_regret"
    if "policygradient" in s_lower or "policy gradient" in s_lower:
        return "pg"
    if "uniform random" in s_lower:
        return "uniform"

    return name


def _default_figures_dir() -> Path:
    # project_root/figures (benchmark.py lives in project_root/src)
    return Path(__file__).resolve().parent.parent / "figures"


def plot_regret_two_players(
    game: RepeatedGame,
    results: Dict,
    title: str = None,
    cumulative: bool = True,
    save_path: str = None,
    show: bool = True,
):
    if cumulative:
        row_key = 'row_cumulative_regrets'
        col_key = 'col_cumulative_regrets'
        ylabel = "Cumulative Regret"
    else:
        row_key = 'row_average_regrets'
        col_key = 'col_average_regrets'
        ylabel = "Average Regret"

    if row_key not in results or col_key not in results:
        raise KeyError(
            f"Missing {row_key!r} / {col_key!r} in results. "
            "Re-run with the updated code in classical_algorithms.py / policy_gradient.py."
        )

    row_y = np.asarray(results[row_key], dtype=float)
    col_y = np.asarray(results[col_key], dtype=float)

    if title is None:
        algo_name = results.get('algorithm', 'Algorithm')
        opp_name = results.get('opponent', 'Opponent')
        title = f"{game.name}: {algo_name} vs {opp_name} Regret Over Time"

    algo_name = results.get('algorithm', 'Algorithm')
    opp_name = results.get('opponent', 'Opponent')
    algo_role = results.get('algorithm_role', 'row')
    if algo_role == 'row':
        row_label = algo_name
        col_label = opp_name
    else:
        row_label = opp_name
        col_label = algo_name

    x = np.arange(1, len(row_y) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x, row_y, label=row_label)
    plt.plot(x, col_y, label=col_label)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def run_matchup(game: RepeatedGame, 
                algo, 
                opponent,
                n_rounds: int,
                algo_role: str = "row",
                algo_params: Dict = None,
                opp_params: Dict = None,
                verbose: bool = False,
                plot_regret: bool = False,
                plot_cumulative_regret: bool = False,
                plot_save_path: str = None,
                plot_show: bool = True,
                figures_dir: str = None,
                save_plots: bool = False,
                plot_both: bool = True):
    
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
    elif isinstance(algo, (PolicyGradient, PolicyGradientNoRegret)):
        results = run_policy_gradient(algo, opponent, n_rounds, verbose=verbose, algorithm_role=algo_role)
    else:
        raise ValueError(f"Unknown algorithm type: {type(algo)}")

    results['algorithm_role'] = algo_role
    
    print_metrics(results, n_rounds)

    if plot_regret:
        fig_dir = Path(figures_dir) if figures_dir is not None else _default_figures_dir()
        if save_plots:
            game_name = _sanitize_filename(getattr(game, 'name', 'game'))
            algo_name_raw = results.get('algorithm', algo.__class__.__name__)
            opp_name_raw = results.get('opponent', getattr(opponent, 'name', 'opponent'))

            algo_name = _sanitize_filename(_short_name_for_path(algo_name_raw))
            opp_name = _sanitize_filename(_short_name_for_path(opp_name_raw))

            match_dir = fig_dir / game_name / opp_name / algo_name / "regret_plots"
            match_dir.mkdir(parents=True, exist_ok=True)

            if plot_both:
                plot_regret_two_players(
                    game,
                    results,
                    cumulative=False,
                    save_path=str(match_dir / "avg_regret.png"),
                    show=False,
                )
                plot_regret_two_players(
                    game,
                    results,
                    cumulative=True,
                    save_path=str(match_dir / "cum_regret.png"),
                    show=False,
                )
            else:
                suffix = "cum_regret" if plot_cumulative_regret else "avg_regret"
                plot_regret_two_players(
                    game,
                    results,
                    cumulative=plot_cumulative_regret,
                    save_path=str(match_dir / f"{suffix}.png"),
                    show=False,
                )
        else:
            plot_regret_two_players(
                game,
                results,
                cumulative=plot_cumulative_regret,
                save_path=plot_save_path,
                show=plot_show,
            )
    
    return results


def benchmark_vs_deterministic(game: RepeatedGame, n_rounds: int = 3000, plot_regret: bool = False, figures_dir: str = None, save_plots: bool = False):
    print_section_header(f"BENCHMARK: {game.name} vs Deterministic Opponent")
    
    det_action = 0
    opponent_params = {"action": game.action_names[det_action]}
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    det = DeterministicOpponent(game, det_action)
    run_matchup(game, mw, det, n_rounds, "row", 
                {"eta": 0.1}, opponent_params,
                plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    det = DeterministicOpponent(game, det_action)
    run_matchup(game, exp3, det, n_rounds, "row",
                {"eta": 0.1, "gamma": 0.1}, opponent_params,
                plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    det = DeterministicOpponent(game, det_action)
    run_matchup(
        game,
        pg,
        det,
        n_rounds,
        "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01},
        opponent_params,
        plot_regret=plot_regret,
        plot_cumulative_regret=False,
        figures_dir=figures_dir,
        save_plots=save_plots,
    )

    pg_nr = PolicyGradientNoRegret(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    det = DeterministicOpponent(game, det_action)
    run_matchup(
        game,
        pg_nr,
        det,
        n_rounds,
        "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01},
        opponent_params,
        plot_regret=plot_regret,
        plot_cumulative_regret=False,
        figures_dir=figures_dir,
        save_plots=save_plots,
    )


def benchmark_vs_uniform(game: RepeatedGame, n_rounds: int = 3000, plot_regret: bool = False, figures_dir: str = None, save_plots: bool = False):
    print_section_header(f"BENCHMARK: {game.name} vs Uniform Random")
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    uniform = UniformRandomOpponent(game)
    run_matchup(game, mw, uniform, n_rounds, "row", {"eta": 0.1}, {},
                plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    uniform = UniformRandomOpponent(game)
    run_matchup(game, exp3, uniform, n_rounds, "row",
                {"eta": 0.1, "gamma": 0.1}, {},
                plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    uniform = UniformRandomOpponent(game)
    run_matchup(
        game,
        pg,
        uniform,
        n_rounds,
        "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01},
        {},
        plot_regret=plot_regret,
        plot_cumulative_regret=False,
        figures_dir=figures_dir,
        save_plots=save_plots,
    )

    pg_nr = PolicyGradientNoRegret(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    uniform = UniformRandomOpponent(game)
    run_matchup(
        game,
        pg_nr,
        uniform,
        n_rounds,
        "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01},
        {},
        plot_regret=plot_regret,
        plot_cumulative_regret=False,
        figures_dir=figures_dir,
        save_plots=save_plots,
    )


def benchmark_vs_biased(game: RepeatedGame, n_rounds: int = 3000, plot_regret: bool = False, figures_dir: str = None, save_plots: bool = False):
    print_section_header(f"BENCHMARK: {game.name} vs Biased Opponent")
    
    if game.n_actions == 2:
        probs = np.array([0.7, 0.3])
    else:
        probs = np.array([0.5, 0.3, 0.2])
    
    opponent_params = {"probabilities": probs}
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    biased = FixedBiasedOpponent(game, probs)
    run_matchup(game, mw, biased, n_rounds, "row", {"eta": 0.1}, opponent_params,
                plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    biased = FixedBiasedOpponent(game, probs)
    run_matchup(game, exp3, biased, n_rounds, "row",
                {"eta": 0.1, "gamma": 0.1}, opponent_params,
                plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    biased = FixedBiasedOpponent(game, probs)
    run_matchup(
        game,
        pg,
        biased,
        n_rounds,
        "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01},
        opponent_params,
        plot_regret=plot_regret,
        plot_cumulative_regret=False,
        figures_dir=figures_dir,
        save_plots=save_plots,
    )

    pg_nr = PolicyGradientNoRegret(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    biased = FixedBiasedOpponent(game, probs)
    run_matchup(
        game,
        pg_nr,
        biased,
        n_rounds,
        "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01},
        opponent_params,
        plot_regret=plot_regret,
        plot_cumulative_regret=False,
        figures_dir=figures_dir,
        save_plots=save_plots,
    )


def benchmark_vs_mw(game: RepeatedGame, n_rounds: int = 3000, plot_regret: bool = False, figures_dir: str = None, save_plots: bool = False):
    print_section_header(f"BENCHMARK: {game.name} vs MW Opponent")
    
    opponent_params = {"eta": 0.1}
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    mw_opp = MultiplicativeWeightsOpponent(game, eta=0.1, role="col")
    run_matchup(game, mw, mw_opp, n_rounds, "row", {"eta": 0.1}, opponent_params,
                plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    mw_opp = MultiplicativeWeightsOpponent(game, eta=0.1, role="col")
    run_matchup(game, exp3, mw_opp, n_rounds, "row",
                {"eta": 0.1, "gamma": 0.1}, opponent_params,
                plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    mw_opp = MultiplicativeWeightsOpponent(game, eta=0.1, role="col")
    run_matchup(
        game,
        pg,
        mw_opp,
        n_rounds,
        "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01},
        opponent_params,
        plot_regret=plot_regret,
        plot_cumulative_regret=False,
        figures_dir=figures_dir,
        save_plots=save_plots,
    )

    pg_nr = PolicyGradientNoRegret(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    mw_opp = MultiplicativeWeightsOpponent(game, eta=0.1, role="col")
    run_matchup(
        game,
        pg_nr,
        mw_opp,
        n_rounds,
        "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01},
        opponent_params,
        plot_regret=plot_regret,
        plot_cumulative_regret=False,
        figures_dir=figures_dir,
        save_plots=save_plots,
    )


def benchmark_vs_exp3(game: RepeatedGame, n_rounds: int = 3000, plot_regret: bool = False, figures_dir: str = None, save_plots: bool = False):
    print_section_header(f"BENCHMARK: {game.name} vs EXP3 Opponent")
    
    opponent_params = {"eta": 0.1, "gamma": 0.1}
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    exp3_opp = EXP3Opponent(game, eta=0.1, gamma=0.1, role="col")
    run_matchup(game, mw, exp3_opp, n_rounds, "row", {"eta": 0.1}, opponent_params,
                plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    exp3_opp = EXP3Opponent(game, eta=0.1, gamma=0.1, role="col")
    run_matchup(game, exp3, exp3_opp, n_rounds, "row",
                {"eta": 0.1, "gamma": 0.1}, opponent_params,
                plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    exp3_opp = EXP3Opponent(game, eta=0.1, gamma=0.1, role="col")
    run_matchup(
        game,
        pg,
        exp3_opp,
        n_rounds,
        "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01},
        opponent_params,
        plot_regret=plot_regret,
        plot_cumulative_regret=False,
        figures_dir=figures_dir,
        save_plots=save_plots,
    )

    pg_nr = PolicyGradientNoRegret(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    exp3_opp = EXP3Opponent(game, eta=0.1, gamma=0.1, role="col")
    run_matchup(
        game,
        pg_nr,
        exp3_opp,
        n_rounds,
        "row",
        {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01},
        opponent_params,
        plot_regret=plot_regret,
        plot_cumulative_regret=False,
        figures_dir=figures_dir,
        save_plots=save_plots,
    )


def run_full_benchmark(n_rounds: int = 3000, plot_regret: bool = False, figures_dir: str = None, save_plots: bool = False):
    games = [MatchingPennies(), RockPaperScissors()]
    
    for game in games:
        print("\n" + "=" * 80)
        print(f"GAME: {game.name}")
        print("=" * 80)
        
        benchmark_vs_deterministic(game, n_rounds, plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)
        benchmark_vs_uniform(game, n_rounds, plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)
        benchmark_vs_biased(game, n_rounds, plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)
        benchmark_vs_mw(game, n_rounds, plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)
        benchmark_vs_exp3(game, n_rounds, plot_regret=plot_regret, figures_dir=figures_dir, save_plots=save_plots)


if __name__ == "__main__":
    np.random.seed(42)
    run_full_benchmark(n_rounds=3000, plot_regret=True, save_plots=True)