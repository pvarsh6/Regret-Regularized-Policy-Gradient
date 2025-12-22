import numpy as np
from typing import Dict, List, Tuple
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


def run_matchup(game: RepeatedGame, 
                algo, 
                opponent,
                n_rounds: int,
                algo_role: str = "row",
                algo_params: Dict = None,
                opp_params: Dict = None,
                verbose: bool = False):
    
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
    
    print_metrics(results, n_rounds)
    
    return results


def benchmark_vs_deterministic(game: RepeatedGame, n_rounds: int = 3000):
    print_section_header(f"BENCHMARK: {game.name} vs Deterministic Opponent")
    
    det_action = 0
    opponent_params = {"action": game.action_names[det_action]}
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    det = DeterministicOpponent(game, det_action)
    run_matchup(game, mw, det, n_rounds, "row", 
                {"eta": 0.1}, opponent_params)
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    det = DeterministicOpponent(game, det_action)
    run_matchup(game, exp3, det, n_rounds, "row",
                {"eta": 0.1, "gamma": 0.1}, opponent_params)
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    det = DeterministicOpponent(game, det_action)
    run_matchup(game, pg, det, n_rounds, "row",
                {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01}, opponent_params)


def benchmark_vs_uniform(game: RepeatedGame, n_rounds: int = 3000):
    print_section_header(f"BENCHMARK: {game.name} vs Uniform Random")
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    uniform = UniformRandomOpponent(game)
    run_matchup(game, mw, uniform, n_rounds, "row", {"eta": 0.1}, {})
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    uniform = UniformRandomOpponent(game)
    run_matchup(game, exp3, uniform, n_rounds, "row",
                {"eta": 0.1, "gamma": 0.1}, {})
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    uniform = UniformRandomOpponent(game)
    run_matchup(game, pg, uniform, n_rounds, "row",
                {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01}, {})


def benchmark_vs_biased(game: RepeatedGame, n_rounds: int = 3000):
    print_section_header(f"BENCHMARK: {game.name} vs Biased Opponent")
    
    if game.n_actions == 2:
        probs = np.array([0.7, 0.3])
    else:
        probs = np.array([0.5, 0.3, 0.2])
    
    opponent_params = {"probabilities": probs}
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    biased = FixedBiasedOpponent(game, probs)
    run_matchup(game, mw, biased, n_rounds, "row", {"eta": 0.1}, opponent_params)
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    biased = FixedBiasedOpponent(game, probs)
    run_matchup(game, exp3, biased, n_rounds, "row",
                {"eta": 0.1, "gamma": 0.1}, opponent_params)
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    biased = FixedBiasedOpponent(game, probs)
    run_matchup(game, pg, biased, n_rounds, "row",
                {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01}, opponent_params)


def benchmark_vs_mw(game: RepeatedGame, n_rounds: int = 3000):
    print_section_header(f"BENCHMARK: {game.name} vs MW Opponent")
    
    opponent_params = {"eta": 0.1}
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    mw_opp = MultiplicativeWeightsOpponent(game, eta=0.1, role="col")
    run_matchup(game, mw, mw_opp, n_rounds, "row", {"eta": 0.1}, opponent_params)
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    mw_opp = MultiplicativeWeightsOpponent(game, eta=0.1, role="col")
    run_matchup(game, exp3, mw_opp, n_rounds, "row",
                {"eta": 0.1, "gamma": 0.1}, opponent_params)
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    mw_opp = MultiplicativeWeightsOpponent(game, eta=0.1, role="col")
    run_matchup(game, pg, mw_opp, n_rounds, "row",
                {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01}, opponent_params)


def benchmark_vs_exp3(game: RepeatedGame, n_rounds: int = 3000):
    print_section_header(f"BENCHMARK: {game.name} vs EXP3 Opponent")
    
    opponent_params = {"eta": 0.1, "gamma": 0.1}
    
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    exp3_opp = EXP3Opponent(game, eta=0.1, gamma=0.1, role="col")
    run_matchup(game, mw, exp3_opp, n_rounds, "row", {"eta": 0.1}, opponent_params)
    
    exp3 = EXP3(game, eta=0.1, gamma=0.1, role="row")
    exp3_opp = EXP3Opponent(game, eta=0.1, gamma=0.1, role="col")
    run_matchup(game, exp3, exp3_opp, n_rounds, "row",
                {"eta": 0.1, "gamma": 0.1}, opponent_params)
    
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    exp3_opp = EXP3Opponent(game, eta=0.1, gamma=0.1, role="col")
    run_matchup(game, pg, exp3_opp, n_rounds, "row",
                {"hidden_size": 64, "lr": 1e-2, "entropy_coef": 0.01}, opponent_params)


def run_full_benchmark(n_rounds: int = 3000):
    games = [MatchingPennies(), RockPaperScissors()]
    
    for game in games:
        print("\n" + "=" * 80)
        print(f"GAME: {game.name}")
        print("=" * 80)
        
        benchmark_vs_deterministic(game, n_rounds)
        benchmark_vs_uniform(game, n_rounds)
        benchmark_vs_biased(game, n_rounds)
        benchmark_vs_mw(game, n_rounds)
        benchmark_vs_exp3(game, n_rounds)


if __name__ == "__main__":
    np.random.seed(42)
    run_full_benchmark(n_rounds=3000)