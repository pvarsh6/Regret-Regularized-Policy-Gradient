import numpy as np
from games import MatchingPennies, RockPaperScissors
from opponents import DeterministicOpponent
from classical_algorithms import MultiplicativeWeights, run_algorithm
from policy_gradient import PolicyGradient, run_policy_gradient


def test_games():
    mp = MatchingPennies()
    nash_policy = np.array([0.5, 0.5])
    exploit = mp.compute_exploitability(nash_policy)
    assert abs(exploit) < 1e-6

    biased = np.array([0.8, 0.2])
    exploit = mp.compute_exploitability(biased)
    assert exploit > 0

    rps = RockPaperScissors()
    nash_policy = np.array([1 / 3, 1 / 3, 1 / 3])
    exploit = rps.compute_exploitability(nash_policy)
    assert abs(exploit) < 1e-6


def test_classical_algorithms():
    game = MatchingPennies()
    n_rounds = 2000

    det = DeterministicOpponent(game, 0)  # row plays Heads
    mw_row = MultiplicativeWeights(game, eta=0.1, role="row")
    results = run_algorithm(mw_row, det, n_rounds, algorithm_role="row")
    assert results["policies"][-1][0] > 0.9

    det = DeterministicOpponent(game, 0)  # row plays Heads
    mw_col = MultiplicativeWeights(game, eta=0.1, role="col")
    results = run_algorithm(mw_col, det, n_rounds, algorithm_role="col")
    assert results["policies"][-1][1] > 0.9


def test_policy_gradient():
    game = MatchingPennies()
    n_rounds = 3000

    det = DeterministicOpponent(game, 0)  # row plays Heads
    pg_row = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    results = run_policy_gradient(pg_row, det, n_rounds, algorithm_role="row")
    assert results["policies"][-1][0] > 0.7

    det = DeterministicOpponent(game, 0)  # row plays Heads
    pg_col = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="col")
    results = run_policy_gradient(pg_col, det, n_rounds, algorithm_role="col")
    assert results["policies"][-1][1] > 0.7


if __name__ == "__main__":
    test_games()
    test_classical_algorithms()
    test_policy_gradient()
    print("All tests passed")
