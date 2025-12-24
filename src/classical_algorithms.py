import numpy as np
from typing import List, Tuple, Dict
from games import PlayerRole, RepeatedGame


class NoRegretAlgorithm:
    # base class for no-regret learning algorithms
    
    def __init__(self, game: RepeatedGame, name: str, role: PlayerRole = "row"):
        self.game = game
        self.name = name
        self.n_actions = game.n_actions
        self.role = role
        
        # history 
        self.actions_history = []
        self.opponent_actions_history = []
        self.payoffs_history = []
        
    def select_action(self) -> int:
        raise NotImplementedError
    
    def observe(self, own_action: int, opponent_action: int, payoff: float):
        self.actions_history.append(own_action)
        self.opponent_actions_history.append(opponent_action)
        self.payoffs_history.append(payoff)
        self._update(own_action, opponent_action, payoff)
    
    def _update(self, own_action: int, opponent_action: int, payoff: float):
        raise NotImplementedError
    
    def get_policy(self) -> np.ndarray:
        raise NotImplementedError
    
    def get_cumulative_regret(self) -> float:
        # external regret
        if len(self.actions_history) == 0:
            return 0.0
        
        actions = np.array(self.actions_history)
        opp_actions = np.array(self.opponent_actions_history)
        return self.game.compute_regret(actions, opp_actions, role=self.role)
    
    def get_average_regret(self) -> float:
        # average regret per round
        T = len(self.actions_history)
        if T == 0:
            return 0.0
        return self.get_cumulative_regret() / T
    
    def reset(self):
        self.actions_history = []
        self.opponent_actions_history = []
        self.payoffs_history = []


class MultiplicativeWeights(NoRegretAlgorithm):
    def __init__(self, game: RepeatedGame, eta: float = None, role: PlayerRole = "row"):
        super().__init__(game, "Multiplicative Weights", role=role)
        
        # lr
        if eta is None:
            # use eta = sqrt(log(n) / T)
            T_approx = 10000
            self.eta = np.sqrt(np.log(self.n_actions) / T_approx)
        else:
            self.eta = eta

        # Use log-weights + softmax for numerical stability over long horizons.
        self.log_weights = np.zeros(self.n_actions, dtype=float)
        
    def select_action(self) -> int:
        policy = self.get_policy()
        return np.random.choice(self.n_actions, p=policy)
    
    def _update(self, own_action: int, opponent_action: int, payoff: float):
        utilities = self.game.counterfactual_utilities(opponent_action, role=self.role)
        self.log_weights += self.eta * np.asarray(utilities, dtype=float)
    
    def get_policy(self) -> np.ndarray:
        z = self.log_weights - float(np.max(self.log_weights))
        w = np.exp(z)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            # fall back to uniform; should be extremely rare with log-weights
            return np.ones(self.n_actions, dtype=float) / float(self.n_actions)
        return w / s
    
    def reset(self):
        super().reset()
        self.log_weights = np.zeros(self.n_actions, dtype=float)


class EXP3(NoRegretAlgorithm):
    def __init__(self, game: RepeatedGame, eta: float = None, gamma: float = None, role: PlayerRole = "row"):
        super().__init__(game, "EXP3", role=role)
        
        # lr and exploration parameters
        if eta is None or gamma is None:
            # standard choice for EXP3
            T_approx = 10000
            if gamma is None:
                self.gamma = min(1.0, np.sqrt(self.n_actions * np.log(self.n_actions) / T_approx))
            else:
                self.gamma = gamma
                
            if eta is None:
                self.eta = self.gamma / self.n_actions
            else:
                self.eta = eta
        else:
            self.eta = eta
            self.gamma = gamma
        
        # Use log-weights + softmax for numerical stability over long horizons.
        self.log_weights = np.zeros(self.n_actions, dtype=float)
        self.last_action = None
        self.last_policy = None
        
    def select_action(self) -> int:
        policy = self.get_policy()
        self.last_policy = policy.copy()
        self.last_action = np.random.choice(self.n_actions, p=policy)
        return self.last_action
    
    def _update(self, own_action: int, opponent_action: int, payoff: float):
        if self.last_action is None or self.last_policy is None:
            return
        
        # importance-weighted payoff estimate
        # self-note: remember to normalize payoffs if they are not in [0, 1]
        p = float(self.last_policy[self.last_action])
        payoff_hat = payoff / max(p, 1e-12)

        self.log_weights[self.last_action] += self.eta * float(payoff_hat)
    
    def get_policy(self) -> np.ndarray:
        z = self.log_weights - float(np.max(self.log_weights))
        w = np.exp(z)
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0:
            base_policy = np.ones(self.n_actions, dtype=float) / float(self.n_actions)
        else:
            base_policy = w / w_sum
        
        # add exploration
        policy = (1 - self.gamma) * base_policy + (self.gamma / self.n_actions)
        # normalize for safety
        policy = np.asarray(policy, dtype=float)
        s = float(policy.sum())
        if not np.isfinite(s) or s <= 0:
            return np.ones(self.n_actions, dtype=float) / float(self.n_actions)
        return policy / s
    
    def reset(self):
        super().reset()
        self.log_weights = np.zeros(self.n_actions, dtype=float)
        self.last_action = None
        self.last_policy = None


def run_algorithm(algorithm: NoRegretAlgorithm, 
                  opponent, 
                  n_rounds: int,
                  verbose: bool = False,
                  algorithm_role: PlayerRole = "row") -> Dict:
    regrets = []
    avg_regrets = []
    policies = []
    exploitabilities = []

    if algorithm_role not in ("row", "col"):
        raise ValueError(f"algorithm_role must be 'row' or 'col', got {algorithm_role!r}")
    opponent_role: PlayerRole = "col" if algorithm_role == "row" else "row"
    algorithm.role = algorithm_role
    if hasattr(opponent, "role"):
        opponent.role = opponent_role

    for t in range(n_rounds):
        # Select actions (we always evaluate payoffs as (row_action, col_action)).
        if algorithm_role == "row":
            row_action = algorithm.select_action()
            col_action = opponent.select_action()
        else:
            col_action = algorithm.select_action()
            row_action = opponent.select_action()

        u_row = algorithm.game.get_payoff(row_action, col_action)
        u_col = -u_row

        # Update both with (own_action, opponent_action, own_payoff)
        if algorithm_role == "row":
            algorithm.observe(row_action, col_action, u_row)
            opponent.observe(col_action, row_action, u_col)
        else:
            algorithm.observe(col_action, row_action, u_col)
            opponent.observe(row_action, col_action, u_row)
        
        # track metrics
        cum_regret = algorithm.get_cumulative_regret()
        avg_regret = algorithm.get_average_regret()
        policy = algorithm.get_policy()
        exploit = algorithm.game.compute_exploitability(policy, role=algorithm_role)
        
        regrets.append(cum_regret)
        avg_regrets.append(avg_regret)
        policies.append(policy.copy())
        exploitabilities.append(exploit)
        
        # log every 10% of rounds
        log_every = max(1, n_rounds // 10)
        if verbose and (t + 1) % log_every == 0:
            print(f"Round {t+1}/{n_rounds}: "
                  f"Avg Regret = {avg_regret:.4f}, "
                  f"Exploit = {exploit:.4f}, "
                  f"Policy = {policy}")
    
    # dict with results
    return {
        'algorithm': algorithm.name,
        'opponent': opponent.name,
        'n_rounds': n_rounds,
        'cumulative_regrets': np.array(regrets),
        'average_regrets': np.array(avg_regrets),
        'policies': np.array(policies),
        'exploitabilities': np.array(exploitabilities),
        'actions': np.array(algorithm.actions_history),
        'opponent_actions': np.array(algorithm.opponent_actions_history),
        'payoffs': np.array(algorithm.payoffs_history),
    }


if __name__ == "__main__":
    from games import MatchingPennies
    from opponents import UniformRandomOpponent, DeterministicOpponent
    
    game = MatchingPennies()
    n_rounds = 3000

    det = DeterministicOpponent(game, 0)  # always Heads
    mw = MultiplicativeWeights(game, eta=0.1, role="row")
    results = run_algorithm(mw, det, n_rounds, verbose=False, algorithm_role="row")
    print(f"{game.name}: MW(row) vs deterministic(Heads)")
    print(f"Final policy: {results['policies'][-1]}")
    print(f"Avg regret:   {results['average_regrets'][-1]:.4f}")

    det = DeterministicOpponent(game, 0)  # always Heads as row
    mw = MultiplicativeWeights(game, eta=0.1, role="col")
    results = run_algorithm(mw, det, n_rounds, verbose=False, algorithm_role="col")
    print(f"{game.name}: MW(col) vs deterministic(Heads)")
    print(f"Final policy: {results['policies'][-1]}")
    print(f"Avg regret:   {results['average_regrets'][-1]:.4f}")
