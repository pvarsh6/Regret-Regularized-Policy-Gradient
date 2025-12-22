import numpy as np
from typing import Optional
from games import PlayerRole, RepeatedGame


class Opponent:
    def __init__(self, game: RepeatedGame, name: str, role: PlayerRole = "col"):
        self.game = game
        self.name = name
        self.n_actions = game.n_actions
        self.role = role
        
    def select_action(self) -> int:
        raise NotImplementedError
    
    def observe(self, own_action: int, opponent_action: int, payoff: float):
        pass
    
    def reset(self):
        pass


class UniformRandomOpponent(Opponent):
    def __init__(self, game: RepeatedGame):
        super().__init__(game, "Uniform Random")
        
    def select_action(self) -> int:
        return np.random.randint(self.n_actions)


class FixedBiasedOpponent(Opponent):
    def __init__(self, game: RepeatedGame, probabilities: np.ndarray):
        super().__init__(game, f"Fixed Biased {probabilities}")
        assert len(probabilities) == game.n_actions
        assert np.abs(np.sum(probabilities) - 1.0) < 1e-6
        self.probabilities = probabilities
        
    def select_action(self) -> int:
        return np.random.choice(self.n_actions, p=self.probabilities)


class DeterministicOpponent(Opponent):
    def __init__(self, game: RepeatedGame, action: int):
        super().__init__(game, f"Deterministic({game.action_names[action]})")
        self.action = action
        
    def select_action(self) -> int:
        return self.action


class MultiplicativeWeightsOpponent(Opponent):
    def __init__(self, game: RepeatedGame, eta: float = 0.1, role: PlayerRole = "col"):
        super().__init__(game, f"MW(eta={eta})", role=role)
        self.eta = eta
        self.weights = np.ones(self.n_actions)
        
    def select_action(self) -> int:
        probs = self.weights / np.sum(self.weights)
        return np.random.choice(self.n_actions, p=probs)
    
    def observe(self, own_action: int, opponent_action: int, payoff: float):
        # full-information update based on opponent action and role
        utilities = self.game.counterfactual_utilities(opponent_action, role=self.role)
        for a in range(self.n_actions):
            self.weights[a] *= np.exp(self.eta * float(utilities[a]))
    
    def reset(self):
        self.weights = np.ones(self.n_actions)


class EXP3Opponent(Opponent):
    def __init__(self, game: RepeatedGame, eta: float = 0.1, gamma: float = 0.1, role: PlayerRole = "col"):
        super().__init__(game, f"EXP3(eta={eta}, gamma={gamma})", role=role)
        self.eta = eta
        self.gamma = gamma
        self.weights = np.ones(self.n_actions)
        self.last_action = None
        self.last_probs = None
        
    def select_action(self) -> int:
        w_sum = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / w_sum) + \
                (self.gamma / self.n_actions)
        
        self.last_probs = probs
        self.last_action = np.random.choice(self.n_actions, p=probs)
        return self.last_action
    
    def observe(self, own_action: int, opponent_action: int, payoff: float):
        if self.last_action is None:
            return

        # importance-weighted payoff estimate
        p = float(self.last_probs[self.last_action])
        payoff_hat = payoff / max(p, 1e-12)

        # exp(eta * payoff_hat)
        self.weights[self.last_action] *= np.exp(self.eta * payoff_hat)
    
    def reset(self):
        self.weights = np.ones(self.n_actions)
        self.last_action = None
        self.last_probs = None


def get_opponent(opponent_type: str, game: RepeatedGame, **kwargs) -> Opponent:
    opponent_type = opponent_type.lower()
    role = kwargs.get("role", "col")
    
    if opponent_type == 'uniform':
        return UniformRandomOpponent(game)
    
    elif opponent_type == 'deterministic':
        action = kwargs.get('action', 0)
        return DeterministicOpponent(game, action)
    
    elif opponent_type == 'biased':
        probs = kwargs.get('probabilities')
        if probs is None:
            # biased towards an action
            probs = np.ones(game.n_actions) / game.n_actions
            probs[0] += 0.2
            probs /= np.sum(probs)
        return FixedBiasedOpponent(game, probs)
    
    elif opponent_type == 'mw':
        eta = kwargs.get('eta', 0.1)
        return MultiplicativeWeightsOpponent(game, eta, role=role)
    
    elif opponent_type == 'exp3':
        eta = kwargs.get('eta', 0.1)
        gamma = kwargs.get('gamma', 0.1)
        return EXP3Opponent(game, eta, gamma, role=role)
    
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

# simple testing
if __name__ == "__main__":
    from games import MatchingPennies
    
    game = MatchingPennies()
    mw = MultiplicativeWeightsOpponent(game, eta=0.1, role="row")
    exp3 = EXP3Opponent(game, eta=0.1, gamma=0.1, role="row")

    opp_action = 0  # "Heads"
    for _ in range(50):
        a = mw.select_action()
        mw.observe(a, opp_action, game.get_utility(a, opp_action, role=mw.role))

        b = exp3.select_action()
        exp3.observe(b, opp_action, game.get_utility(b, opp_action, role=exp3.role))

    mw_probs = mw.weights / np.sum(mw.weights)
    exp3_probs = exp3.weights / np.sum(exp3.weights)
    print(f"{game.name}: vs always-Heads (row role)")
    print(f"MW probs:   {mw_probs}")
    print(f"EXP3 probs: {exp3_probs}")
