import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
from games import PlayerRole, RepeatedGame


class PolicyNetwork(nn.Module):
    def __init__(self, n_actions: int, hidden_size: int = 64):
        super().__init__()
        
        # constant input for stateless games
        self.constant_input_size = 1
        
        self.network = nn.Sequential(
            nn.Linear(self.constant_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward pass
        return self.network(x)
    
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        # get action probabilities via softmax
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


class PolicyGradient:
    # vanilla policy gradient (REINFORCE) with baseline
    def __init__(self, 
                 game: RepeatedGame,
                 hidden_size: int = 64,
                 lr: float = 1e-3,
                 entropy_coef: float = 0.0,
                 use_baseline: bool = True,
                 baseline_momentum: float = 0.9,
                 role: PlayerRole = "row"):
        # save hyperparameters
        self.game = game
        self.n_actions = game.n_actions
        self.entropy_coef = entropy_coef
        self.use_baseline = use_baseline
        self.baseline_momentum = baseline_momentum
        self.role = role
        self._hidden_size = hidden_size
        self._lr = lr
        
        # initialize network
        self.policy_net = PolicyNetwork(self.n_actions, hidden_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # constant input for stateless game
        self.constant_input = torch.ones(1, 1)
        
        # baseline (EMA of payoffs)
        self.baseline = 0.0
        self._steps = 0
        
        # history
        self.actions_history = []
        self.opponent_actions_history = []
        self.payoffs_history = []
        self.policies_history = []
        self.losses_history = []
        
    def select_action(self) -> int:
        with torch.no_grad():
            probs = self.policy_net.get_action_probs(self.constant_input)
            probs = probs.squeeze().numpy()
            
        action = np.random.choice(self.n_actions, p=probs)
        return action
    
    def get_policy(self) -> np.ndarray:
        with torch.no_grad():
            probs = self.policy_net.get_action_probs(self.constant_input)
            return probs.squeeze().numpy()
    
    def observe(self, own_action: int, opponent_action: int, payoff: float):
        # history
        self.actions_history.append(own_action)
        self.opponent_actions_history.append(opponent_action)
        self.payoffs_history.append(payoff)
        self.policies_history.append(self.get_policy())
        
        # Update policy
        loss = self._update(own_action, payoff)
        self.losses_history.append(loss)
        
    def _update(self, action: int, payoff: float) -> float:
        # update policy using REINFORCE
        self.optimizer.zero_grad()
        
        # current policy
        logits = self.policy_net(self.constant_input).squeeze()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        
        # baseline subtraction
        if self.use_baseline:
            if self._steps == 0:
                advantage = payoff
            else:
                advantage = payoff - self.baseline

            # update baseline *after* computing the advantage
            if self._steps == 0:
                self.baseline = payoff
            else:
                self.baseline = (self.baseline_momentum * self.baseline +
                                 (1 - self.baseline_momentum) * payoff)
            reward = advantage
        else:
            reward = payoff
        
        # policy gradient loss
        pg_loss = -reward * log_probs[action]
        
        # entropy regularization (encourage exploration)
        entropy = -(probs * log_probs).sum()
        entropy_loss = -self.entropy_coef * entropy
        
        # total loss
        loss = pg_loss + entropy_loss
        
        # backprop and update
        loss.backward()
        self.optimizer.step()

        self._steps += 1
        
        return loss.item()
    
    def get_cumulative_regret(self) -> float:
        # cumulative external regret
        if len(self.actions_history) == 0:
            return 0.0
        
        actions = np.array(self.actions_history)
        opp_actions = np.array(self.opponent_actions_history)
        return self.game.compute_regret(actions, opp_actions, role=self.role)
    
    def get_average_regret(self) -> float:
        T = len(self.actions_history)
        if T == 0:
            return 0.0
        return self.get_cumulative_regret() / T
    
    def reset(self):
        self.policy_net = PolicyNetwork(self.n_actions, hidden_size=self._hidden_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self._lr)
        
        # reset tracking
        self.baseline = 0.0
        self._steps = 0
        self.actions_history = []
        self.opponent_actions_history = []
        self.payoffs_history = []
        self.policies_history = []
        self.losses_history = []


def run_policy_gradient(pg: PolicyGradient,
                       opponent,
                       n_rounds: int,
                       verbose: bool = False,
                       algorithm_role: PlayerRole = "row") -> Dict:
    # run policy gradient algorithm against an opponent
    regrets = []
    avg_regrets = []
    policies = []
    exploitabilities = []
    losses = []
    entropies = []
    
    if algorithm_role not in ("row", "col"):
        raise ValueError(f"algorithm_role must be 'row' or 'col', got {algorithm_role!r}")
    pg.role = algorithm_role
    if hasattr(opponent, "role"):
        opponent.role = "col" if algorithm_role == "row" else "row"

    # algorithm
    for t in range(n_rounds):
        if algorithm_role == "row":
            row_action = pg.select_action()
            col_action = opponent.select_action()
        else:
            col_action = pg.select_action()
            row_action = opponent.select_action()

        u_row = pg.game.get_payoff(row_action, col_action)
        u_col = -u_row

        if algorithm_role == "row":
            pg.observe(row_action, col_action, u_row)
            opponent.observe(col_action, row_action, u_col)
        else:
            pg.observe(col_action, row_action, u_col)
            opponent.observe(row_action, col_action, u_row)
        
        # track metrics
        cum_regret = pg.get_cumulative_regret()
        avg_regret = pg.get_average_regret()
        policy = pg.get_policy()
        exploit = pg.game.compute_exploitability(policy, role=algorithm_role)
        
        entropy = -np.sum(policy * np.log(policy + 1e-10))
        
        regrets.append(cum_regret)
        avg_regrets.append(avg_regret)
        policies.append(policy.copy())
        exploitabilities.append(exploit)
        losses.append(pg.losses_history[-1] if pg.losses_history else 0.0)
        entropies.append(entropy)
        
        # logging every 10% of rounds
        log_every = max(1, n_rounds // 10)
        if verbose and (t + 1) % log_every == 0:
            print(f"Round {t+1}/{n_rounds}: "
                  f"Avg Regret = {avg_regret:.4f}, "
                  f"Exploit = {exploit:.4f}, "
                  f"Entropy = {entropy:.4f}, "
                  f"Policy = {policy}")
    
    return {
        'algorithm': f'PolicyGradient(entropy={pg.entropy_coef})',
        'opponent': opponent.name,
        'n_rounds': n_rounds,
        'cumulative_regrets': np.array(regrets),
        'average_regrets': np.array(avg_regrets),
        'policies': np.array(policies),
        'exploitabilities': np.array(exploitabilities),
        'losses': np.array(losses),
        'entropies': np.array(entropies),
        'actions': np.array(pg.actions_history),
        'opponent_actions': np.array(pg.opponent_actions_history),
        'payoffs': np.array(pg.payoffs_history),
    }


if __name__ == "__main__":
    from games import MatchingPennies
    from opponents import UniformRandomOpponent, DeterministicOpponent
    
    game = MatchingPennies()
    n_rounds = 3000

    det = DeterministicOpponent(game, 0)  # always Heads
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="row")
    results = run_policy_gradient(pg, det, n_rounds, verbose=False, algorithm_role="row")
    print(f"{game.name}: PG(row) vs deterministic(Heads)")
    print(f"Final policy: {results['policies'][-1]}")
    print(f"Avg regret:   {results['average_regrets'][-1]:.4f}")

    det = DeterministicOpponent(game, 0)  # always Heads as row
    pg = PolicyGradient(game, hidden_size=64, lr=1e-2, entropy_coef=0.01, role="col")
    results = run_policy_gradient(pg, det, n_rounds, verbose=False, algorithm_role="col")
    print(f"{game.name}: PG(col) vs deterministic(Heads)")
    print(f"Final policy: {results['policies'][-1]}")
    print(f"Avg regret:   {results['average_regrets'][-1]:.4f}")
