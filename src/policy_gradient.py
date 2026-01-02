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
    def __init__(
        self,
        game: RepeatedGame,
        hidden_size: int = 64,
        lr: float = 1e-3,
        entropy_coef: float = 0.0,
        regret_matching_coef: float = 0.0,
        # regret matching config
        rm_type: str = "plus",  # "plus", "no_clip", "exponential", "softmax"
        rm_tau: float = 1.0,  # temperature for softmax/exponential RM
        rm_discount: float = 1.0,  # discount factor for regret (1.0=cumulative, <1.0 exponential weighting)
        rm_schedule: str = "constant",  # "constant", "linear_decay", "exponential_decay"
        rm_schedule_alpha: float = 1.0,  # decay rate for schedule
        use_baseline: bool = True,
        baseline_momentum: float = 0.9,
        role: PlayerRole = "row",
        name: str = None,  # if None, auto-generate from parameters
    ):
        # save hyperparameters
        self.game = game
        self.n_actions = game.n_actions
        self.entropy_coef = entropy_coef
        self.regret_matching_coef = regret_matching_coef
        self.rm_type = str(rm_type)
        self.rm_tau = float(rm_tau)
        self.rm_discount = float(rm_discount)
        self.rm_schedule = str(rm_schedule)
        self.rm_schedule_alpha = float(rm_schedule_alpha)
        self.use_baseline = use_baseline
        self.baseline_momentum = baseline_momentum
        self.role = role
        self._hidden_size = hidden_size
        self._lr = lr

        # Validate config early (fail fast)
        if self.rm_type not in {"plus", "no_clip", "exponential", "softmax"}:
            raise ValueError(f"Unknown rm_type: {self.rm_type!r}")
        if self.rm_tau <= 0:
            raise ValueError(f"rm_tau must be > 0, got {self.rm_tau}")
        if not (0.0 <= self.rm_discount <= 1.0):
            raise ValueError(f"rm_discount must be in [0, 1], got {self.rm_discount}")
        if self.rm_schedule not in {"constant", "linear_decay", "exponential_decay"}:
            raise ValueError(f"Unknown rm_schedule: {self.rm_schedule!r}")
        if self.rm_schedule_alpha <= 0:
            raise ValueError(f"rm_schedule_alpha must be > 0, got {self.rm_schedule_alpha}")

        if name is None:
            name_parts = ["PG"]

            if regret_matching_coef > 0:
                if self.rm_type == "plus":
                    name_parts.append(f"RM+(λ={regret_matching_coef})")
                elif self.rm_type == "no_clip":
                    name_parts.append(f"RM-NoClip(λ={regret_matching_coef})")
                elif self.rm_type == "exponential":
                    name_parts.append(f"ExpRM(λ={regret_matching_coef},τ={self.rm_tau})")
                elif self.rm_type == "softmax":
                    name_parts.append(f"SoftmaxRM(λ={regret_matching_coef},τ={self.rm_tau})")

                if self.rm_discount < 1.0:
                    name_parts.append(f"Disc={self.rm_discount}")

                if self.rm_schedule != "constant":
                    name_parts.append(f"Sched={self.rm_schedule}")

            if entropy_coef > 0:
                name_parts.append(f"H={entropy_coef}")

            if regret_matching_coef == 0 and entropy_coef == 0:
                name_parts.append("Vanilla")

            self.name = "_".join(name_parts)
        else:
            self.name = name
        
        # initialize network
        self.policy_net = PolicyNetwork(self.n_actions, hidden_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # constant input for stateless game
        self.constant_input = torch.ones(1, 1)
        
        # baseline (EMA of payoffs)
        self.baseline = 0.0
        self._steps = 0

        # regret-matching (full-information) accumulators
        self.regret_sums = np.zeros(self.n_actions, dtype=float)

        # if set, schedules use this horizon instead of a hard-coded default
        self.rm_horizon_steps: Optional[int] = None
        
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
        loss = self._update(own_action, opponent_action, payoff)
        self.losses_history.append(loss)

    def _compute_rm_target(self) -> np.ndarray:
        """Compute q_t(a) based on rm_type and current regret_sums."""
        if self.rm_type == "plus":
            # RM+: max(R_t(a), 0)
            pos = np.maximum(self.regret_sums, 0.0)
        elif self.rm_type == "no_clip":
            # RM without clipping: shift to make valid
            pos = self.regret_sums - float(np.min(self.regret_sums))
        elif self.rm_type in {"exponential", "softmax"}:
            # Exponential / Softmax: exp(R_t(a) / tau)
            z = self.regret_sums / float(self.rm_tau)
            z = z - float(np.max(z))  # numerical stability
            pos = np.exp(z)
        else:
            raise ValueError(f"Unknown rm_type: {self.rm_type!r}")

        denom = float(np.sum(pos))
        if denom <= 1e-12 or not np.isfinite(denom):
            return np.ones(self.n_actions, dtype=float) / float(self.n_actions)
        q = pos / denom
        return np.asarray(q, dtype=float)

    def _current_rm_lambda(self) -> float:
        """Compute current lambda_R based on schedule (constant/decay)."""
        base = float(self.regret_matching_coef)
        if base <= 0.0:
            return 0.0

        if self.rm_schedule == "constant":
            return base

        horizon = int(self.rm_horizon_steps) if self.rm_horizon_steps is not None else 10000
        progress = float(self._steps) / float(max(1, horizon))
        progress = max(0.0, min(1.0, progress))

        if self.rm_schedule == "linear_decay":
            return base * (1.0 - progress)

        if self.rm_schedule == "exponential_decay":
            return base * float(np.exp(-float(self.rm_schedule_alpha) * progress))

        raise ValueError(f"Unknown rm_schedule: {self.rm_schedule!r}")
        
    def _update(self, action: int, opponent_action: int, payoff: float) -> float:
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

        # regret-matching regularizer (full information)
        rm_loss = torch.tensor(0.0)
        current_lambda_R = self._current_rm_lambda()
        if current_lambda_R > 0.0:
            utilities = self.game.counterfactual_utilities(opponent_action, role=self.role)
            instant_regret = np.asarray(utilities, dtype=float) - float(payoff)
            self.regret_sums = (float(self.rm_discount) * self.regret_sums) + instant_regret

            q = self._compute_rm_target()
            q_t = torch.tensor(q, dtype=log_probs.dtype)
            rm_loss = -float(current_lambda_R) * torch.sum(q_t * log_probs)
        
        # total loss
        loss = pg_loss + rm_loss + entropy_loss
        
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
        self.regret_sums = np.zeros(self.n_actions, dtype=float)
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
    instant_regrets_cumulative = []
    instant_regrets_average = []
    policies = []
    exploitabilities = []
    time_avg_exploitabilities = []
    avg_policy: np.ndarray | None = None
    losses = []
    entropies = []

    # Track both players' regret per-iteration (row/col), independent of algorithm_role.
    row_cum_regrets = []
    col_cum_regrets = []
    row_avg_regrets = []
    col_avg_regrets = []

    A = np.asarray(pg.game.payoff_matrix, dtype=float)
    n_actions = int(pg.game.n_actions)
    row_realized = 0.0
    col_realized = 0.0
    row_counterfactual_sums = np.zeros(n_actions, dtype=float)
    col_counterfactual_sums = np.zeros(n_actions, dtype=float)
    instant_cum = 0.0
    
    if algorithm_role not in ("row", "col"):
        raise ValueError(f"algorithm_role must be 'row' or 'col', got {algorithm_role!r}")
    pg.role = algorithm_role
    pg.rm_horizon_steps = int(n_rounds)
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

        # instant regret (best response each round)
        if algorithm_role == "row":
            own_action, opp_action = row_action, col_action
        else:
            own_action, opp_action = col_action, row_action
        utilities = pg.game.counterfactual_utilities(opp_action, role=algorithm_role)
        realized = pg.game.get_utility(own_action, opp_action, role=algorithm_role)
        inst_t = float(np.max(utilities) - float(realized))
        instant_cum += inst_t
        step = t + 1
        instant_regrets_cumulative.append(instant_cum)
        instant_regrets_average.append(instant_cum / step)

        # regret for both players (row + col)
        row_realized += float(A[row_action, col_action])
        row_counterfactual_sums += A[:, col_action]
        row_cum = float(row_counterfactual_sums.max() - row_realized)

        col_realized += float(-A[row_action, col_action])
        col_counterfactual_sums += -A[row_action, :]
        col_cum = float(col_counterfactual_sums.max() - col_realized)

        row_cum_regrets.append(row_cum)
        col_cum_regrets.append(col_cum)
        row_avg_regrets.append(row_cum / step)
        col_avg_regrets.append(col_cum / step)
        
        regrets.append(cum_regret)
        avg_regrets.append(avg_regret)
        policies.append(policy.copy())
        exploitabilities.append(exploit)

        # time-averaged exploitability (avg of policies so far)
        if avg_policy is None:
            avg_policy = policy.copy()
        else:
            avg_policy = ((step - 1) * avg_policy + policy) / step
        time_avg_exploit = pg.game.compute_exploitability(avg_policy, role=algorithm_role)
        time_avg_exploitabilities.append(time_avg_exploit)

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
        'algorithm': getattr(pg, 'name', f'PolicyGradient(entropy={pg.entropy_coef})'),
        'opponent': opponent.name,
        'n_rounds': n_rounds,
        'cumulative_regrets': np.array(regrets),
        'average_regrets': np.array(avg_regrets),
        'instant_regrets_cumulative': np.array(instant_regrets_cumulative),
        'instant_regrets_average': np.array(instant_regrets_average),
        'row_cumulative_regrets': np.array(row_cum_regrets),
        'col_cumulative_regrets': np.array(col_cum_regrets),
        'row_average_regrets': np.array(row_avg_regrets),
        'col_average_regrets': np.array(col_avg_regrets),
        'policies': np.array(policies),
        'exploitabilities': np.array(exploitabilities),
        'time_avg_exploitabilities': np.array(time_avg_exploitabilities),
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
