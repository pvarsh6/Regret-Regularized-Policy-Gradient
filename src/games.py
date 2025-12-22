from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np

PlayerRole = Literal["row", "col"]


@dataclass
class RepeatedGame:
    payoff_matrix: np.ndarray
    action_names: List[str]
    name: str = "RepeatedGame"
    nash_value: float = 0.0

    def __post_init__(self) -> None:
        self.payoff_matrix = np.asarray(self.payoff_matrix, dtype=float)
        if self.payoff_matrix.ndim != 2:
            raise ValueError("payoff_matrix must be 2D")
        if self.payoff_matrix.shape[0] != self.payoff_matrix.shape[1]:
            raise ValueError(
                "Assuming same action set for both players; "
                f"Received shape {self.payoff_matrix.shape}."
            )
        if len(self.action_names) != self.payoff_matrix.shape[0]:
            raise ValueError(
                "action_names length must match number of actions: "
                f"{len(self.action_names)} vs {self.payoff_matrix.shape[0]}"
            )

    @property
    def n_actions(self) -> int:
        return int(self.payoff_matrix.shape[0])

    def get_payoff(self, row_action: int, col_action: int) -> float:
        return float(self.payoff_matrix[row_action, col_action])

    def get_utility(self, own_action: int, opponent_action: int, role: PlayerRole) -> float:
        
        # utility for the player in the given role, using (own_action, opponent_action)
        if role == "row":
            return self.get_payoff(own_action, opponent_action)
        if role == "col":
            return -self.get_payoff(opponent_action, own_action)
        raise ValueError(f"Unknown role: {role!r}")

    def counterfactual_utilities(self, opponent_action: int, role: PlayerRole) -> np.ndarray:
        # vector of utilities for choosing each action, given the opponent action, for the given role
        if role == "row":
            return np.asarray(self.payoff_matrix[:, opponent_action], dtype=float)
        if role == "col":
            return -np.asarray(self.payoff_matrix[opponent_action, :], dtype=float)
        raise ValueError(f"Unknown role: {role!r}")

    def compute_regret(
        self,
        actions: np.ndarray,
        opponent_actions: np.ndarray,
        role: PlayerRole = "row",
    ) -> float:
        # external regret against the realized opponent action sequence
        actions = np.asarray(actions, dtype=int)
        opponent_actions = np.asarray(opponent_actions, dtype=int)
        if actions.shape != opponent_actions.shape:
            raise ValueError("actions and opponent_actions must have same shape")
        if actions.size == 0:
            return 0.0

        if role == "row":
            realized = self.payoff_matrix[actions, opponent_actions].sum()
            best_fixed = self.payoff_matrix[:, opponent_actions].sum(axis=1).max()
            return float(best_fixed - realized)

        if role == "col":
            row_actions = opponent_actions
            col_actions = actions
            realized = (-self.payoff_matrix[row_actions, col_actions]).sum()
            best_fixed = (-self.payoff_matrix[row_actions, :]).sum(axis=0).max()
            return float(best_fixed - realized)

        raise ValueError(f"Unknown role: {role!r}")

    def compute_exploitability(self, policy: np.ndarray, role: PlayerRole = "row") -> float:
        # exploitability of a mixed strategy for either role in a zero-sum game
        p = np.asarray(policy, dtype=float)
        if p.shape != (self.n_actions,):
            raise ValueError(f"policy must have shape ({self.n_actions},)")
        if np.any(p < -1e-12):
            raise ValueError("policy has negative entries")
        s = p.sum()
        if s <= 0:
            raise ValueError("policy must sum to > 0")
        p = p / s

        if role == "row":
            value_vs_col = p @ self.payoff_matrix
            worst_case_value = float(value_vs_col.min())
            exploit = float(self.nash_value - worst_case_value)
            return max(0.0, exploit)

        if role == "col":
            value_vs_row = self.payoff_matrix @ p
            best_response_value = float(value_vs_row.max())
            exploit = float(best_response_value - self.nash_value)
            return max(0.0, exploit)

        raise ValueError(f"Unknown role: {role!r}")


def MatchingPennies() -> RepeatedGame:
    # matching pennies (zero-sum, value 0)
    # row payoff is +1 if actions match, -1 otherwise
    A = np.array([[+1.0, -1.0],
                  [-1.0, +1.0]])
    return RepeatedGame(
        payoff_matrix=A,
        action_names=["Heads", "Tails"],
        name="Matching Pennies",
        nash_value=0.0,
    )


def RockPaperScissors() -> RepeatedGame:
    # rock–paper–scissors (zero-sum, value 0)
    A = np.array([
        [0.0, -1.0, +1.0],   # Rock vs Rock/Paper/Scissors
        [+1.0, 0.0, -1.0],   # Paper
        [-1.0, +1.0, 0.0],   # Scissors
    ])
    return RepeatedGame(
        payoff_matrix=A,
        action_names=["Rock", "Paper", "Scissors"],
        name="Rock-Paper-Scissors",
        nash_value=0.0,
    )
