"""
Q-Learning Policy (Tabular)

Implements tabular Q-learning using a discrete state representation.
Continuous states are first discretized via StateDiscretizer, then
stored in a Q-table (numpy array).

The agent uses epsilon-greedy exploration:
- With probability epsilon: random action (exploration)
- With probability 1-epsilon: greedy action (exploitation)

Epsilon decays over episodes to shift from exploration to exploitation.

Q-value update (Bellman equation):
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
"""

import numpy as np
from typing import Optional, Dict, Any

from .base import Policy
from src.utils.discretizer import StateDiscretizer


class QTable:
    """
    Tabular Q-value storage.

    Internally stores Q(s, a) as a 2D numpy array of shape
    [n_states, n_actions]. Efficient for ~20k states.

    Parameters
    ----------
    n_states : int
    n_actions : int
    init_value : float
        Initial Q-value for all entries.
        0.0 = neutral (standard). Positive = optimistic initialization.
    """

    def __init__(self, n_states: int, n_actions: int, init_value: float = 0.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self._table = np.full((n_states, n_actions), init_value, dtype=np.float64)

    def get(self, state_idx: int, action_idx: int) -> float:
        return float(self._table[state_idx, action_idx])

    def update(self, state_idx: int, action_idx: int, value: float) -> None:
        self._table[state_idx, action_idx] = value

    def max_value(self, state_idx: int) -> float:
        """max_a Q(s, a)"""
        return float(np.max(self._table[state_idx]))

    def greedy_action(self, state_idx: int) -> int:
        """argmax_a Q(s, a)"""
        return int(np.argmax(self._table[state_idx]))

    def get_all_values(self, state_idx: int) -> np.ndarray:
        return self._table[state_idx].copy()

    @property
    def table(self) -> np.ndarray:
        """Full Q-table copy, shape (n_states, n_actions)."""
        return self._table.copy()

    def save(self, path: str) -> None:
        np.save(path, self._table)

    @classmethod
    def load(cls, path: str, n_states: int, n_actions: int) -> "QTable":
        obj = cls(n_states, n_actions)
        obj._table = np.load(path)
        return obj

    def __repr__(self) -> str:
        return f"QTable(n_states={self.n_states}, n_actions={self.n_actions})"


class QPolicy(Policy):
    """
    Tabular Q-learning agent.

    Inherits from Policy for compatibility with existing utilities.
    Adds Q-learning-specific methods: update_q() and decay_epsilon().

    Parameters
    ----------
    state_discretizer : StateDiscretizer
        Converts continuous states to integer indices.
    n_actions : int
        Number of discrete actions. Default 11.
    action_values : np.ndarray | None
        Force values for each action index.
        Default: np.linspace(-10.0, 10.0, n_actions) matching force_mag=10.
    alpha : float
        Learning rate. Recommended: 0.1
    gamma : float
        Discount factor. Recommended: 0.99
    epsilon : float
        Initial exploration rate. Recommended: 1.0 (start fully random)
    epsilon_min : float
        Minimum exploration rate. Recommended: 0.01
    epsilon_decay : float
        Per-episode multiplicative decay. Recommended: 0.995
    init_value : float
        Initial Q-table value. 0.0 = standard, positive = optimistic.

    Examples
    --------
    >>> disc = StateDiscretizer()
    >>> policy = QPolicy(disc)
    >>> action = policy.get_action(env.reset())
    """

    def __init__(
        self,
        state_discretizer: StateDiscretizer,
        n_actions: int = 11,
        action_values: Optional[np.ndarray] = None,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        init_value: float = 0.0,
    ):
        self.discretizer = state_discretizer
        self.n_actions = n_actions
        self.action_values = (
            action_values if action_values is not None
            else np.linspace(-10.0, 10.0, n_actions)
        )
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = QTable(
            n_states=state_discretizer.n_states,
            n_actions=n_actions,
            init_value=init_value,
        )
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Policy interface (required by base class)
    # ------------------------------------------------------------------

    def get_action(self, state: np.ndarray) -> float:
        """
        Select action using epsilon-greedy policy.

        Returns
        -------
        float : continuous force value from action_values
        """
        state_idx = self.discretizer.encode(state)
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.n_actions)
        else:
            action_idx = self.q_table.greedy_action(state_idx)
        return float(self.action_values[action_idx])

    def get_params(self) -> Dict[str, Any]:
        return {
            "q_table": self.q_table.table,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "n_actions": self.n_actions,
            "action_values": self.action_values.copy(),
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        if "q_table" in params:
            self.q_table._table = params["q_table"].copy()
        if "epsilon" in params:
            self.epsilon = params["epsilon"]

    # ------------------------------------------------------------------
    # Q-learning specific methods
    # ------------------------------------------------------------------

    def update_q(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """
        Perform one Q-learning update step (Bellman equation).

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        Returns
        -------
        float : TD error (useful for monitoring convergence)
        """
        state_idx = self.discretizer.encode(state)
        action_idx = int(np.argmin(np.abs(self.action_values - action)))
        next_state_idx = self.discretizer.encode(next_state)

        current_q = self.q_table.get(state_idx, action_idx)
        target = reward if done else reward + self.gamma * self.q_table.max_value(next_state_idx)
        td_error = target - current_q
        self.q_table.update(state_idx, action_idx, current_q + self.alpha * td_error)
        return float(td_error)

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode. Call once per training episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._episode_count += 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "episode": self._episode_count,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }

    def __repr__(self) -> str:
        return (
            f"QPolicy(n_states={self.discretizer.n_states}, "
            f"n_actions={self.n_actions}, "
            f"alpha={self.alpha}, gamma={self.gamma}, "
            f"epsilon={self.epsilon:.3f})"
        )
