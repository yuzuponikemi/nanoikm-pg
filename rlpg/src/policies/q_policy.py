"""
Q-Learning Policy (Table-based)

Implements tabular Q-learning for the inverted pendulum environment.
The continuous state space is discretized into bins, and a Q-table
stores action values for each (discretized state, action) pair.

Key concepts:
- State discretization: continuous -> bins via np.digitize
- Epsilon-greedy exploration: balance explore/exploit
- TD update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]

Unlike policy-gradient methods (LinearPolicy, NeuralPolicy), Q-learning
learns a *value function* that maps (state, action) pairs to expected
cumulative reward. The policy is derived implicitly by always choosing
the action with the highest Q-value (greedy policy).

This is a tabular method, meaning we maintain an explicit table rather
than approximating the Q-function with a neural network (that would be DQN).
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .base import Policy


# Default state discretization: [low, high, n_bins] per dimension
# State: [x, x_dot, theta, theta_dot]
# Velocity dimensions are theoretically unbounded; we clip to practical ranges.
DEFAULT_STATE_BINS = [
    [-2.4,    2.4,    6],    # x: cart position (±2.4m)
    [-3.0,    3.0,    6],    # x_dot: cart velocity (clipped to ±3 m/s)
    [-0.2095, 0.2095, 10],   # theta: pole angle (±12 degrees, most critical)
    [-3.0,    3.0,    10],   # theta_dot: angular velocity (clipped to ±3 rad/s)
]


class QPolicy(Policy):
    """
    Tabular Q-learning policy.

    Discretizes the continuous state space into bins and maintains
    a Q-table mapping (discrete_state, action) -> expected return.

    Attributes:
        q_table:     ndarray of shape (*bin_counts, n_actions)
        alpha:       Learning rate
        gamma:       Discount factor
        epsilon:     Exploration probability (epsilon-greedy)
        n_actions:   Number of discrete action values
        state_bins:  List of [low, high, n_bins] per state dimension
        action_low:  Minimum action value
        action_high: Maximum action value

    Training interface:
        Call update(state, action, reward, next_state, done) after
        each env.step(). Use train_q_learning() in training.py for
        the full training loop.

    Evaluation:
        Set policy.epsilon = 0.0 before calling evaluate_policy()
        to use the greedy policy.

    Example:
        >>> env = InvertedPendulumEnv()
        >>> policy = QPolicy()
        >>> state = env.reset()
        >>> action = policy.get_action(state)
        >>> next_state, reward, done, _ = env.step(action)
        >>> policy.update(state, action, reward, next_state, done)
    """

    def __init__(
        self,
        state_bins: Optional[List[List]] = None,
        n_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        action_low: float = -10.0,
        action_high: float = 10.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Q-learning policy.

        Args:
            state_bins:  List of [low, high, n_bins] for each state dimension.
                         Defaults to DEFAULT_STATE_BINS.
            n_actions:   Number of discrete actions. Actions are evenly spaced
                         between action_low and action_high.
            alpha:       Learning rate in (0, 1].
            gamma:       Discount factor in [0, 1].
            epsilon:     Initial epsilon for epsilon-greedy. Typically starts
                         at 1.0 (pure exploration) and decays during training.
            action_low:  Minimum action value (should match env force_mag).
            action_high: Maximum action value.
            seed:        Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        self.state_bins = state_bins if state_bins is not None else DEFAULT_STATE_BINS
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_low = action_low
        self.action_high = action_high

        # Precompute bin edges for fast discretization
        # For n_bins intervals, we use the interior edges (excludes endpoints)
        self._bin_edges: List[np.ndarray] = []
        self._bin_counts: List[int] = []
        for low, high, n_bins in self.state_bins:
            # np.linspace gives n_bins+1 points; interior points are the edges
            edges = np.linspace(low, high, n_bins + 1)[1:-1]
            self._bin_edges.append(edges)
            self._bin_counts.append(n_bins)

        # Discrete action values: evenly spaced between action_low and action_high
        # e.g., n_actions=3 -> [-10, 0, 10] for default action range
        self._action_values = np.linspace(action_low, action_high, n_actions)

        # Q-table: shape (*bin_counts, n_actions)
        # Zero-initialized (epsilon-greedy handles initial exploration)
        q_shape = tuple(self._bin_counts) + (n_actions,)
        self.q_table = np.zeros(q_shape, dtype=np.float64)

        # Internal stats for diagnostics
        self._n_updates: int = 0
        self._total_td_error: float = 0.0

    # ------------------------------------------------------------------
    # Policy interface implementation
    # ------------------------------------------------------------------

    def get_action(self, state: np.ndarray) -> float:
        """
        Select an action using epsilon-greedy policy.

        With probability epsilon, selects a random action (exploration).
        Otherwise, selects the action with the highest Q-value (exploitation).

        Args:
            state: Current environment state [x, x_dot, theta, theta_dot]

        Returns:
            Action value (force to apply to cart)
        """
        state_idx = self._discretize_state(state)

        if np.random.random() < self.epsilon:
            # Exploration: random action
            action_idx = np.random.randint(self.n_actions)
        else:
            # Exploitation: greedy action (random tiebreak for equal Q-values)
            q_values = self.q_table[state_idx]
            action_idx = int(np.random.choice(
                np.flatnonzero(q_values == q_values.max())
            ))

        return float(self._action_values[action_idx])

    def get_params(self) -> Dict[str, Any]:
        """Get all policy parameters for saving/loading."""
        return {
            'q_table': self.q_table.copy(),
            'state_bins': [list(b) for b in self.state_bins],
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'action_low': self.action_low,
            'action_high': self.action_high,
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set policy parameters from a dictionary."""
        if 'q_table' in params:
            self.q_table = np.array(params['q_table'], dtype=np.float64)
        for key in ('alpha', 'gamma', 'epsilon', 'action_low', 'action_high'):
            if key in params:
                setattr(self, key, params[key])

    def get_num_params(self) -> int:
        """Return the total number of entries in the Q-table."""
        return int(self.q_table.size)

    # ------------------------------------------------------------------
    # Q-learning specific methods
    # ------------------------------------------------------------------

    def update(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """
        Update Q-value for one step using the TD error.

        Applies the Q-learning update rule:
            Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        For terminal states (done=True), the target is just the reward
        (no future value).

        Args:
            state:      Current state (continuous values)
            action:     Action taken (the float value returned by get_action)
            reward:     Reward received
            next_state: Next state (continuous values)
            done:       Whether the episode ended

        Returns:
            TD error (absolute value indicates convergence; should decrease
            over training as Q-values stabilize)
        """
        s_idx = self._discretize_state(state)
        s_next_idx = self._discretize_state(next_state)
        a_idx = self._action_to_index(action)

        current_q = self.q_table[s_idx][a_idx]

        if done:
            # Terminal state: no future reward
            td_target = reward
        else:
            td_target = reward + self.gamma * self.q_table[s_next_idx].max()

        td_error = td_target - current_q
        self.q_table[s_idx][a_idx] += self.alpha * td_error

        # Update internal stats
        self._n_updates += 1
        self._total_td_error += abs(td_error)

        return td_error

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Return Q-values for all actions in the given state.

        Useful for visualization and debugging.

        Args:
            state: Current state (continuous values)

        Returns:
            Array of Q-values, one per action
        """
        return self.q_table[self._discretize_state(state)].copy()

    def get_visited_fraction(self) -> float:
        """
        Return the fraction of Q-table entries that have been updated.

        A value close to 1.0 means the agent has explored most of the
        discretized state-action space. Low values suggest the agent
        is only visiting a small region.
        """
        return float(np.count_nonzero(self.q_table)) / self.q_table.size

    def reset_stats(self) -> None:
        """Reset internal training statistics."""
        self._n_updates = 0
        self._total_td_error = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """
        Convert continuous state to a tuple of bin indices.

        Values outside the defined range are clipped to the boundary bins.
        """
        indices = []
        for i, (edges, (low, high, n_bins)) in enumerate(
            zip(self._bin_edges, self.state_bins)
        ):
            clipped = np.clip(state[i], low, high)
            idx = int(np.digitize(clipped, edges))
            # digitize returns [0, n_bins], clamp to [0, n_bins-1]
            idx = min(idx, n_bins - 1)
            indices.append(idx)
        return tuple(indices)

    def _action_to_index(self, action: float) -> int:
        """
        Map a continuous action value to the nearest discrete action index.

        Since get_action always returns one of self._action_values, this
        is effectively an exact lookup with tolerance for floating point.
        """
        return int(np.argmin(np.abs(self._action_values - action)))
