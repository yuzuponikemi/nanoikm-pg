"""
Random Policy

This is the simplest possible policy - it ignores the state entirely
and just selects random actions. This serves as a baseline to compare
against learned policies.

A random policy is useful for:
1. Testing that the environment works
2. Establishing a performance baseline
3. Initial exploration in some algorithms

Of course, a random policy will perform poorly on the inverted pendulum
task since balancing requires responding appropriately to the state.
"""

import numpy as np
from typing import Dict, Any
from .base import Policy


class RandomPolicy(Policy):
    """
    A policy that selects random actions.

    This policy ignores the state and uniformly samples actions
    from the valid action range.

    Attributes:
        action_low: Minimum action value
        action_high: Maximum action value
        discrete: If True, only return -action_high or +action_high

    Example:
        >>> policy = RandomPolicy(action_high=10.0)
        >>> state = np.array([0, 0, 0.1, 0])
        >>> action = policy.get_action(state)  # Random value in [-10, 10]
    """

    def __init__(
        self,
        action_low: float = -10.0,
        action_high: float = 10.0,
        discrete: bool = False,
        seed: int = None
    ):
        """
        Initialize the random policy.

        Args:
            action_low: Minimum action value
            action_high: Maximum action value
            discrete: If True, only return extreme values (bang-bang control)
            seed: Random seed for reproducibility
        """
        self.action_low = action_low
        self.action_high = action_high
        self.discrete = discrete

        if seed is not None:
            np.random.seed(seed)

    def get_action(self, state: np.ndarray) -> float:
        """
        Select a random action (ignoring the state).

        Args:
            state: Current state (ignored)

        Returns:
            Random action value
        """
        if self.discrete:
            # Bang-bang control: only extreme values
            return np.random.choice([self.action_low, self.action_high])
        else:
            # Continuous random action
            return np.random.uniform(self.action_low, self.action_high)

    def get_params(self) -> Dict[str, Any]:
        """Get policy parameters."""
        return {
            'action_low': self.action_low,
            'action_high': self.action_high,
            'discrete': self.discrete
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set policy parameters."""
        if 'action_low' in params:
            self.action_low = params['action_low']
        if 'action_high' in params:
            self.action_high = params['action_high']
        if 'discrete' in params:
            self.discrete = params['discrete']

    def get_num_params(self) -> int:
        """Random policy has no trainable parameters."""
        return 0
