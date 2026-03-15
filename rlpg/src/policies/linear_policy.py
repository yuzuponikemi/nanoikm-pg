"""
Linear Policy

A linear policy computes actions as a weighted sum of state features:

    action = w0 + w1*x + w2*x_dot + w3*theta + w4*theta_dot

This is one of the simplest parameterized policies, but it can be
surprisingly effective for the inverted pendulum! The physics of the
problem are relatively simple, and a linear controller can stabilize it.

Linear policies are:
- Easy to understand and interpret
- Fast to compute
- Have few parameters (easy to train)
- Can be optimal for some problems (LQR)

For the inverted pendulum, the key insight is:
- If theta > 0 (leaning right), push right to catch it
- If theta < 0 (leaning left), push left to catch it
- Also consider velocities to anticipate future states
"""

import numpy as np
from typing import Dict, Any, Optional
from .base import Policy


class LinearPolicy(Policy):
    """
    A linear policy that computes action = weights @ state + bias.

    This policy is parameterized by:
    - weights: 4-element array (one weight per state variable)
    - bias: scalar offset

    The action is clipped to the valid range [action_low, action_high].

    Attributes:
        weights: Weight vector for state features
        bias: Constant offset
        action_low: Minimum action value
        action_high: Maximum action value

    Example:
        >>> policy = LinearPolicy()
        >>> # Set weights manually (positive theta weight = push in direction of lean)
        >>> policy.weights = np.array([0, 0, 10, 1])
        >>> state = np.array([0, 0, 0.1, 0])  # Leaning right
        >>> action = policy.get_action(state)  # Positive force (push right)
    """

    def __init__(
        self,
        weights: Optional[np.ndarray] = None,
        bias: float = 0.0,
        action_low: float = -10.0,
        action_high: float = 10.0
    ):
        """
        Initialize the linear policy.

        Args:
            weights: Initial weights (4,). If None, initialized to zeros.
            bias: Initial bias value
            action_low: Minimum action value
            action_high: Maximum action value
        """
        if weights is None:
            self.weights = np.zeros(4)
        else:
            self.weights = np.array(weights, dtype=np.float64)

        self.bias = bias
        self.action_low = action_low
        self.action_high = action_high

    def get_action(self, state: np.ndarray) -> float:
        """
        Compute action as linear combination of state features.

        action = clip(weights @ state + bias, action_low, action_high)

        Args:
            state: Current state [x, x_dot, theta, theta_dot]

        Returns:
            Action value (clipped to valid range)
        """
        # Linear combination
        action = np.dot(self.weights, state) + self.bias

        # Clip to valid range
        action = np.clip(action, self.action_low, self.action_high)

        return float(action)

    def get_params(self) -> Dict[str, Any]:
        """Get policy parameters."""
        return {
            'weights': self.weights.copy(),
            'bias': self.bias,
            'action_low': self.action_low,
            'action_high': self.action_high
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set policy parameters."""
        if 'weights' in params:
            self.weights = np.array(params['weights'], dtype=np.float64)
        if 'bias' in params:
            self.bias = params['bias']
        if 'action_low' in params:
            self.action_low = params['action_low']
        if 'action_high' in params:
            self.action_high = params['action_high']

    def get_num_params(self) -> int:
        """Return number of trainable parameters (weights + bias)."""
        return len(self.weights) + 1

    def get_flat_params(self) -> np.ndarray:
        """
        Get all parameters as a flat array.

        Useful for optimization algorithms.

        Returns:
            Array of [weights..., bias]
        """
        return np.concatenate([self.weights, [self.bias]])

    def set_flat_params(self, flat_params: np.ndarray) -> None:
        """
        Set parameters from a flat array.

        Args:
            flat_params: Array of [weights..., bias]
        """
        self.weights = flat_params[:-1].copy()
        self.bias = flat_params[-1]

    def perturb(self, noise_scale: float = 0.1) -> 'LinearPolicy':
        """
        Create a new policy with perturbed parameters.

        Useful for evolutionary strategies and exploration.

        Args:
            noise_scale: Standard deviation of Gaussian noise

        Returns:
            New LinearPolicy with perturbed parameters
        """
        new_weights = self.weights + np.random.randn(4) * noise_scale
        new_bias = self.bias + np.random.randn() * noise_scale

        return LinearPolicy(
            weights=new_weights,
            bias=new_bias,
            action_low=self.action_low,
            action_high=self.action_high
        )
