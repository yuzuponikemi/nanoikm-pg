"""
Base Policy Class

This module defines the abstract base class for all policies.
A policy maps states to actions - it's the "brain" of the agent.

In reinforcement learning, the policy is what we're trying to learn.
It can be:
- Deterministic: pi(s) = a (same state always gives same action)
- Stochastic: pi(a|s) = P(a|s) (probability distribution over actions)

This base class provides the interface that all policy implementations
must follow.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any


class Policy(ABC):
    """
    Abstract base class for policies.

    A policy defines how an agent selects actions given states.
    All policy implementations should inherit from this class.

    The key methods are:
    - get_action: Select an action given a state
    - get_params: Get the policy parameters (for saving/loading)
    - set_params: Set the policy parameters
    """

    @abstractmethod
    def get_action(self, state: np.ndarray) -> float:
        """
        Select an action given the current state.

        Args:
            state: Current environment state [x, x_dot, theta, theta_dot]

        Returns:
            Action to take (force to apply to cart)
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the policy parameters.

        Returns:
            Dictionary containing all policy parameters
        """
        pass

    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set the policy parameters.

        Args:
            params: Dictionary containing policy parameters
        """
        pass

    def get_num_params(self) -> int:
        """
        Get the total number of trainable parameters.

        Returns:
            Number of parameters (default: 0)
        """
        return 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.get_num_params()})"
