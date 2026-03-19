"""
REINFORCE Policy (Policy Gradient)

Implements the REINFORCE algorithm (Williams, 1992) for the inverted pendulum.

Algorithm overview:
1. Run one full episode using the current stochastic policy
2. Compute discounted returns G_t for each timestep
3. Update policy parameters: theta += alpha * sum( G_t * grad log pi(a_t|s_t) )

Key properties vs Q-learning (07):
- Works directly with continuous action space (no discretization needed)
- On-policy: only uses samples from the current policy
- Monte Carlo: waits until episode end before updating
- Higher variance than TD methods, but unbiased

Loss function:
    L = -sum( log pi(a_t | s_t) * G_t )
    (negative because we maximize J but minimize loss)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple

try:
    import torch
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base import Policy
from .neural_policy import NeuralNetworkPolicy


class REINFORCEPolicy(Policy):
    """
    REINFORCE policy gradient agent.

    Uses NeuralNetworkPolicy as a component (composition) to keep
    network architecture and gradient-based learning separate.
    The optimizer lives here so learning rate scheduling and gradient
    clipping are managed in one place.

    Parameters
    ----------
    hidden_sizes : list[int]
        Hidden layer sizes for the neural network. Default: [64, 64]
    lr : float
        Adam optimizer learning rate. Default: 3e-4
    gamma : float
        Discount factor for compute_returns(). Default: 0.99
    action_std : float
        Gaussian noise std for stochastic action sampling. Default: 0.5
    use_baseline : bool
        If True, subtract a moving-average baseline from returns to
        reduce variance. Default: True
    baseline_momentum : float
        EMA coefficient for the baseline. Default: 0.99

    Examples
    --------
    >>> policy = REINFORCEPolicy()
    >>> state = env.reset()
    >>> action = policy.get_action(state)          # inference (no grad)
    >>> action, lp = policy.get_action_train(state) # training (keeps graph)
    """

    def __init__(
        self,
        hidden_sizes: List[int] = [64, 64],
        lr: float = 3e-4,
        gamma: float = 0.99,
        action_std: float = 0.5,
        use_baseline: bool = True,
        baseline_momentum: float = 0.99,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for REINFORCEPolicy. "
                "Install with: pip install torch"
            )

        self.gamma = gamma
        self.action_std = action_std
        self.use_baseline = use_baseline
        self.baseline_momentum = baseline_momentum

        # Composition: network owned by the policy
        self._network = NeuralNetworkPolicy(
            hidden_sizes=hidden_sizes,
            action_low=-10.0,
            action_high=10.0,
        )

        # Optimizer owned here so lr scheduling / grad clipping is centralised
        self._optimizer = optim.Adam(
            self._network.network.parameters(), lr=lr
        )

        # Exponential moving average baseline (single scalar, no extra params)
        self._baseline: float = 0.0

        # Per-episode trajectory buffers, cleared after each update
        self._log_probs: List["torch.Tensor"] = []
        self._rewards: List[float] = []

    # ------------------------------------------------------------------
    # Policy ABC interface
    # ------------------------------------------------------------------

    def get_action(self, state: np.ndarray) -> float:
        """
        Inference-time action (no gradient graph retained).

        Returns
        -------
        float : force value in [-10, 10]
        """
        action, _ = self._network.get_action_and_log_prob(
            state, action_std=self.action_std
        )
        return action

    def get_params(self) -> Dict[str, Any]:
        return self._network.get_params()

    def set_params(self, params: Dict[str, Any]) -> None:
        self._network.set_params(params)

    def get_num_params(self) -> int:
        return self._network.get_num_params()

    # ------------------------------------------------------------------
    # REINFORCE-specific methods
    # ------------------------------------------------------------------

    def get_action_train(
        self, state: np.ndarray
    ) -> Tuple[float, "torch.Tensor"]:
        """
        Training-time action. Returns action AND log_prob with grad graph.

        Use this inside the training loop; use get_action() for evaluation.

        Returns
        -------
        action : float
        log_prob : torch.Tensor (scalar, requires_grad=True)
        """
        return self._network.get_action_and_log_prob(
            state, action_std=self.action_std
        )

    def store_transition(self, log_prob: "torch.Tensor", reward: float) -> None:
        """Accumulate one step's log_prob and reward into the episode buffer."""
        self._log_probs.append(log_prob)
        self._rewards.append(reward)

    def update_on_episode(self) -> float:
        """
        Perform one REINFORCE gradient update using the buffered episode.

        Called once per episode after the episode ends.
        Clears the trajectory buffer after updating.

        Returns
        -------
        float : loss value (for logging)
        """
        from src.utils.training import compute_returns, normalize_returns

        returns_np = compute_returns(self._rewards, gamma=self.gamma)
        returns_np = normalize_returns(returns_np)

        if self.use_baseline:
            episode_mean = float(np.mean(returns_np))
            self._baseline = (
                self.baseline_momentum * self._baseline
                + (1.0 - self.baseline_momentum) * episode_mean
            )
            returns_np = returns_np - self._baseline

        returns_t = torch.FloatTensor(returns_np)
        log_probs_t = torch.stack(self._log_probs)  # shape: (T,)

        # L = -E[ log pi(a|s) * G_t ]  (sum over timesteps)
        loss = -(log_probs_t * returns_t).sum()

        self._optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients (common in PG)
        torch.nn.utils.clip_grad_norm_(
            self._network.network.parameters(), max_norm=1.0
        )
        self._optimizer.step()

        # Clear buffers for next episode
        self._log_probs = []
        self._rewards = []

        return float(loss.detach())

    def __repr__(self) -> str:
        return (
            f"REINFORCEPolicy("
            f"params={self.get_num_params()}, "
            f"gamma={self.gamma}, "
            f"use_baseline={self.use_baseline})"
        )
