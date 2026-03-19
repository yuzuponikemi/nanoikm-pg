"""
REINFORCE Policy

Implements the REINFORCE (Monte-Carlo Policy Gradient) algorithm.
Uses a stochastic Gaussian policy with an EMA baseline for variance reduction.

Update rule:
    theta += alpha * sum_t [ (G_t - b) * grad log pi(a_t | s_t) ]

where b is an exponential moving average baseline.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from .base import Policy

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ReinforcePolicy(Policy):
    """
    REINFORCE policy with EMA baseline.

    Attributes:
        actor_net:   Neural network for action mean
        optimizer:   Adam optimizer
        action_low:  Minimum action value
        action_high: Maximum action value
        action_std:  Gaussian exploration noise std
        gamma:       Discount factor
        lr:          Learning rate
        baseline:    EMA baseline for variance reduction
    """

    def __init__(
        self,
        hidden_sizes: List[int] = [64, 64],
        action_low: float = -10.0,
        action_high: float = 10.0,
        action_std: float = 0.5,
        gamma: float = 0.99,
        lr: float = 3e-4,
        baseline_decay: float = 0.99,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ReinforcePolicy.")

        self.hidden_sizes = hidden_sizes
        self.action_low = action_low
        self.action_high = action_high
        self.action_std = action_std
        self.gamma = gamma
        self.lr = lr
        self.baseline_decay = baseline_decay
        self.baseline = 0.0

        self.action_scale = (action_high - action_low) / 2.0
        self.action_offset = (action_high + action_low) / 2.0

        # Build actor network
        layers = []
        prev = 4
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Tanh()]
        self.actor_net = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.actor_net.parameters(), lr=lr)

        # Episode buffers
        self._log_probs: List["torch.Tensor"] = []
        self._rewards: List[float] = []

    def get_action(self, state: np.ndarray) -> float:
        """Deterministic action for inference."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            raw = self.actor_net(state_tensor)
        return raw.item() * self.action_scale + self.action_offset

    def get_action_train(self, state: np.ndarray) -> Tuple[float, "torch.Tensor"]:
        """Stochastic action for training (returns action and log_prob)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mean_raw = self.actor_net(state_tensor)
        mean_action = mean_raw * self.action_scale + self.action_offset

        std = torch.tensor([[self.action_std]])
        dist = torch.distributions.Normal(mean_action, std)
        action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor)

        action_clipped = torch.clamp(action_tensor, self.action_low, self.action_high)
        return action_clipped.item(), log_prob

    def store_transition(self, log_prob: "torch.Tensor", reward: float) -> None:
        """Store one transition."""
        self._log_probs.append(log_prob)
        self._rewards.append(reward)

    def update_on_episode(self) -> Dict[str, float]:
        """REINFORCE update with EMA baseline."""
        if not self._rewards:
            return {'actor_loss': 0.0, 'total_reward': 0.0}

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_tensor = torch.FloatTensor(returns)

        # EMA baseline
        episode_return = returns[0]
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * episode_return
        baseline_tensor = torch.tensor(self.baseline)

        advantages = returns_tensor - baseline_tensor

        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = torch.cat(self._log_probs)
        actor_loss = -(log_probs * advantages).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        total_reward = sum(self._rewards)

        self._log_probs = []
        self._rewards = []

        return {
            'actor_loss': actor_loss.item(),
            'total_reward': total_reward,
        }

    def get_params(self) -> Dict[str, Any]:
        return {
            'actor_state': self.actor_net.state_dict(),
            'hidden_sizes': self.hidden_sizes,
            'action_low': self.action_low,
            'action_high': self.action_high,
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        if 'actor_state' in params:
            self.actor_net.load_state_dict(params['actor_state'])

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.actor_net.parameters())
