"""
Actor-Critic Policy (A2C)

This module implements the Advantage Actor-Critic (A2C) algorithm.

The Actor-Critic architecture separates the policy (Actor) from the
value function (Critic):

    Actor:  pi(a|s)  -- selects actions stochastically
    Critic: V(s)     -- estimates state value to compute advantage

The key insight is that instead of using a baseline (like a running mean),
we use the Critic's value estimate to compute the Advantage:

    A(s, a) = Q(s, a) - V(s)
            ≈ G_t - V(s_t)   (Monte Carlo estimate)

where G_t is the discounted return from timestep t.

This reduces the variance of the policy gradient compared to REINFORCE,
while introducing some bias from the Critic's approximation.

Update rules:
    Actor loss:  -log pi(a|s) * A(s, a)   (policy gradient)
    Critic loss: (G_t - V(s_t))^2          (MSE regression)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .base import Policy

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CriticNetwork(nn.Module):
    """
    Critic network that estimates state value V(s).

    Architecture:
        Input (state_dim) -> Hidden layers -> Output (1)
    """

    def __init__(self, state_dim: int = 4, hidden_sizes: List[int] = [64, 64]):
        super().__init__()
        layers = []
        prev_size = state_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))  # Single value output
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class ActorCriticPolicy(Policy):
    """
    Advantage Actor-Critic (A2C) policy.

    Combines a stochastic Actor (neural network policy) with a
    Critic (state-value network) for more stable policy gradient updates.

    Attributes:
        actor_net:    Neural network for action mean (Actor)
        critic_net:   Neural network for state value (Critic)
        actor_optimizer:  Optimizer for Actor
        critic_optimizer: Optimizer for Critic
        action_low:   Minimum action value
        action_high:  Maximum action value
        action_std:   Standard deviation for action sampling
        gamma:        Discount factor
        actor_lr:     Actor learning rate
        critic_lr:    Critic learning rate

    Example:
        >>> policy = ActorCriticPolicy(hidden_sizes=[64, 64])
        >>> state = np.array([0, 0, 0.1, 0])
        >>> action = policy.get_action(state)
    """

    def __init__(
        self,
        hidden_sizes: List[int] = [64, 64],
        action_low: float = -10.0,
        action_high: float = 10.0,
        action_std: float = 0.5,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
    ):
        """
        Initialize the Actor-Critic policy.

        Args:
            hidden_sizes: Hidden layer sizes for both Actor and Critic
            action_low:   Minimum action value
            action_high:  Maximum action value
            action_std:   Standard deviation of Gaussian action noise
            gamma:        Discount factor for return computation
            actor_lr:     Learning rate for the Actor network
            critic_lr:    Learning rate for the Critic network
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ActorCriticPolicy. "
                "Install it with: pip install torch"
            )

        self.hidden_sizes = hidden_sizes
        self.action_low = action_low
        self.action_high = action_high
        self.action_std = action_std
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # Action scaling helpers
        self.action_scale = (action_high - action_low) / 2.0
        self.action_offset = (action_high + action_low) / 2.0

        # ---- Actor network (output: action mean in [-1, 1] via Tanh) ----
        actor_layers = []
        prev = 4  # state_dim
        for h in hidden_sizes:
            actor_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        actor_layers += [nn.Linear(prev, 1), nn.Tanh()]
        self.actor_net = nn.Sequential(*actor_layers)

        # ---- Critic network (output: scalar V(s)) ----
        self.critic_net = CriticNetwork(state_dim=4, hidden_sizes=hidden_sizes)

        # ---- Optimizers ----
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_lr)

        # ---- Episode storage ----
        self._log_probs: List["torch.Tensor"] = []
        self._rewards: List[float] = []
        self._values: List["torch.Tensor"] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_action(self, state: np.ndarray) -> float:
        """
        Select an action for evaluation/inference (no gradients).

        Uses the Actor's mean output without stochastic sampling.

        Args:
            state: Current state array

        Returns:
            Deterministic action (Actor mean, scaled to action range)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            raw = self.actor_net(state_tensor)
        action = raw.item() * self.action_scale + self.action_offset
        return action

    def get_action_train(self, state: np.ndarray) -> Tuple[float, "torch.Tensor", "torch.Tensor"]:
        """
        Select an action for training (keeps gradients and Critic value).

        Samples from a Gaussian distribution centred on the Actor's output,
        and queries the Critic for V(s).

        Args:
            state: Current state array

        Returns:
            Tuple of (action_float, log_prob_tensor, value_tensor)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Actor forward pass (with grad)
        mean_raw = self.actor_net(state_tensor)
        mean_action = mean_raw * self.action_scale + self.action_offset

        # Gaussian sampling
        std = torch.tensor([[self.action_std]])
        dist = torch.distributions.Normal(mean_action, std)
        action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor)

        # Clip to valid range
        action_clipped = torch.clamp(action_tensor, self.action_low, self.action_high)

        # Critic forward pass
        value = self.critic_net(state_tensor)

        return action_clipped.item(), log_prob, value

    # ------------------------------------------------------------------
    # Transition storage
    # ------------------------------------------------------------------

    def store_transition(
        self,
        log_prob: "torch.Tensor",
        reward: float,
        value: "torch.Tensor",
    ) -> None:
        """
        Store one transition for the current episode.

        Args:
            log_prob: Log probability of the taken action (from get_action_train)
            reward:   Reward received after the action
            value:    Critic's value estimate V(s_t) (from get_action_train)
        """
        self._log_probs.append(log_prob)
        self._rewards.append(reward)
        self._values.append(value)

    # ------------------------------------------------------------------
    # A2C update
    # ------------------------------------------------------------------

    def update_on_episode(self) -> Dict[str, float]:
        """
        Perform an A2C update at the end of an episode.

        Algorithm:
            1. Compute Monte-Carlo returns G_t (discounted).
            2. Advantage A_t = G_t - V(s_t)  (Critic baseline).
            3. Actor loss  = -mean(log_prob * A_t.detach())
            4. Critic loss = mean((G_t - V(s_t))^2)
            5. Gradient descent on both networks.

        Returns:
            Dict with 'actor_loss', 'critic_loss', 'total_reward'
        """
        if not self._rewards:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'total_reward': 0.0}

        # --- Compute discounted returns ---
        returns = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_tensor = torch.FloatTensor(returns)

        # Optionally normalise for stability
        if returns_tensor.std() > 1e-8:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        # --- Stack stored tensors ---
        log_probs = torch.cat(self._log_probs)          # (T,)
        values = torch.cat(self._values).squeeze(-1)    # (T,)

        # --- Advantages ---
        advantages = returns_tensor - values.detach()

        # --- Actor loss (policy gradient with Advantage baseline) ---
        actor_loss = -(log_probs * advantages).mean()

        # --- Critic loss (MSE) ---
        critic_loss = nn.functional.mse_loss(values, returns_tensor)

        # --- Backprop ---
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        total_reward = sum(self._rewards)

        # Clear episode buffers
        self._log_probs = []
        self._rewards = []
        self._values = []

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_reward': total_reward,
        }

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Return all policy parameters for saving/loading."""
        return {
            'actor_state': self.actor_net.state_dict(),
            'critic_state': self.critic_net.state_dict(),
            'hidden_sizes': self.hidden_sizes,
            'action_low': self.action_low,
            'action_high': self.action_high,
            'action_std': self.action_std,
            'gamma': self.gamma,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Load policy parameters."""
        if 'actor_state' in params:
            self.actor_net.load_state_dict(params['actor_state'])
        if 'critic_state' in params:
            self.critic_net.load_state_dict(params['critic_state'])
        if 'action_low' in params:
            self.action_low = params['action_low']
        if 'action_high' in params:
            self.action_high = params['action_high']
            self.action_scale = (self.action_high - self.action_low) / 2.0
            self.action_offset = (self.action_high + self.action_low) / 2.0
        if 'action_std' in params:
            self.action_std = params['action_std']
        if 'gamma' in params:
            self.gamma = params['gamma']

    def get_num_params(self) -> int:
        """Total number of trainable parameters (Actor + Critic)."""
        actor_params = sum(p.numel() for p in self.actor_net.parameters())
        critic_params = sum(p.numel() for p in self.critic_net.parameters())
        return actor_params + critic_params
