"""
Soft Actor-Critic (SAC) Policy

Implements SAC as described in Haarnoja et al. 2018
"Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
with a Stochastic Actor"

Key ideas
---------
1. **Maximum entropy framework**
   The agent maximises the sum of rewards *and* the entropy of its policy:

       J(π) = Σ E[r(s_t, a_t) + α · H(π(·|s_t))]

   Higher entropy → more exploration; α (temperature) controls the trade-off.

2. **Twin Critic networks (Clipped Double-Q)**
   Two independent Q-networks Q₁, Q₂ are trained simultaneously.
   TD targets use min(Q₁, Q₂) to avoid overestimation:

       y = r + γ · (min(Q₁(s', ã'), Q₂(s', ã')) − α · log π(ã'|s'))

   where ã' ~ π(·|s') is sampled via the reparameterization trick.

3. **Reparameterization trick (Gaussian policy)**
   Instead of sampling a ~ π(·|s) directly (no gradient), we write:

       a = tanh(μ_θ(s) + σ_θ(s) · ε),   ε ~ N(0, I)

   This lets gradients flow through the sampling step.
   The log-probability of a squashed Gaussian action is:

       log π(a|s) = log N(u|μ, σ²) − Σ log(1 − tanh²(u_i))

4. **Automatic temperature tuning**
   α is treated as a learnable parameter optimised to satisfy
   a target entropy constraint H_target ≈ −dim(A):

       L(α) = −α · (log π(a|s) + H_target)

5. **Soft target update (Polyak averaging)**
   Target critics are updated by exponential moving average:

       θ⁻ ← τ·θ + (1−τ)·θ⁻

Comparison with existing rlpg algorithms
-----------------------------------------
| Feature            | DQN (10)  | PPO (11)  | SAC (12)  |
|--------------------|-----------|-----------|-----------|
| Action space       | discrete  | continuous| continuous|
| On/Off-policy      | off       | on        | off       |
| Replay buffer      | yes       | no        | yes       |
| Entropy bonus      | no        | implicit  | explicit  |
| Target network     | hard copy | —         | soft copy |
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from .base import Policy

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-6   # numerical stability for log(1 - tanh^2)


# ---------------------------------------------------------------------------
# Replay Buffer (same structure as DQN's but included here for self-containment)
# ---------------------------------------------------------------------------

class SACTransition(NamedTuple):
    """Single experience tuple stored in the SAC replay buffer."""
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool


class SACReplayBuffer:
    """
    Fixed-size circular replay buffer for SAC (continuous actions).

    Parameters
    ----------
    capacity : int
        Maximum number of transitions (FIFO eviction).

    Example
    -------
    >>> buf = SACReplayBuffer(capacity=100_000)
    >>> buf.push(state, 1.5, 1.0, next_state, False)
    >>> batch = buf.sample(64)
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._buffer: Deque[SACTransition] = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add one transition to the buffer."""
        self._buffer.append(
            SACTransition(
                state=np.asarray(state, dtype=np.float32),
                action=float(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> List[SACTransition]:
        """Sample a random mini-batch of transitions."""
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"SACReplayBuffer(size={len(self)}/{self.capacity})"


# ---------------------------------------------------------------------------
# Actor Network (Gaussian policy with squashing)
# ---------------------------------------------------------------------------

class SACActorNet(nn.Module):
    """
    Stochastic Actor network that outputs a squashed Gaussian policy.

    Forward pass returns (mean, log_std) in the unsquashed space.
    Sampling and log-probability computation happen in SACPolicy.

    Architecture
    ------------
    state_dim → hidden… → [mean_head (action_dim), log_std_head (action_dim)]

    Parameters
    ----------
    state_dim : int
        Dimensionality of the input state.
    action_dim : int
        Dimensionality of the continuous action space (1 for InvertedPendulum).
    hidden_sizes : list of int
        Hidden layer widths.
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 1,
        hidden_sizes: List[int] = [256, 256],
    ) -> None:
        super().__init__()

        # Shared trunk
        layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)

        # Two output heads (mean and log_std)
        self.mean_head = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)

    def forward(
        self, x: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Returns
        -------
        mean : Tensor of shape (batch, action_dim)
        log_std : Tensor of shape (batch, action_dim)  clamped to [LOG_STD_MIN, LOG_STD_MAX]
        """
        h = self.trunk(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std


# ---------------------------------------------------------------------------
# Critic Network (Q-function)
# ---------------------------------------------------------------------------

class SACCriticNet(nn.Module):
    """
    Q-network: Q(s, a) → scalar.

    Takes (state, action) concatenated as input.

    Parameters
    ----------
    state_dim : int
    action_dim : int
    hidden_sizes : list of int
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 1,
        hidden_sizes: List[int] = [256, 256],
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = state_dim + action_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self, state: "torch.Tensor", action: "torch.Tensor"
    ) -> "torch.Tensor":
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# SAC Policy
# ---------------------------------------------------------------------------

class SACPolicy(Policy):
    """
    Soft Actor-Critic (SAC) policy.

    Continuous-action off-policy algorithm that maximises entropy-regularised
    expected return.  Compared with Actor-Critic (A2C, notebook 09) and DQN
    (notebook 10), SAC adds:

    * Maximum-entropy objective for automatic exploration.
    * Twin-critic clipping to reduce Q-overestimation.
    * Reparameterization trick for low-variance policy gradients.
    * Polyak-averaged target critics for stable TD targets.
    * Automatic temperature (α) tuning.

    Parameters
    ----------
    state_dim : int
        State vector size (4 for InvertedPendulum).
    action_dim : int
        Action vector size (1 for InvertedPendulum).
    action_scale : float
        Half-range of the action space (max_force / 1.0 because tanh ∈ [−1, 1]).
    action_bias : float
        Center of the action space.
    hidden_sizes : list of int
        Hidden layer widths for actor and critics.
    lr : float
        Learning rate for all optimizers.
    gamma : float
        Discount factor.
    tau : float
        Polyak averaging coefficient for soft target update.
    alpha : float
        Initial entropy temperature (overridden by auto-tuning).
    auto_alpha : bool
        If True, automatically tune α to hit target_entropy.
    target_entropy : float or None
        Desired entropy. Defaults to −action_dim.
    batch_size : int
        Mini-batch size for gradient updates.
    buffer_capacity : int
        Replay buffer capacity.
    warmup_steps : int
        Number of random-action steps before learning starts.

    Example
    -------
    >>> policy = SACPolicy()
    >>> action = policy.get_action(env.reset())          # deterministic mean
    >>> action = policy.get_action_train(env.reset())    # stochastic sample
    >>> stats = policy.update()                          # one gradient step
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 1,
        action_scale: float = 10.0,
        action_bias: float = 0.0,
        hidden_sizes: List[int] = [256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        batch_size: int = 256,
        buffer_capacity: int = 100_000,
        warmup_steps: int = 1_000,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for SACPolicy. "
                "Install it with: pip install torch"
            )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

        # --- Networks ---
        self.actor = SACActorNet(state_dim, action_dim, hidden_sizes)
        self.critic1 = SACCriticNet(state_dim, action_dim, hidden_sizes)
        self.critic2 = SACCriticNet(state_dim, action_dim, hidden_sizes)
        # Target critics (frozen copies)
        self.target_critic1 = SACCriticNet(state_dim, action_dim, hidden_sizes)
        self.target_critic2 = SACCriticNet(state_dim, action_dim, hidden_sizes)
        self._hard_update_targets()   # initialise targets = critics

        # --- Optimizers ---
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # --- Temperature α ---
        self.auto_alpha = auto_alpha
        self.target_entropy = (
            target_entropy if target_entropy is not None else -float(action_dim)
        )
        if auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # --- Replay buffer ---
        self.buffer = SACReplayBuffer(capacity=buffer_capacity)

        # --- Counters ---
        self._step_count = 0
        self._total_pushes = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hard_update_targets(self) -> None:
        """Copy online critic weights to target critics."""
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def _soft_update_targets(self) -> None:
        """Polyak averaging: θ⁻ ← τ·θ + (1−τ)·θ⁻"""
        for p, tp in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def _sample_action(
        self, state_t: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Sample action via the reparameterization trick.

        Returns
        -------
        action_scaled : Tensor (batch, action_dim)  — in [action_bias ± action_scale]
        log_prob : Tensor (batch, 1)                — log π(a|s)
        """
        mean, log_std = self.actor(state_t)
        std = log_std.exp()

        # Reparameterization: u = mean + std * eps,  eps ~ N(0,I)
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()                          # gradient flows through ε
        a_raw = torch.tanh(u)                         # squash to (−1, 1)

        # Squashed Gaussian log-prob
        # log π(a|s) = log N(u|μ,σ²) − Σ log(1 − tanh²(u_i))
        log_prob = normal.log_prob(u) - torch.log(1 - a_raw.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # (batch, 1)

        action_scaled = a_raw * self.action_scale + self.action_bias
        return action_scaled, log_prob

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def get_action(self, state: np.ndarray) -> float:
        """
        Deterministic action (use actor mean) – for evaluation only.

        Returns
        -------
        float
            Continuous force value.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self.actor(state_t)
            action = torch.tanh(mean) * self.action_scale + self.action_bias
        return float(action.item())

    def get_action_train(self, state: np.ndarray) -> float:
        """
        Stochastic action sampled from the policy – use during training.

        Returns
        -------
        float
            Continuous force value sampled from π(·|s).
        """
        # During warmup, return random actions
        if self._total_pushes < self.warmup_steps:
            return float(np.random.uniform(
                self.action_bias - self.action_scale,
                self.action_bias + self.action_scale,
            ))

        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_t, _ = self._sample_action(state_t)
        return float(action_t.item())

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
        self._total_pushes += 1

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def update(self) -> Optional[Dict[str, float]]:
        """
        Sample a mini-batch and perform one SAC gradient step.

        Steps
        -----
        1. Compute TD targets using target critics and policy entropy.
        2. Update Critic 1 and Critic 2 to minimise Bellman error.
        3. Update Actor to maximise Q − α · log π.
        4. (Optional) Update α to maintain target entropy.
        5. Soft-update target critics.

        Returns
        -------
        dict with 'critic1_loss', 'critic2_loss', 'actor_loss', 'alpha'
        or None if buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(np.stack([t.state for t in batch]))
        actions = torch.FloatTensor([[t.action] for t in batch])
        rewards = torch.FloatTensor([[t.reward] for t in batch])
        next_states = torch.FloatTensor(np.stack([t.next_state for t in batch]))
        dones = torch.FloatTensor([[float(t.done)] for t in batch])

        # ----------------------------------------------------------------
        # 1. Compute TD targets
        # ----------------------------------------------------------------
        with torch.no_grad():
            next_actions, next_log_probs = self._sample_action(next_states)
            q1_next = self.target_critic1(next_states, next_actions)
            q2_next = self.target_critic2(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1.0 - dones) * min_q_next

        # ----------------------------------------------------------------
        # 2. Update Critics
        # ----------------------------------------------------------------
        q1_pred = self.critic1(states, actions)
        q2_pred = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1_pred, target_q)
        critic2_loss = F.mse_loss(q2_pred, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ----------------------------------------------------------------
        # 3. Update Actor
        # ----------------------------------------------------------------
        sampled_actions, log_probs = self._sample_action(states)
        q1_val = self.critic1(states, sampled_actions)
        q2_val = self.critic2(states, sampled_actions)
        min_q_val = torch.min(q1_val, q2_val)

        actor_loss = (self.alpha * log_probs - min_q_val).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------------------------------------------------
        # 4. Update α (auto temperature tuning)
        # ----------------------------------------------------------------
        alpha_loss = 0.0
        if self.auto_alpha:
            with torch.no_grad():
                _, log_probs_for_alpha = self._sample_action(states)
            alpha_loss_t = -(
                self.log_alpha.exp() * (log_probs_for_alpha + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss_t.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss_t.item()

        # ----------------------------------------------------------------
        # 5. Soft-update target critics
        # ----------------------------------------------------------------
        self._soft_update_targets()
        self._step_count += 1

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss,
        }

    # ------------------------------------------------------------------
    # get_params / set_params (Policy interface)
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        params = {
            "actor_state": self.actor.state_dict(),
            "critic1_state": self.critic1.state_dict(),
            "critic2_state": self.critic2.state_dict(),
            "target_critic1_state": self.target_critic1.state_dict(),
            "target_critic2_state": self.target_critic2.state_dict(),
            "alpha": self.alpha,
            "step_count": self._step_count,
            "total_pushes": self._total_pushes,
        }
        if self.auto_alpha:
            params["log_alpha"] = self.log_alpha.data.clone()
        return params

    def set_params(self, params: Dict[str, Any]) -> None:
        if "actor_state" in params:
            self.actor.load_state_dict(params["actor_state"])
        if "critic1_state" in params:
            self.critic1.load_state_dict(params["critic1_state"])
        if "critic2_state" in params:
            self.critic2.load_state_dict(params["critic2_state"])
        if "target_critic1_state" in params:
            self.target_critic1.load_state_dict(params["target_critic1_state"])
        if "target_critic2_state" in params:
            self.target_critic2.load_state_dict(params["target_critic2_state"])
        if "alpha" in params:
            self.alpha = params["alpha"]
        if "log_alpha" in params and self.auto_alpha:
            self.log_alpha.data.copy_(params["log_alpha"])
        if "step_count" in params:
            self._step_count = params["step_count"]
        if "total_pushes" in params:
            self._total_pushes = params["total_pushes"]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return current training statistics."""
        return {
            "alpha": self.alpha,
            "step_count": self._step_count,
            "total_pushes": self._total_pushes,
            "buffer_size": len(self.buffer),
        }

    def get_num_params(self) -> int:
        """Total number of trainable parameters."""
        total = 0
        for net in [self.actor, self.critic1, self.critic2]:
            total += sum(p.numel() for p in net.parameters())
        return total

    def __repr__(self) -> str:
        return (
            f"SACPolicy(action_scale={self.action_scale}, "
            f"hidden={self.hidden_sizes}, "
            f"α={self.alpha:.4f}, "
            f"steps={self._step_count})"
        )
