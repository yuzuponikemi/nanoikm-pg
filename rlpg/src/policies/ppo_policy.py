"""
Proximal Policy Optimization (PPO) Policy

Implements PPO-Clip as described in:
    Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
    https://arxiv.org/abs/1707.06347

Key ideas
---------
1. **Clipped Surrogate Objective**
   L_CLIP = E[min(r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t)]
   where r_t = π_θ(a|s) / π_θ_old(a|s)

   Clipping prevents the new policy from moving too far from the old one,
   providing more stable updates than vanilla policy gradient.

2. **GAE – Generalized Advantage Estimation** (Schulman et al., 2015)
   δ_t  = r_t + γ V(s_{t+1}) - V(s_t)
   A_t  = Σ_{l=0}^{T} (γλ)^l · δ_{t+l}

   The λ parameter interpolates between:
     λ=0 → TD(0) advantage (low variance, high bias)
     λ=1 → Monte Carlo advantage (high variance, low bias)

3. **Shared Actor-Critic network** with separate output heads.

4. **Multiple update epochs** per rollout batch (PPO-specific).

Algorithm flow (per iteration)
-------------------------------
    1. Collect T-step rollout with current policy.
    2. Compute GAE advantages and returns.
    3. Normalise advantages.
    4. For K epochs: sample mini-batches → compute clip loss + value loss
       → gradient step.
    5. Repeat.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import Policy

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared Actor-Critic Network
# ---------------------------------------------------------------------------


class PPOActorCriticNet(nn.Module):
    """
    Shared-body Actor-Critic network.

    Architecture
    ------------
    state_dim  →  hidden[0] → ReLU → … → hidden[-1] → ReLU
                                                         ├── actor_head  → action_mean (Tanh)
                                                         └── critic_head → V(s)

    Sharing the body reduces parameters and lets the policy and value
    function benefit from the same feature representation.
    """

    def __init__(
        self,
        state_dim: int = 4,
        hidden_sizes: List[int] = [64, 64],
    ) -> None:
        super().__init__()

        # ---- Shared trunk ----
        trunk_layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden_sizes:
            trunk_layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.trunk = nn.Sequential(*trunk_layers)

        # ---- Actor head: outputs action mean in (-1, 1) via Tanh ----
        self.actor_head = nn.Sequential(nn.Linear(prev, 1), nn.Tanh())

        # ---- Critic head: outputs scalar V(s) ----
        self.critic_head = nn.Linear(prev, 1)

        # ---- Log-std as a learnable parameter (action-independent) ----
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Forward pass.

        Returns
        -------
        action_mean : Tensor  – action mean in (-1, 1)
        value       : Tensor  – state value V(s)
        log_std     : Tensor  – log standard deviation (shared)
        """
        features = self.trunk(x)
        action_mean = self.actor_head(features)
        value = self.critic_head(features)
        return action_mean, value, self.log_std.expand_as(action_mean)


# ---------------------------------------------------------------------------
# PPO Rollout Buffer
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """
    On-policy rollout buffer for PPO.

    Unlike the DQN replay buffer, this buffer stores a fixed-length
    rollout and is cleared after each policy update.

    Stores
    ------
    states, actions, log_probs (old policy), rewards, dones, values
    """

    def __init__(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[float] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def push(
        self,
        state: np.ndarray,
        action: float,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """Store one transition."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self) -> None:
        """Clear the buffer after a policy update."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPO Policy
# ---------------------------------------------------------------------------


class PPOPolicy(Policy):
    """
    Proximal Policy Optimization (PPO-Clip) policy.

    Usage pattern
    -------------
    1. Collect a rollout using ``get_action_train()`` and ``push()``.
    2. Call ``update()`` at the end of the rollout.
    3. Repeat.

    Parameters
    ----------
    hidden_sizes : list of int
        Hidden layer widths for the shared trunk.
    action_low : float
        Minimum continuous action value.
    action_high : float
        Maximum continuous action value.
    gamma : float
        Discount factor for return computation.
    gae_lambda : float
        λ for GAE advantage estimation.
    clip_eps : float
        ε for PPO clipping (controls how far the new policy can deviate).
    lr : float
        Adam learning rate.
    n_epochs : int
        Number of gradient-update epochs per rollout.
    mini_batch_size : int
        Mini-batch size for each epoch's gradient step.
    value_coef : float
        Weight for the value (critic) loss term.
    entropy_coef : float
        Weight for the entropy bonus (encourages exploration).
    max_grad_norm : float
        Gradient clipping threshold.

    Example
    -------
    >>> policy = PPOPolicy(hidden_sizes=[64, 64])
    >>> state = env.reset()
    >>> action, log_prob, value = policy.get_action_train(state)
    >>> policy.push(state, action, log_prob, reward, done, value)
    >>> # after rollout:
    >>> stats = policy.update(last_value=0.0)
    """

    def __init__(
        self,
        hidden_sizes: List[int] = [64, 64],
        action_low: float = -10.0,
        action_high: float = 10.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        n_epochs: int = 10,
        mini_batch_size: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for PPOPolicy. "
                "Install it with: pip install torch"
            )

        self.hidden_sizes = hidden_sizes
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.lr = lr
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Action scaling: network outputs ∈ (-1, 1), we scale to [action_low, action_high]
        self.action_scale = (action_high - action_low) / 2.0
        self.action_offset = (action_high + action_low) / 2.0

        # ---- Network & optimiser ----
        self.net = PPOActorCriticNet(state_dim=4, hidden_sizes=hidden_sizes)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # ---- Rollout buffer ----
        self.buffer = RolloutBuffer()

        # ---- Stats ----
        self._update_count = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_action(self, state: np.ndarray) -> float:
        """
        Deterministic action for evaluation (no exploration).

        Uses the Actor mean without sampling.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_mean, _, _ = self.net(state_t)
        action = action_mean.item() * self.action_scale + self.action_offset
        return float(np.clip(action, self.action_low, self.action_high))

    def get_action_train(
        self, state: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Stochastic action for training (samples from Gaussian policy).

        Returns
        -------
        action   : float  – continuous action value in [action_low, action_high]
        log_prob : float  – log π(action|state) under current policy
        value    : float  – critic estimate V(state)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_mean, value, log_std = self.net(state_t)

        std = log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        # Scale from (-∞, ∞) → clip to valid range
        action = raw_action.item() * self.action_scale + self.action_offset
        action = float(np.clip(action, self.action_low, self.action_high))

        return action, float(log_prob.item()), float(value.item())

    # ------------------------------------------------------------------
    # Buffer interface
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: float,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """Store one transition in the rollout buffer."""
        self.buffer.push(state, action, log_prob, reward, done, value)

    # ------------------------------------------------------------------
    # GAE advantage computation
    # ------------------------------------------------------------------

    def _compute_gae(
        self, rewards: List[float], values: List[float], dones: List[bool], last_value: float
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Compute GAE advantages and discounted returns.

        Parameters
        ----------
        rewards     : list of per-step rewards
        values      : list of V(s_t) from the old policy
        dones       : list of episode-end flags
        last_value  : V(s_T) bootstrap value (0 if terminal)

        Returns
        -------
        advantages : Tensor (T,)  – GAE advantage estimates
        returns    : Tensor (T,)  – targets for the value function
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = last_value

        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + np.array(values, dtype=np.float32)
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Perform PPO-Clip update over the collected rollout.

        Parameters
        ----------
        last_value : float
            Bootstrap value V(s_T) for GAE. Use 0.0 if the episode ended,
            or V(s_T) if the rollout was truncated mid-episode.

        Returns
        -------
        dict with 'policy_loss', 'value_loss', 'entropy', 'total_loss'
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        # ---- Unpack buffer ----
        states_np = np.array(self.buffer.states, dtype=np.float32)
        actions_np = np.array(self.buffer.actions, dtype=np.float32)
        old_log_probs_np = np.array(self.buffer.log_probs, dtype=np.float32)

        advantages, returns = self._compute_gae(
            self.buffer.rewards, self.buffer.values, self.buffer.dones, last_value
        )

        # Normalise advantages (zero mean, unit std) for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.FloatTensor(states_np)
        actions_t = torch.FloatTensor(actions_np).unsqueeze(1)
        old_log_probs_t = torch.FloatTensor(old_log_probs_np)

        T = len(self.buffer)
        indices = list(range(T))

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        # ---- K epochs of mini-batch updates ----
        for _ in range(self.n_epochs):
            random.shuffle(indices)
            for start in range(0, T, self.mini_batch_size):
                batch_idx = indices[start : start + self.mini_batch_size]
                if len(batch_idx) == 0:
                    continue

                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                # ---- Forward pass with current policy ----
                action_mean, values_pred, log_std = self.net(b_states)
                std = log_std.exp()
                dist = torch.distributions.Normal(action_mean, std)

                # Recover raw (un-scaled) actions for log_prob
                raw_actions = (b_actions - self.action_offset) / self.action_scale
                new_log_probs = dist.log_prob(raw_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # ---- PPO Clipped Objective ----
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- Value Loss (MSE) ----
                value_loss = nn.functional.mse_loss(
                    values_pred.squeeze(-1), b_returns
                )

                # ---- Combined Loss ----
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.buffer.clear()
        self._update_count += 1

        if n_updates == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "total_loss": (total_policy_loss + self.value_coef * total_value_loss) / n_updates,
        }

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Return all policy parameters for saving/loading."""
        return {
            "net_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "hidden_sizes": self.hidden_sizes,
            "action_low": self.action_low,
            "action_high": self.action_high,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_eps": self.clip_eps,
            "lr": self.lr,
            "update_count": self._update_count,
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Load policy parameters."""
        if "net_state" in params:
            self.net.load_state_dict(params["net_state"])
        if "optimizer_state" in params:
            self.optimizer.load_state_dict(params["optimizer_state"])
        if "update_count" in params:
            self._update_count = params["update_count"]

    def get_num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.net.parameters())

    def __repr__(self) -> str:
        return (
            f"PPOPolicy(hidden={self.hidden_sizes}, "
            f"clip_eps={self.clip_eps}, "
            f"updates={self._update_count})"
        )
