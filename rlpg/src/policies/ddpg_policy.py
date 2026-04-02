"""
Deep Deterministic Policy Gradient (DDPG) Policy

Implements DDPG as described in:
    Lillicrap et al., "Continuous Control with Deep Reinforcement Learning" (2015)
    https://arxiv.org/abs/1509.02971

Key ideas
---------
1. **Deterministic Actor-Critic**
   Unlike stochastic policies (PPO, SAC), the actor directly outputs a deterministic
   action: a = μ_θ(s). Exploration is done via noise added during training.

   Actor loss:   L = -E[Q(s, μ_θ(s))]  (maximize expected Q-value)
   Critic loss:  L = E[(y - Q(s, a))²] where y = r + γ Q⁻(s', μ⁻(s'))

2. **Experience Replay (Off-policy)**
   Like DQN and SAC, DDPG uses a replay buffer to:
   - Break temporal correlations in data
   - Improve sample efficiency
   - Enable off-policy learning

3. **Target Networks**
   Separate target networks (θ⁻) updated via Polyak averaging:
       θ⁻ ← τ·θ + (1−τ)·θ⁻  (soft update)
   Stabilises learning by slowly moving towards current parameters.

4. **Ornstein-Uhlenbeck (OU) Noise**
   Exploration is done by adding temporally-correlated noise to actions:
       a_train = a + OU_noise
   OU noise is more natural for control than Gaussian noise (dθ = θ(μ - x)dt + σ dW).

5. **Replay Buffer**
   Stores (s, a, r, s', done) transitions and samples random mini-batches
   for off-policy updates.

Comparison with existing rlpg algorithms
-----------------------------------------
| Feature            | DQN (10)  | PPO (11)  | SAC (12)  | DDPG (13) |
|--------------------|-----------|-----------|-----------|-----------|
| Action space       | discrete  | continuous| continuous| continuous|
| On/Off-policy      | off       | on        | off       | off       |
| Actor deterministic| —         | no        | no        | yes       |
| Replay buffer      | yes       | no        | yes       | yes       |
| Exploration type   | ε-greedy  | entropy   | entropy   | OU noise  |
| Target network     | hard copy | —         | soft copy | soft copy |
| Twin critics       | no        | —         | yes       | no        |
"""

from __future__ import annotations

import random
import numpy as np
from collections import deque
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Tuple

from .base import Policy

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck Noise
# ---------------------------------------------------------------------------

class OUNoise:
    """
    Ornstein-Uhlenbeck noise for continuous action exploration.

    Generates temporally-correlated noise that is more natural for
    control problems than uncorrelated Gaussian noise.

    The OU process is defined by:
        dθ = θ(μ - x)dt + σ dW

    Parameters
    ----------
    action_dim : int
        Dimensionality of action space (1 for pendulum).
    mu : float
        Long-term mean (0 by default).
    theta : float
        Mean reversion rate (controls how fast noise reverts to mu).
    sigma : float
        Volatility of the noise process.
    dt : float
        Time step (discrete approximation).
    """

    def __init__(
        self,
        action_dim: int = 1,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.3,
        dt: float = 0.01,
    ) -> None:
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x = np.ones(action_dim) * mu
        self.reset()

    def reset(self) -> None:
        """Reset the noise state."""
        self.x = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        """
        Sample one step of OU noise.

        Returns
        -------
        noise : ndarray
            OU noise for the current step.
        """
        dx = self.theta * (self.mu - self.x) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.x = self.x + dx
        return self.x.copy()


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class DDPGTransition(NamedTuple):
    """Single experience tuple for DDPG."""
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool


class DDPGReplayBuffer:
    """
    Fixed-size circular replay buffer for DDPG.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._buffer: Deque[DDPGTransition] = deque(maxlen=capacity)
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
            DDPGTransition(
                state=np.asarray(state, dtype=np.float32),
                action=float(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> List[DDPGTransition]:
        """Sample a random mini-batch of transitions."""
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"DDPGReplayBuffer(size={len(self)}/{self.capacity})"


# ---------------------------------------------------------------------------
# Actor Network (Deterministic Policy)
# ---------------------------------------------------------------------------

class DDPGActorNet(nn.Module):
    """
    Deterministic Actor network for DDPG.

    Maps state → deterministic action (no stochasticity).
    Output is squashed to [-action_scale, action_scale] via Tanh.

    Architecture
    ------------
    state_dim → hidden[0] → ReLU → … → hidden[-1] → ReLU → 1 → Tanh
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 1,
        hidden_sizes: List[int] = [64, 64],
    ) -> None:
        super().__init__()
        layers = []
        prev = state_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev, hidden_size))
            layers.append(nn.ReLU())
            prev = hidden_size

        # Output: deterministic action in [-1, 1]
        layers.append(nn.Linear(prev, action_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass.

        Args
        ----
        x : Tensor
            State tensor (batch_size, state_dim).

        Returns
        -------
        action : Tensor
            Deterministic action in [-1, 1] (batch_size, action_dim).
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Critic Network (Q-function)
# ---------------------------------------------------------------------------

class DDPGCriticNet(nn.Module):
    """
    Critic network for DDPG.

    Maps (state, action) → Q-value.
    Both state and action are concatenated and passed through the network.

    Architecture
    ------------
    state_dim + action_dim → hidden[0] → ReLU → … → hidden[-1] → ReLU → 1
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 1,
        hidden_sizes: List[int] = [64, 64],
    ) -> None:
        super().__init__()
        layers = []
        prev = state_dim + action_dim  # Concatenate state and action

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev, hidden_size))
            layers.append(nn.ReLU())
            prev = hidden_size

        # Output: scalar Q-value
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)

    def forward(
        self, state: "torch.Tensor", action: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Forward pass.

        Args
        ----
        state : Tensor
            State tensor (batch_size, state_dim).
        action : Tensor
            Action tensor (batch_size, action_dim).

        Returns
        -------
        q_value : Tensor
            Q-value (batch_size, 1).
        """
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# DDPG Policy
# ---------------------------------------------------------------------------

class DDPGPolicy(Policy):
    """
    Deep Deterministic Policy Gradient (DDPG) policy.

    Usage pattern
    -------------
    1. Call ``get_action_train(state)`` to select an action (with OU noise).
    2. Take a step in the environment.
    3. Call ``push(state, action, reward, next_state, done)`` to store transition.
    4. Once buffer has enough samples, call ``update()`` for training.
    5. Repeat.

    Parameters
    ----------
    hidden_sizes : list of int
        Hidden layer sizes for both Actor and Critic.
    action_low : float
        Minimum action value.
    action_high : float
        Maximum action value.
    gamma : float
        Discount factor.
    tau : float
        Soft update coefficient (Polyak averaging).
    actor_lr : float
        Actor network learning rate.
    critic_lr : float
        Critic network learning rate.
    replay_buffer_capacity : int
        Maximum replay buffer size.
    warmup_steps : int
        Number of steps to take with random actions before training.
    """

    def __init__(
        self,
        hidden_sizes: List[int] = [64, 64],
        action_low: float = -10.0,
        action_high: float = 10.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        replay_buffer_capacity: int = 100_000,
        warmup_steps: int = 1000,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DDPGPolicy. "
                "Install it with: pip install torch"
            )

        self.hidden_sizes = hidden_sizes
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # Action scaling
        self.action_scale = (action_high - action_low) / 2.0
        self.action_offset = (action_high + action_low) / 2.0

        # Networks
        self.actor = DDPGActorNet(state_dim=4, action_dim=1, hidden_sizes=hidden_sizes)
        self.critic = DDPGCriticNet(state_dim=4, action_dim=1, hidden_sizes=hidden_sizes)

        # Target networks (initialized with same weights)
        self.actor_target = DDPGActorNet(state_dim=4, action_dim=1, hidden_sizes=hidden_sizes)
        self.critic_target = DDPGCriticNet(state_dim=4, action_dim=1, hidden_sizes=hidden_sizes)
        self._hard_update(self.actor, self.actor_target)
        self._hard_update(self.critic, self.critic_target)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.replay_buffer = DDPGReplayBuffer(capacity=replay_buffer_capacity)

        # Exploration noise
        self.ou_noise = OUNoise(action_dim=1, mu=0.0, theta=0.15, sigma=0.3)

        # Statistics
        self._episode_rewards = []
        self._critic_losses = []
        self._actor_losses = []

    def get_action(self, state: np.ndarray) -> float:
        """
        Select a deterministic action for evaluation (no noise).

        Args
        ----
        state : ndarray
            Current state.

        Returns
        -------
        action : float
            Deterministic action scaled to [action_low, action_high].
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            raw_action = self.actor(state_tensor).squeeze(0).item()
        action = raw_action * self.action_scale + self.action_offset
        return action

    def get_action_train(self, state: np.ndarray) -> float:
        """
        Select an action for training (with OU noise).

        Args
        ----
        state : ndarray
            Current state.

        Returns
        -------
        action : float
            Action with exploration noise, scaled to [action_low, action_high].
        """
        # Warmup: return random action
        if self.step_count < self.warmup_steps:
            return np.random.uniform(self.action_low, self.action_high)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            raw_action = self.actor(state_tensor).squeeze(0).item()

        # Add OU noise for exploration
        noise = self.ou_noise.sample()[0]
        raw_action_with_noise = raw_action + noise

        # Clip raw action to [-1, 1] before scaling
        raw_action_with_noise = np.clip(raw_action_with_noise, -1.0, 1.0)

        action = raw_action_with_noise * self.action_scale + self.action_offset
        action = np.clip(action, self.action_low, self.action_high)

        return action

    def push(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition in the replay buffer.

        Args
        ----
        state : ndarray
            Current state.
        action : float
            Action taken.
        reward : float
            Reward received.
        next_state : ndarray
            Next state.
        done : bool
            Whether the episode is finished.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.step_count += 1

    def update(
        self,
        batch_size: int = 64,
        n_updates: int = 1,
    ) -> Dict[str, float]:
        """
        Perform training updates on the actor and critic networks.

        Args
        ----
        batch_size : int
            Mini-batch size for training.
        n_updates : int
            Number of update steps to perform.

        Returns
        -------
        stats : dict
            Dictionary with 'actor_loss', 'critic_loss' keys.
        """
        if len(self.replay_buffer) < batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(n_updates):
            # Sample mini-batch
            batch = self.replay_buffer.sample(batch_size)
            states = torch.FloatTensor(np.array([t.state for t in batch]))
            actions = torch.FloatTensor(np.array([[t.action] for t in batch]))
            rewards = torch.FloatTensor(np.array([[t.reward] for t in batch]))
            next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
            dones = torch.FloatTensor(np.array([[1.0 - t.done] for t in batch]))

            # ---- Update Critic ----
            with torch.no_grad():
                next_actions = self.actor_target(next_states)
                target_q = self.critic_target(next_states, next_actions)
                y = rewards + self.gamma * dones * target_q

            current_q = self.critic(states, actions)
            critic_loss = F.mse_loss(current_q, y)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ---- Update Actor ----
            # Actor wants to maximize Q, so we minimize the negative Q
            actor_loss = -self.critic(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ---- Soft Update Target Networks ----
            self._soft_update(self.actor, self.actor_target, self.tau)
            self._soft_update(self.critic, self.critic_target, self.tau)

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        return {
            "actor_loss": total_actor_loss / n_updates,
            "critic_loss": total_critic_loss / n_updates,
        }

    def _soft_update(
        self,
        source_net: nn.Module,
        target_net: nn.Module,
        tau: float,
    ) -> None:
        """
        Soft update target network via Polyak averaging.

        θ⁻ ← τ·θ + (1−τ)·θ⁻
        """
        for source_param, target_param in zip(
            source_net.parameters(), target_net.parameters()
        ):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )

    def _hard_update(self, source_net: nn.Module, target_net: nn.Module) -> None:
        """Hard update: copy all parameters from source to target."""
        for source_param, target_param in zip(
            source_net.parameters(), target_net.parameters()
        ):
            target_param.data.copy_(source_param.data)

    def get_params(self) -> Dict[str, Any]:
        """Get network parameters for serialization."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set network parameters from serialization."""
        self.actor.load_state_dict(params["actor"])
        self.critic.load_state_dict(params["critic"])
        self.actor_target.load_state_dict(params["actor_target"])
        self.critic_target.load_state_dict(params["critic_target"])

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "replay_buffer_size": len(self.replay_buffer),
            "step_count": self.step_count,
            "warmup_steps": self.warmup_steps,
            "in_warmup": self.step_count < self.warmup_steps,
        }
