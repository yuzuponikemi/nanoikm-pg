"""
Deep Q-Network (DQN) Policy

Implements DQN with two key stabilization techniques introduced in
Mnih et al., 2015 (Nature):

  1. Experience Replay Buffer  – breaks correlation between consecutive samples
  2. Target Network            – stabilizes the TD target during training

Algorithm overview
------------------
Online network  Q(s, a; θ)        — updated every step
Target network  Q(s, a; θ⁻)       — periodically synced from θ

TD target:
    y = r + γ · max_{a'} Q(s', a'; θ⁻)   (if not done)
    y = r                                  (if done)

Loss:
    L(θ) = E[(y − Q(s, a; θ))²]

Exploration:
    ε-greedy with linear or exponential decay.

Differences from tabular Q-learning (q_policy.py):
  - Neural network instead of Q-table → handles continuous states directly
  - Replay buffer → decorrelates training samples
  - Target network → reduces oscillation of TD target
  - Discrete action set (still) – continuous extensions need SAC/TD3
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Tuple

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
# Replay Buffer
# ---------------------------------------------------------------------------


class Transition(NamedTuple):
    """Single experience tuple stored in the replay buffer."""

    state: np.ndarray
    action_idx: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Fixed-size circular replay buffer.

    Stores past transitions (s, a, r, s', done) and samples random
    mini-batches for DQN updates.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store (FIFO eviction).

    Example
    -------
    >>> buf = ReplayBuffer(capacity=10_000)
    >>> buf.push(state, 2, 1.0, next_state, False)
    >>> batch = buf.sample(batch_size=64)
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self._buffer: Deque[Transition] = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add one transition to the buffer."""
        self._buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action_idx=int(action_idx),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a random mini-batch of transitions."""
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}/{self.capacity})"


# ---------------------------------------------------------------------------
# Q-Network (MLP)
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    """
    Feedforward Q-network: Q(s, a) for all actions simultaneously.

    Architecture
    ------------
    state_dim → hidden[0] → ReLU → … → hidden[-1] → ReLU → n_actions

    Each output neuron corresponds to Q(s, aᵢ).

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state vector.
    n_actions : int
        Number of discrete actions.
    hidden_sizes : list of int
        Widths of the hidden layers.
    """

    def __init__(
        self,
        state_dim: int = 4,
        n_actions: int = 11,
        hidden_sizes: List[int] = [64, 64],
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        return self.net(x)


# ---------------------------------------------------------------------------
# DQN Policy
# ---------------------------------------------------------------------------


class DQNPolicy(Policy):
    """
    Deep Q-Network (DQN) policy.

    Extends the tabular QPolicy concept to neural networks by using:

    * **QNetwork** – approximates Q(s, a) for all discrete actions.
    * **Target network** – a periodic copy of the online network used as a
      stable TD target.
    * **ReplayBuffer** – stores past transitions to break temporal correlation.

    Parameters
    ----------
    n_actions : int
        Number of discrete actions.
    action_values : np.ndarray or None
        Continuous force values for each discrete action index.
        Default: ``np.linspace(-10, 10, n_actions)``.
    state_dim : int
        State vector dimensionality (4 for InvertedPendulum).
    hidden_sizes : list of int
        Hidden layer widths of the Q-network.
    lr : float
        Adam learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Initial ε for ε-greedy exploration.
    epsilon_min : float
        Minimum ε.
    epsilon_decay : float
        Multiplicative per-step ε decay.
    batch_size : int
        Mini-batch size for each gradient update.
    buffer_capacity : int
        Maximum number of transitions stored.
    target_update_freq : int
        Hard-update the target network every N gradient steps.

    Example
    -------
    >>> policy = DQNPolicy(n_actions=11)
    >>> action = policy.get_action(env.reset())          # inference
    >>> action = policy.get_action_train(env.reset())    # ε-greedy
    >>> td_err = policy.update(batch_size=64)            # gradient step
    """

    def __init__(
        self,
        n_actions: int = 11,
        action_values: Optional[np.ndarray] = None,
        state_dim: int = 4,
        hidden_sizes: List[int] = [64, 64],
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 10_000,
        target_update_freq: int = 200,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DQNPolicy. "
                "Install it with: pip install torch"
            )

        self.n_actions = n_actions
        self.action_values = (
            action_values
            if action_values is not None
            else np.linspace(-10.0, 10.0, n_actions)
        )
        self.state_dim = state_dim
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # --- Networks ---
        self.online_net = QNetwork(state_dim, n_actions, hidden_sizes)
        self.target_net = QNetwork(state_dim, n_actions, hidden_sizes)
        self._sync_target()  # initialise target = online

        # --- Optimizer & loss ---
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # --- Replay buffer ---
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        # --- Counters ---
        self._step_count = 0
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------

    def _sync_target(self) -> None:
        """Hard-copy online → target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def get_action(self, state: np.ndarray) -> float:
        """
        Greedy action (no exploration) – use for evaluation.

        Returns
        -------
        float
            Continuous force value corresponding to the best discrete action.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        action_idx = int(q_values.argmax(dim=1).item())
        return float(self.action_values[action_idx])

    def get_action_train(self, state: np.ndarray) -> Tuple[float, int]:
        """
        ε-greedy action – use during training.

        Returns
        -------
        action : float
            Continuous force value.
        action_idx : int
            Index of the chosen action (needed for ``push``).
        """
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.n_actions)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.online_net(state_t)
            action_idx = int(q_values.argmax(dim=1).item())
        return float(self.action_values[action_idx]), action_idx

    # ------------------------------------------------------------------
    # Replay buffer helpers
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action_idx, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def update(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient descent step.

        Returns
        -------
        float or None
            TD loss value, or ``None`` if the buffer has fewer transitions
            than ``batch_size``.
        """
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)

        # Unpack batch
        states = torch.FloatTensor(np.stack([t.state for t in batch]))
        action_idxs = torch.LongTensor([t.action_idx for t in batch]).unsqueeze(1)
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor(np.stack([t.next_state for t in batch]))
        dones = torch.FloatTensor([float(t.done) for t in batch])

        # Current Q-values for taken actions: Q(s, a; θ)
        current_q = self.online_net(states).gather(1, action_idxs).squeeze(1)

        # TD targets using frozen target network: y = r + γ·max Q(s', ·; θ⁻)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._step_count += 1

        # Periodically sync target network
        if self._step_count % self.target_update_freq == 0:
            self._sync_target()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """Decay ε multiplicatively. Call once per training episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._episode_count += 1

    # ------------------------------------------------------------------
    # get_params / set_params (Policy interface)
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        return {
            "online_state": self.online_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self._step_count,
            "episode_count": self._episode_count,
            "n_actions": self.n_actions,
            "action_values": self.action_values.copy(),
            "hidden_sizes": self.hidden_sizes,
            "gamma": self.gamma,
            "lr": self.lr,
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        if "online_state" in params:
            self.online_net.load_state_dict(params["online_state"])
        if "target_state" in params:
            self.target_net.load_state_dict(params["target_state"])
        if "epsilon" in params:
            self.epsilon = params["epsilon"]
        if "step_count" in params:
            self._step_count = params["step_count"]
        if "episode_count" in params:
            self._episode_count = params["episode_count"]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return current training statistics."""
        return {
            "epsilon": self.epsilon,
            "step_count": self._step_count,
            "episode_count": self._episode_count,
            "buffer_size": len(self.buffer),
        }

    def __repr__(self) -> str:
        return (
            f"DQNPolicy(n_actions={self.n_actions}, "
            f"hidden={self.hidden_sizes}, "
            f"ε={self.epsilon:.3f}, "
            f"steps={self._step_count})"
        )
