"""
Neural Network Policy

A neural network policy uses a deep neural network to map states to actions.
This is more expressive than a linear policy and can learn complex behaviors.

Architecture:
    Input (4) -> Hidden1 (64) -> ReLU -> Hidden2 (64) -> ReLU -> Output (1) -> Tanh

The final Tanh activation scales the output to [-1, 1], which is then
scaled to the action range.

Neural network policies are:
- Highly expressive (can approximate any function)
- Require more data to train
- Harder to interpret
- The foundation of deep reinforcement learning

For the inverted pendulum, a neural network is overkill (a linear policy
works fine), but it's educational to see how neural policies work.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .base import Policy

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class NeuralNetworkPolicy(Policy):
    """
    A neural network policy using PyTorch.

    This implements a simple feedforward neural network that maps
    states to actions.

    Attributes:
        network: PyTorch neural network
        action_low: Minimum action value
        action_high: Maximum action value

    Example:
        >>> policy = NeuralNetworkPolicy(hidden_sizes=[64, 64])
        >>> state = np.array([0, 0, 0.1, 0])
        >>> action = policy.get_action(state)
    """

    def __init__(
        self,
        hidden_sizes: List[int] = [64, 64],
        action_low: float = -10.0,
        action_high: float = 10.0,
        activation: str = 'relu'
    ):
        """
        Initialize the neural network policy.

        Args:
            hidden_sizes: List of hidden layer sizes
            action_low: Minimum action value
            action_high: Maximum action value
            activation: Activation function ('relu' or 'tanh')
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for NeuralNetworkPolicy. "
                "Install it with: pip install torch"
            )

        self.hidden_sizes = hidden_sizes
        self.action_low = action_low
        self.action_high = action_high
        self.activation_name = activation

        # Build the network
        self.network = self._build_network(hidden_sizes, activation)

        # Store action scaling parameters
        self.action_scale = (action_high - action_low) / 2
        self.action_offset = (action_high + action_low) / 2

    def _build_network(self, hidden_sizes: List[int], activation: str) -> nn.Module:
        """
        Build the neural network.

        Args:
            hidden_sizes: List of hidden layer sizes
            activation: Activation function name

        Returns:
            PyTorch Sequential model
        """
        layers = []

        # Input layer
        input_size = 4  # State dimension
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Tanh())  # Output in [-1, 1]

        return nn.Sequential(*layers)

    def get_action(self, state: np.ndarray) -> float:
        """
        Compute action using the neural network.

        Args:
            state: Current state [x, x_dot, theta, theta_dot]

        Returns:
            Action value
        """
        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Forward pass (no gradient needed for inference)
        with torch.no_grad():
            output = self.network(state_tensor)

        # Scale from [-1, 1] to [action_low, action_high]
        action = output.item() * self.action_scale + self.action_offset

        return action

    def get_action_and_log_prob(
        self,
        state: np.ndarray,
        action_std: float = 0.5
    ) -> tuple:
        """
        Get action from a stochastic policy (for policy gradients).

        Adds Gaussian noise to the network output for exploration.

        Args:
            state: Current state
            action_std: Standard deviation of action noise

        Returns:
            Tuple of (action, log_probability)
        """
        # Get mean action
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mean_output = self.network(state_tensor)
        mean_action = mean_output * self.action_scale + self.action_offset

        # Sample from Gaussian
        std = torch.tensor([action_std])
        dist = torch.distributions.Normal(mean_action, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Clip action to valid range
        action_clipped = torch.clamp(
            action,
            self.action_low,
            self.action_high
        )

        return action_clipped.item(), log_prob

    def get_params(self) -> Dict[str, Any]:
        """Get policy parameters (network state dict)."""
        return {
            'network_state': self.network.state_dict(),
            'hidden_sizes': self.hidden_sizes,
            'action_low': self.action_low,
            'action_high': self.action_high,
            'activation': self.activation_name
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set policy parameters."""
        if 'network_state' in params:
            self.network.load_state_dict(params['network_state'])
        if 'action_low' in params:
            self.action_low = params['action_low']
        if 'action_high' in params:
            self.action_high = params['action_high']
            self.action_scale = (self.action_high - self.action_low) / 2
            self.action_offset = (self.action_high + self.action_low) / 2

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.network.parameters())

    def get_flat_params(self) -> np.ndarray:
        """
        Get all network parameters as a flat numpy array.

        Returns:
            Flattened parameter array
        """
        params = []
        for param in self.network.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_flat_params(self, flat_params: np.ndarray) -> None:
        """
        Set network parameters from a flat numpy array.

        Args:
            flat_params: Flattened parameter array
        """
        idx = 0
        for param in self.network.parameters():
            param_size = param.numel()
            param_shape = param.shape
            param.data = torch.FloatTensor(
                flat_params[idx:idx + param_size].reshape(param_shape)
            )
            idx += param_size

    def get_network_summary(self) -> str:
        """
        Get a string summary of the network architecture.

        Returns:
            Human-readable network description
        """
        lines = ["Neural Network Policy Architecture:"]
        lines.append("=" * 40)

        total_params = 0
        for name, module in self.network.named_children():
            if isinstance(module, nn.Linear):
                params = module.weight.numel() + module.bias.numel()
                total_params += params
                lines.append(
                    f"  {name}: Linear({module.in_features} -> {module.out_features}) "
                    f"[{params} params]"
                )
            else:
                lines.append(f"  {name}: {module.__class__.__name__}")

        lines.append("=" * 40)
        lines.append(f"Total parameters: {total_params}")

        return '\n'.join(lines)
