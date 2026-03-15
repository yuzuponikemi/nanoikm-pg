"""
Inverted Pendulum Environment

This module implements a simulation of an inverted pendulum (cart-pole) system.
The goal is to balance a pole attached to a cart by applying horizontal forces
to the cart.

Physics Model:
=============
The inverted pendulum consists of:
- A cart of mass M that can move horizontally on a frictionless track
- A pole of mass m and length L attached to the cart by a frictionless pivot

State Variables:
- x: Cart position (meters)
- x_dot: Cart velocity (m/s)
- theta: Pole angle from vertical (radians, 0 = upright)
- theta_dot: Pole angular velocity (rad/s)

Control Input:
- F: Horizontal force applied to the cart (Newtons)

Equations of Motion (derived from Lagrangian mechanics):
theta_ddot = (g*sin(theta) + cos(theta)*(-F - m*L*theta_dot^2*sin(theta))/(M+m))
             / (L*(4/3 - m*cos^2(theta)/(M+m)))

x_ddot = (F + m*L*(theta_dot^2*sin(theta) - theta_ddot*cos(theta))) / (M+m)

Where g is gravitational acceleration (9.81 m/s^2).
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


class InvertedPendulumEnv:
    """
    Inverted Pendulum (Cart-Pole) Environment

    A classic control problem where the goal is to balance a pole on a cart
    by applying horizontal forces. This implementation provides full access
    to the physics parameters for educational purposes.

    Attributes:
        gravity (float): Gravitational acceleration (m/s^2)
        cart_mass (float): Mass of the cart (kg)
        pole_mass (float): Mass of the pole (kg)
        pole_length (float): Half-length of the pole (m)
        force_mag (float): Magnitude of force for discrete actions (N)
        tau (float): Time step for simulation (s)

    State Space:
        A 4-dimensional continuous space:
        [x, x_dot, theta, theta_dot]

    Action Space:
        Continuous: Force applied to cart in range [-force_mag, force_mag]
        Or discrete: 0 (push left) or 1 (push right)

    Example:
        >>> env = InvertedPendulumEnv()
        >>> state = env.reset()
        >>> for _ in range(100):
        ...     action = 0.0  # No force
        ...     state, reward, done, info = env.step(action)
        ...     if done:
        ...         break
    """

    def __init__(
        self,
        gravity: float = 9.81,
        cart_mass: float = 1.0,
        pole_mass: float = 0.1,
        pole_length: float = 0.5,
        force_mag: float = 10.0,
        tau: float = 0.02,
        theta_threshold: float = 0.2095,  # ~12 degrees
        x_threshold: float = 2.4,
        max_steps: int = 500,
    ):
        """
        Initialize the inverted pendulum environment.

        Args:
            gravity: Gravitational acceleration (m/s^2). Default: 9.81
            cart_mass: Mass of the cart (kg). Default: 1.0
            pole_mass: Mass of the pole (kg). Default: 0.1
            pole_length: Half the pole's length (m). Default: 0.5
            force_mag: Maximum force magnitude (N). Default: 10.0
            tau: Time step for Euler integration (s). Default: 0.02
            theta_threshold: Angle at which episode terminates (rad). Default: ~12 deg
            x_threshold: Position at which episode terminates (m). Default: 2.4
            max_steps: Maximum steps per episode. Default: 500
        """
        # Physical parameters
        self.gravity = gravity
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length  # Actually half the pole's length
        self.total_mass = cart_mass + pole_mass
        self.pole_mass_length = pole_mass * pole_length

        # Simulation parameters
        self.force_mag = force_mag
        self.tau = tau

        # Episode termination thresholds
        self.theta_threshold = theta_threshold
        self.x_threshold = x_threshold
        self.max_steps = max_steps

        # State bounds (for normalization)
        self.state_bounds = np.array([
            self.x_threshold * 2,      # x
            np.inf,                     # x_dot (unbounded)
            self.theta_threshold * 2,   # theta
            np.inf                      # theta_dot (unbounded)
        ])

        # Current state
        self.state: Optional[np.ndarray] = None
        self.steps: int = 0

        # For tracking history
        self.history: Dict[str, list] = {
            'states': [],
            'actions': [],
            'rewards': []
        }

    def reset(
        self,
        seed: Optional[int] = None,
        initial_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            initial_state: Specific initial state [x, x_dot, theta, theta_dot]
                          If None, state is sampled uniformly from [-0.05, 0.05]

        Returns:
            Initial state as numpy array
        """
        if seed is not None:
            np.random.seed(seed)

        if initial_state is not None:
            self.state = np.array(initial_state, dtype=np.float64)
        else:
            # Small random perturbation around upright equilibrium
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))

        self.steps = 0
        self.history = {'states': [self.state.copy()], 'actions': [], 'rewards': []}

        return self.state.copy()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take one step in the environment.

        This method:
        1. Applies the action (force) to the cart
        2. Integrates the equations of motion
        3. Computes the reward
        4. Checks for episode termination

        Args:
            action: Force to apply to cart. Can be:
                   - Continuous value in [-force_mag, force_mag]
                   - Discrete: 0 (left) or 1 (right)

        Returns:
            Tuple of:
            - state: New state [x, x_dot, theta, theta_dot]
            - reward: Reward for this step
            - done: Whether episode has terminated
            - info: Additional information dictionary
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Handle discrete actions
        if isinstance(action, (int, np.integer)):
            force = self.force_mag if action == 1 else -self.force_mag
        else:
            # Continuous action - clip to valid range
            force = np.clip(action, -self.force_mag, self.force_mag)

        # Unpack current state
        x, x_dot, theta, theta_dot = self.state

        # Compute accelerations using equations of motion
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Intermediate calculation
        temp = (force + self.pole_mass_length * theta_dot**2 * sin_theta) / self.total_mass

        # Angular acceleration
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.pole_length * (4.0/3.0 - self.pole_mass * cos_theta**2 / self.total_mass)
        )

        # Linear acceleration
        x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        # Update state
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self.steps += 1

        # Check termination conditions
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
            or self.steps >= self.max_steps
        )

        # Compute reward
        # +1 for each step the pole is balanced
        # You can modify this for different learning objectives
        reward = 1.0 if not done else 0.0

        # Additional info
        info = {
            'x': x,
            'x_dot': x_dot,
            'theta': theta,
            'theta_dot': theta_dot,
            'force': force,
            'steps': self.steps,
            'terminated_by_angle': abs(theta) > self.theta_threshold,
            'terminated_by_position': abs(x) > self.x_threshold,
            'terminated_by_time': self.steps >= self.max_steps
        }

        # Update history
        self.history['states'].append(self.state.copy())
        self.history['actions'].append(force)
        self.history['rewards'].append(reward)

        return self.state.copy(), reward, done, info

    def get_state_labels(self) -> list:
        """Get human-readable labels for state variables."""
        return ['Cart Position (m)', 'Cart Velocity (m/s)',
                'Pole Angle (rad)', 'Pole Angular Velocity (rad/s)']

    def get_normalized_state(self) -> np.ndarray:
        """
        Get the current state normalized to approximately [-1, 1].

        This is useful for neural network inputs.
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        normalized = self.state.copy()
        normalized[0] /= self.x_threshold  # x
        normalized[2] /= self.theta_threshold  # theta
        # Velocities are harder to normalize, use empirical bounds
        normalized[1] /= 3.0  # x_dot
        normalized[3] /= 3.0  # theta_dot

        return normalized

    def get_energy(self) -> Dict[str, float]:
        """
        Calculate the kinetic and potential energy of the system.

        This is useful for understanding the physics and for
        energy-based control approaches.

        Returns:
            Dictionary with 'kinetic', 'potential', and 'total' energy
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        x, x_dot, theta, theta_dot = self.state

        # Cart kinetic energy: (1/2) * M * v^2
        cart_ke = 0.5 * self.cart_mass * x_dot**2

        # Pole kinetic energy (translation + rotation)
        # Velocity of pole center of mass
        pole_x_dot = x_dot + self.pole_length * theta_dot * np.cos(theta)
        pole_y_dot = self.pole_length * theta_dot * np.sin(theta)
        pole_translation_ke = 0.5 * self.pole_mass * (pole_x_dot**2 + pole_y_dot**2)

        # Rotational kinetic energy: (1/2) * I * omega^2
        # For a uniform rod rotating about one end: I = (1/3) * m * L^2
        # But we're using half-length, so full length is 2*pole_length
        pole_inertia = (1/3) * self.pole_mass * (2 * self.pole_length)**2
        pole_rotation_ke = 0.5 * pole_inertia * theta_dot**2

        total_ke = cart_ke + pole_translation_ke + pole_rotation_ke

        # Potential energy (height of pole center of mass)
        # Reference: ground level (cart height)
        pole_height = self.pole_length * np.cos(theta)
        pe = self.pole_mass * self.gravity * pole_height

        return {
            'kinetic': total_ke,
            'potential': pe,
            'total': total_ke + pe
        }

    def get_history(self) -> Dict[str, np.ndarray]:
        """
        Get the episode history as numpy arrays.

        Returns:
            Dictionary with 'states', 'actions', 'rewards' as numpy arrays
        """
        return {
            'states': np.array(self.history['states']),
            'actions': np.array(self.history['actions']),
            'rewards': np.array(self.history['rewards'])
        }

    def render_ascii(self) -> str:
        """
        Create an ASCII art representation of the current state.

        Returns:
            String with ASCII visualization
        """
        if self.state is None:
            return "Environment not initialized"

        x, _, theta, _ = self.state

        # Create a simple ASCII representation
        width = 40
        height = 10

        # Normalize cart position to screen coordinates
        cart_pos = int((x / self.x_threshold + 1) * (width - 4) / 2) + 2
        cart_pos = max(2, min(width - 3, cart_pos))

        # Create display
        lines = []

        # Pole (simplified as a line showing direction)
        pole_char = '|' if abs(theta) < 0.1 else ('/' if theta > 0 else '\\')

        lines.append('=' * width)

        # Empty lines above cart
        for i in range(height - 4):
            line = [' '] * width
            if i == height - 5:
                line[cart_pos] = pole_char
            lines.append(''.join(line))

        # Cart
        cart_line = [' '] * width
        cart_line[cart_pos - 1:cart_pos + 2] = ['[', '#', ']']
        lines.append(''.join(cart_line))

        # Ground
        lines.append('=' * width)

        # State info
        lines.append(f"x={x:.3f}m, theta={np.degrees(theta):.1f}deg")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (f"InvertedPendulumEnv(cart_mass={self.cart_mass}, "
                f"pole_mass={self.pole_mass}, pole_length={self.pole_length})")
