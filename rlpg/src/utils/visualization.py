"""
Visualization Utilities

This module provides functions for visualizing:
- Episode trajectories (state variables over time)
- Training progress (rewards over episodes)
- Animated pendulum rendering

Visualization is crucial for understanding what the agent is learning
and debugging issues with the environment or policy.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from typing import Dict, List, Optional, Any
import warnings


def plot_trajectory(
    history: Dict[str, np.ndarray],
    title: str = "Episode Trajectory",
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Plot the trajectory of an episode.

    Shows all state variables and actions over time.

    Args:
        history: Dictionary with 'states', 'actions', 'rewards' arrays
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    states = history['states']
    actions = history['actions']
    rewards = history['rewards']

    n_steps = len(states)
    time = np.arange(n_steps)

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14)

    # State labels
    state_labels = [
        'Cart Position (m)',
        'Cart Velocity (m/s)',
        'Pole Angle (rad)',
        'Angular Velocity (rad/s)'
    ]

    # Plot state variables
    for i in range(4):
        ax = axes[i // 2, i % 2]
        ax.plot(time, states[:, i], 'b-', linewidth=1)
        ax.set_xlabel('Time Step')
        ax.set_ylabel(state_labels[i])
        ax.grid(True, alpha=0.3)

        # Add zero line for reference
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Plot actions
    ax = axes[2, 0]
    if len(actions) > 0:
        ax.plot(time[:-1], actions, 'r-', linewidth=1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Force (N)')
    ax.set_title('Actions')
    ax.grid(True, alpha=0.3)

    # Plot cumulative reward
    ax = axes[2, 1]
    if len(rewards) > 0:
        cumulative_reward = np.cumsum(rewards)
        ax.plot(time[:-1], cumulative_reward, 'g-', linewidth=1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Rewards')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_training_progress(
    episode_rewards: List[float],
    episode_lengths: Optional[List[int]] = None,
    window_size: int = 10,
    figsize: tuple = (12, 4)
) -> plt.Figure:
    """
    Plot training progress over episodes.

    Args:
        episode_rewards: List of total rewards per episode
        episode_lengths: Optional list of episode lengths
        window_size: Window size for moving average
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    episodes = np.arange(len(episode_rewards))

    if episode_lengths is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
        ax2 = None

    # Plot rewards
    ax1.plot(episodes, episode_rewards, 'b-', alpha=0.3, label='Episode Reward')

    # Moving average
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(
            episode_rewards,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        ax1.plot(
            episodes[window_size - 1:],
            moving_avg,
            'r-',
            linewidth=2,
            label=f'{window_size}-Episode Average'
        )

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot episode lengths
    if ax2 is not None and episode_lengths is not None:
        ax2.plot(episodes, episode_lengths, 'g-', alpha=0.3, label='Episode Length')

        if len(episode_lengths) >= window_size:
            moving_avg = np.convolve(
                episode_lengths,
                np.ones(window_size) / window_size,
                mode='valid'
            )
            ax2.plot(
                episodes[window_size - 1:],
                moving_avg,
                'r-',
                linewidth=2,
                label=f'{window_size}-Episode Average'
            )

        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Lengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def animate_pendulum(
    history: Dict[str, np.ndarray],
    env_params: Optional[Dict[str, float]] = None,
    interval: int = 20,
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None
) -> animation.FuncAnimation:
    """
    Create an animation of the pendulum episode.

    Args:
        history: Dictionary with 'states' array
        env_params: Environment parameters (pole_length, x_threshold)
        interval: Milliseconds between frames
        figsize: Figure size
        save_path: If provided, save animation to this path

    Returns:
        Matplotlib animation object
    """
    states = history['states']
    n_frames = len(states)

    # Default parameters
    if env_params is None:
        env_params = {
            'pole_length': 0.5,
            'x_threshold': 2.4
        }

    pole_length = env_params.get('pole_length', 0.5)
    x_threshold = env_params.get('x_threshold', 2.4)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set up the plot
    ax.set_xlim(-x_threshold - 0.5, x_threshold + 0.5)
    ax.set_ylim(-0.5, pole_length * 2 + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Height (m)')

    # Draw track
    ax.axhline(y=0, color='brown', linewidth=3)

    # Create cart (rectangle)
    cart_width = 0.3
    cart_height = 0.15
    cart = FancyBboxPatch(
        (0 - cart_width / 2, -cart_height / 2),
        cart_width, cart_height,
        boxstyle="round,pad=0.02",
        facecolor='blue',
        edgecolor='black'
    )
    ax.add_patch(cart)

    # Create pole (line)
    pole, = ax.plot([], [], 'o-', color='brown', linewidth=4, markersize=8)

    # Create pivot point
    pivot = Circle((0, 0), 0.03, color='black')
    ax.add_patch(pivot)

    # Time display
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    state_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=8)

    def init():
        """Initialize animation."""
        cart.set_bounds(0 - cart_width / 2, -cart_height / 2, cart_width, cart_height)
        pole.set_data([], [])
        pivot.center = (0, 0)
        time_text.set_text('')
        state_text.set_text('')
        return cart, pole, pivot, time_text, state_text

    def animate(frame):
        """Update animation for each frame."""
        x = states[frame, 0]
        theta = states[frame, 2]

        # Update cart position
        cart.set_bounds(x - cart_width / 2, -cart_height / 2, cart_width, cart_height)

        # Update pivot
        pivot.center = (x, 0)

        # Update pole
        pole_x = [x, x + pole_length * 2 * np.sin(theta)]
        pole_y = [0, pole_length * 2 * np.cos(theta)]
        pole.set_data(pole_x, pole_y)

        # Update text
        time_text.set_text(f'Step: {frame}/{n_frames - 1}')
        state_text.set_text(
            f'x={x:.2f}m, theta={np.degrees(theta):.1f}deg'
        )

        return cart, pole, pivot, time_text, state_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=interval,
        blit=True
    )

    if save_path:
        try:
            anim.save(save_path, writer='pillow', fps=1000 // interval)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            warnings.warn(f"Could not save animation: {e}")

    return anim


def plot_phase_portrait(
    history: Dict[str, np.ndarray],
    figsize: tuple = (10, 4)
) -> plt.Figure:
    """
    Plot phase portraits of the pendulum dynamics.

    Shows theta vs theta_dot and x vs x_dot.

    Args:
        history: Dictionary with 'states' array
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    states = history['states']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Theta vs theta_dot
    ax1.plot(states[:, 2], states[:, 3], 'b-', alpha=0.7)
    ax1.plot(states[0, 2], states[0, 3], 'go', markersize=10, label='Start')
    ax1.plot(states[-1, 2], states[-1, 3], 'ro', markersize=10, label='End')
    ax1.set_xlabel('Pole Angle (rad)')
    ax1.set_ylabel('Angular Velocity (rad/s)')
    ax1.set_title('Pole Phase Portrait')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # X vs x_dot
    ax2.plot(states[:, 0], states[:, 1], 'b-', alpha=0.7)
    ax2.plot(states[0, 0], states[0, 1], 'go', markersize=10, label='Start')
    ax2.plot(states[-1, 0], states[-1, 1], 'ro', markersize=10, label='End')
    ax2.set_xlabel('Cart Position (m)')
    ax2.set_ylabel('Cart Velocity (m/s)')
    ax2.set_title('Cart Phase Portrait')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_policy_surface(
    policy,
    state_ranges: Optional[Dict[str, tuple]] = None,
    fixed_states: Optional[Dict[str, float]] = None,
    resolution: int = 50,
    figsize: tuple = (10, 4)
) -> plt.Figure:
    """
    Plot the policy as a surface over two state variables.

    Args:
        policy: Policy object with get_action method
        state_ranges: Dictionary of state variable ranges to plot
        fixed_states: Dictionary of fixed values for other states
        resolution: Grid resolution
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Default ranges
    if state_ranges is None:
        state_ranges = {
            'theta': (-0.3, 0.3),
            'theta_dot': (-2, 2)
        }

    if fixed_states is None:
        fixed_states = {'x': 0, 'x_dot': 0}

    # Get the two varying dimensions
    var_names = list(state_ranges.keys())
    state_idx = {'x': 0, 'x_dot': 1, 'theta': 2, 'theta_dot': 3}

    # Create meshgrid
    range1 = state_ranges[var_names[0]]
    range2 = state_ranges[var_names[1]]

    v1 = np.linspace(range1[0], range1[1], resolution)
    v2 = np.linspace(range2[0], range2[1], resolution)
    V1, V2 = np.meshgrid(v1, v2)

    # Compute actions
    actions = np.zeros_like(V1)

    for i in range(resolution):
        for j in range(resolution):
            state = np.zeros(4)

            # Set fixed states
            for name, value in fixed_states.items():
                state[state_idx[name]] = value

            # Set varying states
            state[state_idx[var_names[0]]] = V1[i, j]
            state[state_idx[var_names[1]]] = V2[i, j]

            actions[i, j] = policy.get_action(state)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Contour plot
    contour = ax1.contourf(V1, V2, actions, levels=20, cmap='RdBu')
    ax1.set_xlabel(var_names[0])
    ax1.set_ylabel(var_names[1])
    ax1.set_title('Policy Action Map')
    plt.colorbar(contour, ax=ax1, label='Action (Force)')

    # Surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(V1, V2, actions, cmap='RdBu', alpha=0.8)
    ax2.set_xlabel(var_names[0])
    ax2.set_ylabel(var_names[1])
    ax2.set_zlabel('Action')
    ax2.set_title('Policy Surface')

    plt.tight_layout()
    return fig
