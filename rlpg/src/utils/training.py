"""
Training Utilities

This module provides functions for training and evaluating policies:
- Episode collection
- Policy evaluation
- Training loops for different algorithms

These utilities implement the core RL training loop and can be used
with any policy type.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import warnings


def collect_episode(
    env,
    policy,
    max_steps: Optional[int] = None,
    render: bool = False
) -> Tuple[List[np.ndarray], List[float], List[float], Dict[str, Any]]:
    """
    Collect one episode using the given policy.

    This function runs the policy in the environment for one complete
    episode and returns the trajectory data.

    Args:
        env: Environment object
        policy: Policy object with get_action method
        max_steps: Maximum steps (uses env default if None)
        render: Whether to print ASCII rendering

    Returns:
        Tuple of:
        - states: List of state arrays
        - actions: List of actions taken
        - rewards: List of rewards received
        - info: Final info dictionary
    """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    info = {}

    done = False
    step = 0
    max_steps = max_steps or env.max_steps

    while not done and step < max_steps:
        # Get action from policy
        action = policy.get_action(state)

        # Take step
        next_state, reward, done, info = env.step(action)

        # Store data
        actions.append(action)
        rewards.append(reward)
        states.append(next_state)

        # Optional rendering
        if render:
            print(env.render_ascii())
            print()

        state = next_state
        step += 1

    return states, actions, rewards, info


def evaluate_policy(
    env,
    policy,
    n_episodes: int = 10,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a policy over multiple episodes.

    Args:
        env: Environment object
        policy: Policy object
        n_episodes: Number of episodes to run
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Dictionary with:
        - 'mean_reward': Average total reward
        - 'std_reward': Standard deviation of rewards
        - 'mean_length': Average episode length
        - 'episode_rewards': List of all episode rewards
        - 'episode_lengths': List of all episode lengths
    """
    if seed is not None:
        np.random.seed(seed)

    episode_rewards = []
    episode_lengths = []

    iterator = range(n_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Evaluating")

    for _ in iterator:
        states, actions, rewards, info = collect_episode(env, policy)
        episode_rewards.append(sum(rewards))
        episode_lengths.append(len(rewards))

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def train_policy(
    env,
    policy,
    algorithm: str = 'evolutionary',
    n_iterations: int = 100,
    population_size: int = 20,
    elite_frac: float = 0.2,
    noise_scale: float = 0.1,
    learning_rate: float = 0.01,
    n_episodes_per_eval: int = 5,
    verbose: bool = True,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Train a policy using the specified algorithm.

    Supported algorithms:
    - 'evolutionary': Evolution strategies (works with any policy)
    - 'random_search': Simple random search
    - 'hill_climbing': Hill climbing with random perturbations

    Args:
        env: Environment object
        policy: Policy object (must have get_flat_params, set_flat_params)
        algorithm: Training algorithm to use
        n_iterations: Number of training iterations
        population_size: Population size for evolutionary methods
        elite_frac: Fraction of elite individuals to keep
        noise_scale: Scale of parameter perturbations
        learning_rate: Learning rate for parameter updates
        n_episodes_per_eval: Episodes per fitness evaluation
        verbose: Print training progress
        seed: Random seed

    Returns:
        Dictionary with:
        - 'best_params': Best parameters found
        - 'best_reward': Best reward achieved
        - 'reward_history': Rewards over training
    """
    if seed is not None:
        np.random.seed(seed)

    # Check if policy has required methods
    if not hasattr(policy, 'get_flat_params') or not hasattr(policy, 'set_flat_params'):
        raise ValueError("Policy must have get_flat_params and set_flat_params methods")

    if algorithm == 'evolutionary':
        return _train_evolutionary(
            env, policy, n_iterations, population_size,
            elite_frac, noise_scale, n_episodes_per_eval, verbose
        )
    elif algorithm == 'random_search':
        return _train_random_search(
            env, policy, n_iterations, noise_scale,
            n_episodes_per_eval, verbose
        )
    elif algorithm == 'hill_climbing':
        return _train_hill_climbing(
            env, policy, n_iterations, noise_scale,
            n_episodes_per_eval, verbose
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _train_evolutionary(
    env,
    policy,
    n_iterations: int,
    population_size: int,
    elite_frac: float,
    noise_scale: float,
    n_episodes: int,
    verbose: bool
) -> Dict[str, Any]:
    """
    Train using evolution strategies.

    This is a simple but effective algorithm that:
    1. Creates a population of perturbed parameters
    2. Evaluates fitness of each
    3. Updates parameters toward the best performers
    """
    n_elite = int(population_size * elite_frac)
    params = policy.get_flat_params()
    n_params = len(params)

    best_reward = float('-inf')
    best_params = params.copy()
    reward_history = []

    iterator = range(n_iterations)
    if verbose:
        iterator = tqdm(iterator, desc="Training")

    for iteration in iterator:
        # Create population
        noise = np.random.randn(population_size, n_params) * noise_scale
        population = params + noise

        # Evaluate population
        rewards = []
        for i in range(population_size):
            policy.set_flat_params(population[i])
            result = evaluate_policy(env, policy, n_episodes)
            rewards.append(result['mean_reward'])

        rewards = np.array(rewards)

        # Select elites
        elite_idx = np.argsort(rewards)[-n_elite:]
        elite_rewards = rewards[elite_idx]
        elite_noise = noise[elite_idx]

        # Update parameters (weighted by reward)
        weights = elite_rewards - elite_rewards.mean()
        if weights.std() > 0:
            weights /= weights.std()

        params = params + noise_scale * np.dot(weights, elite_noise) / n_elite

        # Track best
        if elite_rewards.max() > best_reward:
            best_reward = elite_rewards.max()
            best_idx = elite_idx[np.argmax(elite_rewards)]
            best_params = population[best_idx].copy()

        reward_history.append(elite_rewards.mean())

        if verbose:
            iterator.set_postfix({
                'mean': f'{elite_rewards.mean():.1f}',
                'best': f'{best_reward:.1f}'
            })

    # Set best parameters
    policy.set_flat_params(best_params)

    return {
        'best_params': best_params,
        'best_reward': best_reward,
        'reward_history': reward_history
    }


def _train_random_search(
    env,
    policy,
    n_iterations: int,
    noise_scale: float,
    n_episodes: int,
    verbose: bool
) -> Dict[str, Any]:
    """
    Train using simple random search.

    Randomly samples parameters and keeps the best.
    Simple but can be surprisingly effective!
    """
    params = policy.get_flat_params()
    n_params = len(params)

    best_reward = float('-inf')
    best_params = params.copy()
    reward_history = []

    iterator = range(n_iterations)
    if verbose:
        iterator = tqdm(iterator, desc="Random Search")

    for _ in iterator:
        # Random parameters
        candidate = np.random.randn(n_params) * noise_scale

        # Evaluate
        policy.set_flat_params(candidate)
        result = evaluate_policy(env, policy, n_episodes)
        reward = result['mean_reward']

        # Track best
        if reward > best_reward:
            best_reward = reward
            best_params = candidate.copy()

        reward_history.append(reward)

        if verbose:
            iterator.set_postfix({'best': f'{best_reward:.1f}'})

    # Set best parameters
    policy.set_flat_params(best_params)

    return {
        'best_params': best_params,
        'best_reward': best_reward,
        'reward_history': reward_history
    }


def _train_hill_climbing(
    env,
    policy,
    n_iterations: int,
    noise_scale: float,
    n_episodes: int,
    verbose: bool
) -> Dict[str, Any]:
    """
    Train using hill climbing.

    Perturbs current parameters and keeps if better.
    Gets stuck in local optima but simple to understand.
    """
    params = policy.get_flat_params()
    n_params = len(params)

    # Evaluate initial parameters
    policy.set_flat_params(params)
    result = evaluate_policy(env, policy, n_episodes)
    best_reward = result['mean_reward']
    best_params = params.copy()
    reward_history = [best_reward]

    iterator = range(n_iterations)
    if verbose:
        iterator = tqdm(iterator, desc="Hill Climbing")

    for _ in iterator:
        # Perturb parameters
        noise = np.random.randn(n_params) * noise_scale
        candidate = best_params + noise

        # Evaluate
        policy.set_flat_params(candidate)
        result = evaluate_policy(env, policy, n_episodes)
        reward = result['mean_reward']

        # Keep if better
        if reward > best_reward:
            best_reward = reward
            best_params = candidate.copy()

        reward_history.append(best_reward)

        if verbose:
            iterator.set_postfix({'best': f'{best_reward:.1f}'})

    # Set best parameters
    policy.set_flat_params(best_params)

    return {
        'best_params': best_params,
        'best_reward': best_reward,
        'reward_history': reward_history
    }


def compute_returns(
    rewards: List[float],
    gamma: float = 0.99
) -> np.ndarray:
    """
    Compute discounted returns for each timestep.

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Args:
        rewards: List of rewards
        gamma: Discount factor

    Returns:
        Array of discounted returns
    """
    returns = np.zeros(len(rewards))
    running_return = 0

    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


def normalize_returns(returns: np.ndarray) -> np.ndarray:
    """
    Normalize returns to have zero mean and unit variance.

    This often improves training stability.

    Args:
        returns: Array of returns

    Returns:
        Normalized returns
    """
    mean = np.mean(returns)
    std = np.std(returns)

    if std < 1e-8:
        return returns - mean

    return (returns - mean) / std
