"""
pytest configuration and shared fixtures for rlpg tests.
"""

import sys
import os
import pytest
import numpy as np

# Add the rlpg src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.pendulum import InvertedPendulumEnv
from src.policies.random_policy import RandomPolicy
from src.policies.linear_policy import LinearPolicy


@pytest.fixture
def env():
    """Standard inverted pendulum environment."""
    return InvertedPendulumEnv()


@pytest.fixture
def env_seeded():
    """Seeded environment for reproducible tests."""
    e = InvertedPendulumEnv()
    e.reset(seed=42)
    return e


@pytest.fixture
def random_policy():
    """Random policy instance."""
    return RandomPolicy(seed=42)


@pytest.fixture
def linear_policy():
    """Linear policy with zero weights (default)."""
    return LinearPolicy()


@pytest.fixture
def linear_policy_tuned():
    """Linear policy with manually tuned weights known to balance the pole."""
    # Positive weights on theta and theta_dot stabilize the pole
    return LinearPolicy(weights=np.array([0.0, 0.0, 10.0, 2.0]))
