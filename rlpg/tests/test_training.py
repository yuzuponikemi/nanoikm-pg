"""Tests for training utilities."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environments.pendulum import InvertedPendulumEnv
from policies.random_policy import RandomPolicy
from utils.training import evaluate_policy, compute_returns, normalize_returns


class TestEvaluatePolicy:
    """Test evaluate_policy() return value."""

    def test_returns_dict(self):
        env = InvertedPendulumEnv()
        policy = RandomPolicy()
        result = evaluate_policy(env, policy, n_episodes=2, seed=42)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        env = InvertedPendulumEnv()
        policy = RandomPolicy()
        result = evaluate_policy(env, policy, n_episodes=2, seed=42)
        expected_keys = {
            "mean_reward",
            "std_reward",
            "mean_length",
            "episode_rewards",
            "episode_lengths",
        }
        assert expected_keys == set(result.keys())

    def test_episode_rewards_length(self):
        env = InvertedPendulumEnv()
        policy = RandomPolicy()
        result = evaluate_policy(env, policy, n_episodes=5, seed=42)
        assert len(result["episode_rewards"]) == 5

    def test_episode_lengths_length(self):
        env = InvertedPendulumEnv()
        policy = RandomPolicy()
        result = evaluate_policy(env, policy, n_episodes=5, seed=42)
        assert len(result["episode_lengths"]) == 5

    def test_mean_reward_is_numeric(self):
        env = InvertedPendulumEnv()
        policy = RandomPolicy()
        result = evaluate_policy(env, policy, n_episodes=2, seed=42)
        assert isinstance(result["mean_reward"], (float, np.floating))

    def test_std_reward_non_negative(self):
        env = InvertedPendulumEnv()
        policy = RandomPolicy()
        result = evaluate_policy(env, policy, n_episodes=3, seed=42)
        assert result["std_reward"] >= 0

    def test_rewards_non_negative(self):
        env = InvertedPendulumEnv()
        policy = RandomPolicy()
        result = evaluate_policy(env, policy, n_episodes=3, seed=42)
        for r in result["episode_rewards"]:
            assert r >= 0


class TestComputeReturns:
    """Test compute_returns() behavior."""

    def test_single_reward(self):
        returns = compute_returns([1.0], gamma=0.99)
        assert len(returns) == 1
        assert abs(returns[0] - 1.0) < 1e-10

    def test_no_discount(self):
        rewards = [1.0, 1.0, 1.0]
        returns = compute_returns(rewards, gamma=1.0)
        np.testing.assert_array_almost_equal(returns, [3.0, 2.0, 1.0])

    def test_with_discount(self):
        rewards = [1.0, 1.0, 1.0]
        gamma = 0.5
        returns = compute_returns(rewards, gamma=gamma)
        expected = [
            1.0 + 0.5 * 1.0 + 0.25 * 1.0,  # 1.75
            1.0 + 0.5 * 1.0,                  # 1.5
            1.0,                               # 1.0
        ]
        np.testing.assert_array_almost_equal(returns, expected)

    def test_returns_ndarray(self):
        returns = compute_returns([1.0, 2.0, 3.0])
        assert isinstance(returns, np.ndarray)

    def test_returns_length_matches_rewards(self):
        rewards = [1.0] * 10
        returns = compute_returns(rewards)
        assert len(returns) == 10

    def test_zero_discount(self):
        rewards = [1.0, 2.0, 3.0]
        returns = compute_returns(rewards, gamma=0.0)
        np.testing.assert_array_almost_equal(returns, [1.0, 2.0, 3.0])

    def test_empty_rewards(self):
        returns = compute_returns([])
        assert len(returns) == 0

    def test_monotone_decreasing_with_discount(self):
        rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
        returns = compute_returns(rewards, gamma=0.9)
        for i in range(len(returns) - 1):
            assert returns[i] >= returns[i + 1]


class TestNormalizeReturns:
    """Test normalize_returns() behavior."""

    def test_output_mean_near_zero(self):
        returns = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        normalized = normalize_returns(returns)
        assert abs(np.mean(normalized)) < 1e-10

    def test_output_std_near_one(self):
        returns = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        normalized = normalize_returns(returns)
        assert abs(np.std(normalized) - 1.0) < 1e-10

    def test_returns_ndarray(self):
        returns = np.array([1.0, 2.0, 3.0])
        normalized = normalize_returns(returns)
        assert isinstance(normalized, np.ndarray)

    def test_length_preserved(self):
        returns = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = normalize_returns(returns)
        assert len(normalized) == 4

    def test_constant_returns_zero_mean(self):
        returns = np.array([5.0, 5.0, 5.0])
        normalized = normalize_returns(returns)
        np.testing.assert_array_almost_equal(normalized, [0.0, 0.0, 0.0])

    def test_single_element(self):
        returns = np.array([42.0])
        normalized = normalize_returns(returns)
        assert abs(normalized[0]) < 1e-6
