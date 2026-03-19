"""
Tests for training utility functions

Covers:
- collect_episode() returns correct structure
- evaluate_policy() returns expected keys and statistics
- train_policy() returns best_params, best_reward, reward_history
- compute_returns() and normalize_returns()
"""

import numpy as np
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments.pendulum import InvertedPendulumEnv
from src.policies.random_policy import RandomPolicy
from src.policies.linear_policy import LinearPolicy
from src.utils.training import (
    collect_episode,
    evaluate_policy,
    train_policy,
    compute_returns,
    normalize_returns,
)


# ---------------------------------------------------------------------------
# collect_episode
# ---------------------------------------------------------------------------

class TestCollectEpisode:
    def setup_method(self):
        self.env = InvertedPendulumEnv()
        self.policy = RandomPolicy()

    def test_returns_four_elements(self):
        result = collect_episode(self.env, self.policy)
        assert len(result) == 4

    def test_states_is_list_of_arrays(self):
        states, _, _, _ = collect_episode(self.env, self.policy)
        assert isinstance(states, list)
        assert all(isinstance(s, np.ndarray) for s in states)

    def test_actions_is_list(self):
        _, actions, _, _ = collect_episode(self.env, self.policy)
        assert isinstance(actions, list)

    def test_rewards_is_list(self):
        _, _, rewards, _ = collect_episode(self.env, self.policy)
        assert isinstance(rewards, list)

    def test_info_is_dict(self):
        _, _, _, info = collect_episode(self.env, self.policy)
        assert isinstance(info, dict)

    def test_states_one_more_than_actions(self):
        """Initial state is included, so len(states) = len(actions) + 1."""
        states, actions, rewards, _ = collect_episode(self.env, self.policy)
        assert len(states) == len(actions) + 1

    def test_actions_same_length_as_rewards(self):
        _, actions, rewards, _ = collect_episode(self.env, self.policy)
        assert len(actions) == len(rewards)

    def test_state_shape(self):
        states, _, _, _ = collect_episode(self.env, self.policy)
        for s in states:
            assert s.shape == (4,)

    def test_max_steps_respected(self):
        """With max_steps=3 the episode should not exceed 3 actions."""
        _, actions, _, _ = collect_episode(self.env, self.policy, max_steps=3)
        assert len(actions) <= 3

    def test_rewards_are_non_negative(self):
        """Reward is 1.0 per alive step and 0.0 at terminal step."""
        _, _, rewards, _ = collect_episode(self.env, self.policy)
        assert all(r >= 0.0 for r in rewards)


# ---------------------------------------------------------------------------
# evaluate_policy
# ---------------------------------------------------------------------------

class TestEvaluatePolicy:
    def setup_method(self):
        self.env = InvertedPendulumEnv()
        self.policy = RandomPolicy(seed=0)

    def test_returns_dict(self):
        result = evaluate_policy(self.env, self.policy, n_episodes=3)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = evaluate_policy(self.env, self.policy, n_episodes=3)
        for key in ("mean_reward", "std_reward", "mean_length",
                    "episode_rewards", "episode_lengths"):
            assert key in result

    def test_episode_rewards_length(self):
        n = 4
        result = evaluate_policy(self.env, self.policy, n_episodes=n)
        assert len(result["episode_rewards"]) == n

    def test_episode_lengths_positive(self):
        result = evaluate_policy(self.env, self.policy, n_episodes=5)
        assert all(l > 0 for l in result["episode_lengths"])

    def test_mean_reward_is_float(self):
        result = evaluate_policy(self.env, self.policy, n_episodes=3)
        assert isinstance(result["mean_reward"], (float, np.floating))

    def test_mean_reward_matches_episodes(self):
        result = evaluate_policy(self.env, self.policy, n_episodes=5)
        expected_mean = np.mean(result["episode_rewards"])
        assert result["mean_reward"] == pytest.approx(expected_mean)

    def test_reproducible_with_seed(self):
        r1 = evaluate_policy(self.env, self.policy, n_episodes=5, seed=42)
        r2 = evaluate_policy(self.env, self.policy, n_episodes=5, seed=42)
        assert r1["mean_reward"] == pytest.approx(r2["mean_reward"])


# ---------------------------------------------------------------------------
# train_policy
# ---------------------------------------------------------------------------

class TestTrainPolicy:
    def _make_env_and_policy(self):
        env = InvertedPendulumEnv(max_steps=50)
        policy = LinearPolicy()
        return env, policy

    def test_returns_dict(self):
        env, policy = self._make_env_and_policy()
        result = train_policy(
            env, policy,
            algorithm="random_search",
            n_iterations=3,
            n_episodes_per_eval=2,
            verbose=False,
        )
        assert isinstance(result, dict)

    def test_required_keys(self):
        env, policy = self._make_env_and_policy()
        result = train_policy(
            env, policy,
            algorithm="random_search",
            n_iterations=3,
            n_episodes_per_eval=2,
            verbose=False,
        )
        for key in ("best_params", "best_reward", "reward_history"):
            assert key in result

    def test_best_reward_is_float(self):
        env, policy = self._make_env_and_policy()
        result = train_policy(
            env, policy,
            algorithm="hill_climbing",
            n_iterations=3,
            n_episodes_per_eval=2,
            verbose=False,
        )
        assert isinstance(result["best_reward"], (float, np.floating))

    def test_reward_history_length(self):
        n = 5
        env, policy = self._make_env_and_policy()
        result = train_policy(
            env, policy,
            algorithm="random_search",
            n_iterations=n,
            n_episodes_per_eval=2,
            verbose=False,
        )
        assert len(result["reward_history"]) == n

    def test_best_params_shape(self):
        env, policy = self._make_env_and_policy()
        result = train_policy(
            env, policy,
            algorithm="random_search",
            n_iterations=3,
            n_episodes_per_eval=2,
            verbose=False,
        )
        assert len(result["best_params"]) == policy.get_num_params()

    def test_evolutionary_algorithm(self):
        env, policy = self._make_env_and_policy()
        result = train_policy(
            env, policy,
            algorithm="evolutionary",
            n_iterations=2,
            population_size=4,
            n_episodes_per_eval=2,
            verbose=False,
        )
        assert "best_reward" in result

    def test_unknown_algorithm_raises(self):
        env, policy = self._make_env_and_policy()
        with pytest.raises(ValueError):
            train_policy(env, policy, algorithm="nonexistent", verbose=False)

    def test_policy_without_flat_params_raises(self):
        env = InvertedPendulumEnv()
        bad_policy = RandomPolicy()
        with pytest.raises(ValueError):
            train_policy(env, bad_policy, verbose=False)


# ---------------------------------------------------------------------------
# compute_returns
# ---------------------------------------------------------------------------

class TestComputeReturns:
    def test_single_reward(self):
        returns = compute_returns([5.0], gamma=0.99)
        assert len(returns) == 1
        assert returns[0] == pytest.approx(5.0)

    def test_two_rewards_no_discount(self):
        # gamma=1.0: returns = [r0 + r1, r1]
        returns = compute_returns([1.0, 1.0], gamma=1.0)
        assert returns[0] == pytest.approx(2.0)
        assert returns[1] == pytest.approx(1.0)

    def test_discount_applied(self):
        gamma = 0.9
        rewards = [1.0, 1.0, 1.0]
        returns = compute_returns(rewards, gamma=gamma)
        expected_0 = 1.0 + 0.9 * 1.0 + 0.81 * 1.0
        assert returns[0] == pytest.approx(expected_0)

    def test_later_returns_smaller_with_discount(self):
        rewards = [1.0] * 5
        returns = compute_returns(rewards, gamma=0.9)
        for i in range(len(returns) - 1):
            assert returns[i] > returns[i + 1]

    def test_output_length_matches_input(self):
        rewards = [1.0, 2.0, 3.0]
        returns = compute_returns(rewards)
        assert len(returns) == len(rewards)

    def test_returns_numpy_array(self):
        returns = compute_returns([1.0, 2.0])
        assert isinstance(returns, np.ndarray)


# ---------------------------------------------------------------------------
# normalize_returns
# ---------------------------------------------------------------------------

class TestNormalizeReturns:
    def test_zero_mean(self):
        returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_returns(returns)
        assert np.mean(normalized) == pytest.approx(0.0, abs=1e-7)

    def test_unit_variance(self):
        returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_returns(returns)
        assert np.std(normalized) == pytest.approx(1.0, rel=1e-5)

    def test_constant_returns(self):
        """Constant returns should not cause division by zero."""
        returns = np.array([3.0, 3.0, 3.0])
        normalized = normalize_returns(returns)
        assert np.all(np.isfinite(normalized))

    def test_output_shape_preserved(self):
        returns = np.array([1.0, 2.0, 3.0])
        normalized = normalize_returns(returns)
        assert normalized.shape == returns.shape
