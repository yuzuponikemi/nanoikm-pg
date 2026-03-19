"""
Tests for Policy classes

Covers:
- RandomPolicy: get_action() returns float in expected range
- LinearPolicy: get_action() returns correct shape and value
- NeuralNetworkPolicy: get_action() returns float (if torch available)
- QPolicy: get_action() returns float from action_values
- REINFORCEPolicy: get_action() returns float (if torch available)
- All policies: get_params() / set_params() roundtrip
"""

import numpy as np
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests._torch_available import TORCH_AVAILABLE

from src.policies.base import Policy
from src.policies.random_policy import RandomPolicy
from src.policies.linear_policy import LinearPolicy
from src.utils.discretizer import StateDiscretizer

# q_policy imports from src.utils.discretizer (no matplotlib/torch dependency)
from src.policies.q_policy import QTable, QPolicy

if TORCH_AVAILABLE:
    try:
        from src.policies.neural_policy import NeuralNetworkPolicy
        from src.policies.policy_gradient import REINFORCEPolicy
    except Exception:
        TORCH_AVAILABLE = False

# Canonical state used across all tests
SAMPLE_STATE = np.array([0.1, 0.2, 0.05, -0.1], dtype=np.float64)
ZERO_STATE = np.zeros(4, dtype=np.float64)


# ---------------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------------

class TestRandomPolicy:
    def test_get_action_returns_float(self):
        policy = RandomPolicy()
        action = policy.get_action(SAMPLE_STATE)
        assert isinstance(action, (float, np.floating))

    def test_get_action_within_range_continuous(self):
        policy = RandomPolicy(action_low=-10.0, action_high=10.0)
        for _ in range(50):
            action = policy.get_action(SAMPLE_STATE)
            assert -10.0 <= action <= 10.0

    def test_get_action_discrete_only_extremes(self):
        policy = RandomPolicy(action_low=-10.0, action_high=10.0, discrete=True)
        for _ in range(30):
            action = policy.get_action(SAMPLE_STATE)
            assert action in (-10.0, 10.0)

    def test_get_num_params_is_zero(self):
        policy = RandomPolicy()
        assert policy.get_num_params() == 0

    def test_get_params_returns_dict(self):
        policy = RandomPolicy()
        params = policy.get_params()
        assert isinstance(params, dict)
        assert "action_low" in params
        assert "action_high" in params

    def test_set_params_updates_range(self):
        policy = RandomPolicy(action_low=-5.0, action_high=5.0)
        policy.set_params({"action_low": -1.0, "action_high": 1.0})
        assert policy.action_low == -1.0
        assert policy.action_high == 1.0
        for _ in range(30):
            action = policy.get_action(SAMPLE_STATE)
            assert -1.0 <= action <= 1.0

    def test_reproducible_with_seed(self):
        np.random.seed(7)
        policy = RandomPolicy()
        actions1 = [policy.get_action(SAMPLE_STATE) for _ in range(5)]
        np.random.seed(7)
        actions2 = [policy.get_action(SAMPLE_STATE) for _ in range(5)]
        assert actions1 == actions2


# ---------------------------------------------------------------------------
# LinearPolicy
# ---------------------------------------------------------------------------

class TestLinearPolicy:
    def test_get_action_returns_float(self):
        policy = LinearPolicy()
        action = policy.get_action(SAMPLE_STATE)
        assert isinstance(action, float)

    def test_get_action_zero_weights_returns_zero(self):
        policy = LinearPolicy(weights=np.zeros(4), bias=0.0)
        action = policy.get_action(ZERO_STATE)
        assert action == pytest.approx(0.0)

    def test_get_action_linear_combination(self):
        weights = np.array([1.0, 0.0, 0.0, 0.0])
        policy = LinearPolicy(weights=weights, bias=0.0, action_low=-100.0, action_high=100.0)
        state = np.array([3.0, 0.0, 0.0, 0.0])
        action = policy.get_action(state)
        assert action == pytest.approx(3.0)

    def test_get_action_clipped_high(self):
        policy = LinearPolicy(weights=np.array([100.0, 0, 0, 0]), action_high=10.0)
        action = policy.get_action(np.array([1.0, 0, 0, 0]))
        assert action == pytest.approx(10.0)

    def test_get_action_clipped_low(self):
        policy = LinearPolicy(weights=np.array([-100.0, 0, 0, 0]), action_low=-10.0)
        action = policy.get_action(np.array([1.0, 0, 0, 0]))
        assert action == pytest.approx(-10.0)

    def test_get_num_params(self):
        policy = LinearPolicy()
        # 4 weights + 1 bias = 5
        assert policy.get_num_params() == 5

    def test_get_params_roundtrip(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        policy = LinearPolicy(weights=weights, bias=0.5)
        params = policy.get_params()
        policy2 = LinearPolicy()
        policy2.set_params(params)
        np.testing.assert_array_almost_equal(policy2.weights, weights)
        assert policy2.bias == pytest.approx(0.5)

    def test_get_flat_params_length(self):
        policy = LinearPolicy()
        flat = policy.get_flat_params()
        assert len(flat) == 5

    def test_set_flat_params_roundtrip(self):
        policy = LinearPolicy()
        flat = np.array([1.0, 2.0, 3.0, 4.0, 0.5])
        policy.set_flat_params(flat)
        assert policy.weights[0] == pytest.approx(1.0)
        assert policy.bias == pytest.approx(0.5)

    def test_perturb_returns_new_policy(self):
        policy = LinearPolicy(weights=np.zeros(4), bias=0.0)
        perturbed = policy.perturb(noise_scale=0.1)
        assert perturbed is not policy
        assert isinstance(perturbed, LinearPolicy)

    def test_perturb_changes_params(self):
        policy = LinearPolicy(weights=np.zeros(4), bias=0.0)
        perturbed = policy.perturb(noise_scale=1.0)
        # With scale=1.0 the params will almost certainly differ
        changed = not np.allclose(policy.weights, perturbed.weights) or (
            policy.bias != perturbed.bias
        )
        assert changed


# ---------------------------------------------------------------------------
# QTable
# ---------------------------------------------------------------------------

class TestQTable:
    def test_init_shape(self):
        qt = QTable(n_states=100, n_actions=5)
        assert qt.table.shape == (100, 5)

    def test_init_value(self):
        qt = QTable(n_states=10, n_actions=3, init_value=1.5)
        assert np.all(qt.table == 1.5)

    def test_get_and_update(self):
        qt = QTable(n_states=10, n_actions=3)
        qt.update(2, 1, 99.9)
        assert qt.get(2, 1) == pytest.approx(99.9)

    def test_greedy_action(self):
        qt = QTable(n_states=5, n_actions=3)
        qt.update(0, 2, 10.0)
        assert qt.greedy_action(0) == 2

    def test_max_value(self):
        qt = QTable(n_states=5, n_actions=3)
        qt.update(0, 1, 7.0)
        assert qt.max_value(0) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# QPolicy
# ---------------------------------------------------------------------------

class TestQPolicy:
    def setup_method(self):
        self.disc = StateDiscretizer()
        self.policy = QPolicy(self.disc, n_actions=3)

    def test_get_action_returns_float(self):
        action = self.policy.get_action(SAMPLE_STATE)
        assert isinstance(action, float)

    def test_get_action_within_action_values(self):
        for _ in range(20):
            action = self.policy.get_action(SAMPLE_STATE)
            assert action in self.policy.action_values

    def test_get_action_shape_is_scalar(self):
        action = self.policy.get_action(SAMPLE_STATE)
        # should be a plain float, not array
        assert np.isscalar(action)

    def test_update_q_returns_td_error(self):
        td = self.policy.update_q(
            state=ZERO_STATE, action=0.0, reward=1.0,
            next_state=SAMPLE_STATE, done=False
        )
        assert isinstance(td, float)

    def test_decay_epsilon(self):
        initial_eps = self.policy.epsilon
        self.policy.decay_epsilon()
        assert self.policy.epsilon < initial_eps

    def test_get_params_has_q_table(self):
        params = self.policy.get_params()
        assert "q_table" in params

    def test_set_params_restores_q_table(self):
        self.policy.update_q(ZERO_STATE, 0.0, 5.0, SAMPLE_STATE, False)
        params = self.policy.get_params()
        policy2 = QPolicy(StateDiscretizer(), n_actions=3)
        policy2.set_params(params)
        np.testing.assert_array_almost_equal(
            policy2.q_table.table, self.policy.q_table.table
        )

    def test_full_greedy_action_uses_q_table(self):
        """Force epsilon=0 (fully greedy) and verify argmax is followed."""
        self.policy.epsilon = 0.0
        state_idx = self.disc.encode(ZERO_STATE)
        # Set Q-value for action index 2 to be very high
        self.policy.q_table.update(state_idx, 2, 999.0)
        action = self.policy.get_action(ZERO_STATE)
        expected = self.policy.action_values[2]
        assert action == pytest.approx(expected)


# ---------------------------------------------------------------------------
# NeuralNetworkPolicy (only when PyTorch is available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestNeuralNetworkPolicy:
    def test_get_action_returns_float(self):
        policy = NeuralNetworkPolicy(hidden_sizes=[16])
        action = policy.get_action(SAMPLE_STATE)
        assert isinstance(action, float)

    def test_get_action_within_range(self):
        policy = NeuralNetworkPolicy(action_low=-10.0, action_high=10.0, hidden_sizes=[16])
        for _ in range(10):
            action = policy.get_action(SAMPLE_STATE)
            assert -10.0 <= action <= 10.0

    def test_get_num_params_positive(self):
        policy = NeuralNetworkPolicy(hidden_sizes=[16])
        assert policy.get_num_params() > 0

    def test_get_flat_params_length_matches_num_params(self):
        policy = NeuralNetworkPolicy(hidden_sizes=[16])
        flat = policy.get_flat_params()
        assert len(flat) == policy.get_num_params()

    def test_set_flat_params_roundtrip(self):
        policy = NeuralNetworkPolicy(hidden_sizes=[16])
        flat_original = policy.get_flat_params()
        # modify
        flat_new = flat_original + 1.0
        policy.set_flat_params(flat_new)
        np.testing.assert_array_almost_equal(policy.get_flat_params(), flat_new)

    def test_get_params_returns_dict(self):
        policy = NeuralNetworkPolicy(hidden_sizes=[16])
        params = policy.get_params()
        assert "network_state" in params
        assert "hidden_sizes" in params

    def test_get_action_and_log_prob(self):
        policy = NeuralNetworkPolicy(hidden_sizes=[16])
        action, log_prob = policy.get_action_and_log_prob(SAMPLE_STATE)
        assert isinstance(action, float)


# ---------------------------------------------------------------------------
# REINFORCEPolicy (only when PyTorch is available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestREINFORCEPolicy:
    def test_get_action_returns_float(self):
        policy = REINFORCEPolicy(hidden_sizes=[16])
        action = policy.get_action(SAMPLE_STATE)
        assert isinstance(action, float)

    def test_get_action_train_returns_tuple(self):
        policy = REINFORCEPolicy(hidden_sizes=[16])
        result = policy.get_action_train(SAMPLE_STATE)
        assert len(result) == 2
