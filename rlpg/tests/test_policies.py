"""Tests for policy implementations."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from policies.random_policy import RandomPolicy
from policies.linear_policy import LinearPolicy
from policies.neural_policy import NeuralNetworkPolicy


SAMPLE_STATE = np.array([0.01, -0.02, 0.05, 0.03])


class TestRandomPolicy:
    """Test RandomPolicy.get_action() behavior."""

    def test_get_action_returns_float(self):
        policy = RandomPolicy()
        action = policy.get_action(SAMPLE_STATE)
        assert isinstance(action, (float, np.floating))

    def test_get_action_within_range(self):
        policy = RandomPolicy(action_low=-5.0, action_high=5.0)
        for _ in range(100):
            action = policy.get_action(SAMPLE_STATE)
            assert -5.0 <= action <= 5.0

    def test_get_action_discrete(self):
        policy = RandomPolicy(action_low=-10.0, action_high=10.0, discrete=True)
        for _ in range(50):
            action = policy.get_action(SAMPLE_STATE)
            assert action in [-10.0, 10.0]

    def test_get_action_ignores_state(self):
        policy = RandomPolicy(seed=42)
        np.random.seed(42)
        a1 = policy.get_action(np.zeros(4))
        np.random.seed(42)
        policy2 = RandomPolicy(seed=42)
        a2 = policy2.get_action(np.ones(4) * 100)
        assert a1 == a2

    def test_num_params_zero(self):
        policy = RandomPolicy()
        assert policy.get_num_params() == 0


class TestLinearPolicy:
    """Test LinearPolicy.get_action() behavior."""

    def test_get_action_returns_float(self):
        policy = LinearPolicy()
        action = policy.get_action(SAMPLE_STATE)
        assert isinstance(action, float)

    def test_get_action_zero_weights_returns_bias(self):
        policy = LinearPolicy(weights=np.zeros(4), bias=3.0)
        action = policy.get_action(SAMPLE_STATE)
        assert action == 3.0

    def test_get_action_clipped_high(self):
        policy = LinearPolicy(
            weights=np.array([1000.0, 0.0, 0.0, 0.0]),
            action_high=10.0,
        )
        action = policy.get_action(np.array([1.0, 0.0, 0.0, 0.0]))
        assert action == 10.0

    def test_get_action_clipped_low(self):
        policy = LinearPolicy(
            weights=np.array([-1000.0, 0.0, 0.0, 0.0]),
            action_low=-10.0,
        )
        action = policy.get_action(np.array([1.0, 0.0, 0.0, 0.0]))
        assert action == -10.0

    def test_get_action_linear_computation(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        policy = LinearPolicy(weights=weights, bias=0.5, action_low=-100.0, action_high=100.0)
        state = np.array([1.0, 1.0, 1.0, 1.0])
        expected = np.dot(weights, state) + 0.5  # 10.5
        action = policy.get_action(state)
        assert abs(action - expected) < 1e-10

    def test_get_action_within_range(self):
        policy = LinearPolicy(action_low=-10.0, action_high=10.0)
        for _ in range(50):
            state = np.random.randn(4)
            action = policy.get_action(state)
            assert -10.0 <= action <= 10.0

    def test_num_params(self):
        policy = LinearPolicy()
        assert policy.get_num_params() == 5  # 4 weights + 1 bias

    def test_get_and_set_params(self):
        policy = LinearPolicy(weights=np.array([1.0, 2.0, 3.0, 4.0]), bias=0.5)
        params = policy.get_params()
        new_policy = LinearPolicy()
        new_policy.set_params(params)
        np.testing.assert_array_equal(new_policy.weights, policy.weights)
        assert new_policy.bias == policy.bias


class TestNeuralNetworkPolicy:
    """Test NeuralNetworkPolicy.get_action() behavior."""

    def test_get_action_returns_float(self):
        policy = NeuralNetworkPolicy(hidden_sizes=[16])
        action = policy.get_action(SAMPLE_STATE)
        assert isinstance(action, float)

    def test_get_action_within_range(self):
        policy = NeuralNetworkPolicy(
            hidden_sizes=[16], action_low=-10.0, action_high=10.0
        )
        for _ in range(50):
            state = np.random.randn(4)
            action = policy.get_action(state)
            assert -10.0 <= action <= 10.0

    def test_get_action_deterministic(self):
        policy = NeuralNetworkPolicy(hidden_sizes=[16])
        a1 = policy.get_action(SAMPLE_STATE)
        a2 = policy.get_action(SAMPLE_STATE)
        assert a1 == a2

    def test_different_hidden_sizes(self):
        for sizes in [[8], [32, 32], [16, 16, 16]]:
            policy = NeuralNetworkPolicy(hidden_sizes=sizes)
            action = policy.get_action(SAMPLE_STATE)
            assert isinstance(action, float)

    def test_num_params_positive(self):
        policy = NeuralNetworkPolicy(hidden_sizes=[16])
        assert policy.get_num_params() > 0

    def test_tanh_activation(self):
        policy = NeuralNetworkPolicy(hidden_sizes=[16], activation="tanh")
        action = policy.get_action(SAMPLE_STATE)
        assert isinstance(action, float)

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError):
            NeuralNetworkPolicy(hidden_sizes=[16], activation="invalid")

    def test_get_and_set_params(self):
        policy = NeuralNetworkPolicy(hidden_sizes=[16])
        a1 = policy.get_action(SAMPLE_STATE)
        params = policy.get_params()
        policy2 = NeuralNetworkPolicy(hidden_sizes=[16])
        policy2.set_params(params)
        a2 = policy2.get_action(SAMPLE_STATE)
        assert a1 == a2

    def test_flat_params_roundtrip(self):
        policy = NeuralNetworkPolicy(hidden_sizes=[16])
        a1 = policy.get_action(SAMPLE_STATE)
        flat = policy.get_flat_params()
        policy.set_flat_params(flat)
        a2 = policy.get_action(SAMPLE_STATE)
        assert abs(a1 - a2) < 1e-6
