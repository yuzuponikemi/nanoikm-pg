"""
Tests for PPO components

Covers:
- RolloutBuffer: push / clear / len
- PPOActorCriticNet: forward pass shapes
- PPOPolicy: get_action, get_action_train, push, update, get/set_params
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests._torch_available import TORCH_AVAILABLE
from src.policies.ppo_policy import RolloutBuffer

# Canonical test state
SAMPLE_STATE = np.array([0.1, 0.2, 0.05, -0.1], dtype=np.float32)
NEXT_STATE = np.array([0.2, 0.1, 0.03, -0.2], dtype=np.float32)
STATE_DIM = 4


# ---------------------------------------------------------------------------
# RolloutBuffer (no torch dependency)
# ---------------------------------------------------------------------------


class TestRolloutBuffer:
    def test_initially_empty(self):
        buf = RolloutBuffer()
        assert len(buf) == 0

    def test_push_increases_length(self):
        buf = RolloutBuffer()
        buf.push(SAMPLE_STATE, 1.0, -0.5, 1.0, False, 0.8)
        assert len(buf) == 1

    def test_multiple_pushes(self):
        buf = RolloutBuffer()
        for i in range(10):
            buf.push(SAMPLE_STATE, float(i), float(i * 0.1), 1.0, False, 0.5)
        assert len(buf) == 10

    def test_clear_resets_length(self):
        buf = RolloutBuffer()
        for _ in range(5):
            buf.push(SAMPLE_STATE, 1.0, -0.3, 0.5, False, 0.4)
        buf.clear()
        assert len(buf) == 0

    def test_stored_values_accessible(self):
        buf = RolloutBuffer()
        buf.push(SAMPLE_STATE, 3.5, -0.7, 2.0, True, 1.2)
        assert buf.actions[0] == pytest.approx(3.5)
        assert buf.log_probs[0] == pytest.approx(-0.7)
        assert buf.rewards[0] == pytest.approx(2.0)
        assert buf.dones[0] is True
        assert buf.values[0] == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# PPOActorCriticNet and PPOPolicy (require torch)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPPOActorCriticNet:
    def test_forward_output_shapes(self):
        import torch
        from src.policies.ppo_policy import PPOActorCriticNet

        net = PPOActorCriticNet(state_dim=STATE_DIM, hidden_sizes=[32, 32])
        x = torch.FloatTensor(SAMPLE_STATE).unsqueeze(0)
        action_mean, value, log_std = net(x)

        assert action_mean.shape == (1, 1)
        assert value.shape == (1, 1)
        assert log_std.shape == (1, 1)

    def test_batch_forward(self):
        import torch
        from src.policies.ppo_policy import PPOActorCriticNet

        net = PPOActorCriticNet(state_dim=STATE_DIM, hidden_sizes=[32, 32])
        batch = torch.FloatTensor(np.tile(SAMPLE_STATE, (8, 1)))
        action_mean, value, log_std = net(batch)

        assert action_mean.shape == (8, 1)
        assert value.shape == (8, 1)

    def test_action_mean_in_range(self):
        """Tanh output should be in (-1, 1)."""
        import torch
        from src.policies.ppo_policy import PPOActorCriticNet

        net = PPOActorCriticNet(state_dim=STATE_DIM, hidden_sizes=[32, 32])
        x = torch.FloatTensor(SAMPLE_STATE).unsqueeze(0)
        action_mean, _, _ = net(x)

        assert -1.0 < action_mean.item() < 1.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPPOPolicy:
    def _make_policy(self) -> "PPOPolicy":  # noqa: F821
        from src.policies.ppo_policy import PPOPolicy

        return PPOPolicy(
            hidden_sizes=[32, 32],
            action_low=-10.0,
            action_high=10.0,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            lr=3e-4,
            n_epochs=2,
            mini_batch_size=16,
        )

    def test_get_action_returns_float(self):
        policy = self._make_policy()
        action = policy.get_action(SAMPLE_STATE)
        assert isinstance(action, float)

    def test_get_action_in_valid_range(self):
        policy = self._make_policy()
        for _ in range(20):
            action = policy.get_action(SAMPLE_STATE)
            assert policy.action_low <= action <= policy.action_high

    def test_get_action_train_returns_tuple(self):
        policy = self._make_policy()
        result = policy.get_action_train(SAMPLE_STATE)
        assert len(result) == 3
        action, log_prob, value = result
        assert isinstance(action, float)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_get_action_train_action_in_range(self):
        policy = self._make_policy()
        for _ in range(20):
            action, _, _ = policy.get_action_train(SAMPLE_STATE)
            assert policy.action_low <= action <= policy.action_high

    def test_push_fills_buffer(self):
        policy = self._make_policy()
        assert len(policy.buffer) == 0
        action, log_prob, value = policy.get_action_train(SAMPLE_STATE)
        policy.push(SAMPLE_STATE, action, log_prob, 1.0, False, value)
        assert len(policy.buffer) == 1

    def test_update_returns_dict_with_losses(self):
        policy = self._make_policy()
        # Fill buffer with enough transitions
        for _ in range(32):
            action, log_prob, value = policy.get_action_train(SAMPLE_STATE)
            policy.push(SAMPLE_STATE, action, log_prob, 1.0, False, value)

        stats = policy.update(last_value=0.0)
        assert isinstance(stats, dict)
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "entropy" in stats
        assert "total_loss" in stats

    def test_update_clears_buffer(self):
        policy = self._make_policy()
        for _ in range(32):
            action, log_prob, value = policy.get_action_train(SAMPLE_STATE)
            policy.push(SAMPLE_STATE, action, log_prob, 1.0, False, value)

        policy.update(last_value=0.0)
        # Buffer should be cleared after update
        assert len(policy.buffer) == 0

    def test_update_empty_buffer_returns_zeros(self):
        policy = self._make_policy()
        stats = policy.update(last_value=0.0)
        assert stats["policy_loss"] == pytest.approx(0.0)
        assert stats["value_loss"] == pytest.approx(0.0)

    def test_update_increments_counter(self):
        policy = self._make_policy()
        assert policy._update_count == 0
        for _ in range(32):
            action, log_prob, value = policy.get_action_train(SAMPLE_STATE)
            policy.push(SAMPLE_STATE, action, log_prob, 1.0, False, value)
        policy.update()
        assert policy._update_count == 1

    def test_get_num_params_positive(self):
        policy = self._make_policy()
        n = policy.get_num_params()
        assert n > 0

    def test_get_params_set_params_roundtrip(self):
        import torch
        from src.policies.ppo_policy import PPOPolicy

        policy = self._make_policy()
        params = policy.get_params()

        # Create a new policy and load params
        policy2 = PPOPolicy(hidden_sizes=[32, 32])
        policy2.set_params({"net_state": params["net_state"]})

        # Both should produce the same action (deterministic)
        action1 = policy.get_action(SAMPLE_STATE)
        action2 = policy2.get_action(SAMPLE_STATE)
        assert action1 == pytest.approx(action2, abs=1e-5)

    def test_gae_computation_terminal(self):
        """GAE with done=True at the end should not propagate value."""
        policy = self._make_policy()
        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.5, 0.5]
        dones = [False, False, True]

        advantages, returns = policy._compute_gae(rewards, values, dones, last_value=0.0)
        assert advantages.shape == (3,)
        assert returns.shape == (3,)
        # Returns should be close to discounted reward sums
        assert returns[2].item() == pytest.approx(1.0, abs=1e-4)

    def test_repr(self):
        policy = self._make_policy()
        r = repr(policy)
        assert "PPOPolicy" in r
