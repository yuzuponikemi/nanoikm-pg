"""
Tests for SAC components

Covers:
- SACReplayBuffer: push / sample / capacity eviction / transition fields
- SACActorNet: forward pass shape, log_std clamping
- SACCriticNet: forward pass shape
- SACPolicy: get_action, get_action_train, push, update, get/set_params, get_stats
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests._torch_available import TORCH_AVAILABLE
from src.policies.sac_policy import SACReplayBuffer

# Canonical state for testing
SAMPLE_STATE = np.array([0.1, 0.2, 0.05, -0.1], dtype=np.float32)
NEXT_STATE   = np.array([0.2, 0.1, 0.03, -0.2], dtype=np.float32)
STATE_DIM    = 4
ACTION_DIM   = 1


# ---------------------------------------------------------------------------
# SACReplayBuffer (no torch dependency)
# ---------------------------------------------------------------------------

class TestSACReplayBuffer:
    def test_push_increases_length(self):
        buf = SACReplayBuffer(capacity=100)
        assert len(buf) == 0
        buf.push(SAMPLE_STATE, 1.5, 1.0, NEXT_STATE, False)
        assert len(buf) == 1

    def test_sample_correct_batch_size(self):
        buf = SACReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(SAMPLE_STATE, float(i) * 0.1, float(i), NEXT_STATE, False)
        batch = buf.sample(16)
        assert len(batch) == 16

    def test_capacity_eviction(self):
        cap = 10
        buf = SACReplayBuffer(capacity=cap)
        for i in range(20):
            buf.push(SAMPLE_STATE, 1.0, float(i), NEXT_STATE, False)
        assert len(buf) == cap

    def test_transition_fields(self):
        buf = SACReplayBuffer(capacity=10)
        buf.push(SAMPLE_STATE, 3.5, 2.0, NEXT_STATE, True)
        t = buf.sample(1)[0]
        np.testing.assert_array_almost_equal(t.state, SAMPLE_STATE)
        assert t.action == pytest.approx(3.5)
        assert t.reward == pytest.approx(2.0)
        np.testing.assert_array_almost_equal(t.next_state, NEXT_STATE)
        assert t.done is True

    def test_repr(self):
        buf = SACReplayBuffer(capacity=50)
        assert "SACReplayBuffer" in repr(buf)


# ---------------------------------------------------------------------------
# SACActorNet and SACCriticNet (require torch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSACActorNet:
    def test_output_shapes(self):
        import torch
        from src.policies.sac_policy import SACActorNet
        net = SACActorNet(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_sizes=[32, 32])
        x = torch.FloatTensor(SAMPLE_STATE).unsqueeze(0)
        mean, log_std = net(x)
        assert mean.shape == (1, ACTION_DIM)
        assert log_std.shape == (1, ACTION_DIM)

    def test_log_std_clamped(self):
        import torch
        from src.policies.sac_policy import SACActorNet, LOG_STD_MIN, LOG_STD_MAX
        net = SACActorNet(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_sizes=[32, 32])
        x = torch.FloatTensor(SAMPLE_STATE).unsqueeze(0)
        _, log_std = net(x)
        assert (log_std >= LOG_STD_MIN).all()
        assert (log_std <= LOG_STD_MAX).all()

    def test_batch_forward(self):
        import torch
        from src.policies.sac_policy import SACActorNet
        net = SACActorNet(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        batch = torch.FloatTensor(np.tile(SAMPLE_STATE, (8, 1)))
        mean, log_std = net(batch)
        assert mean.shape == (8, ACTION_DIM)
        assert log_std.shape == (8, ACTION_DIM)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSACCriticNet:
    def test_output_shape(self):
        import torch
        from src.policies.sac_policy import SACCriticNet
        net = SACCriticNet(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_sizes=[32, 32])
        s = torch.FloatTensor(SAMPLE_STATE).unsqueeze(0)
        a = torch.FloatTensor([[1.5]])
        q = net(s, a)
        assert q.shape == (1, 1)

    def test_batch_forward(self):
        import torch
        from src.policies.sac_policy import SACCriticNet
        net = SACCriticNet(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        s = torch.FloatTensor(np.tile(SAMPLE_STATE, (8, 1)))
        a = torch.FloatTensor(np.random.uniform(-1, 1, (8, 1)).astype(np.float32))
        q = net(s, a)
        assert q.shape == (8, 1)


# ---------------------------------------------------------------------------
# SACPolicy (require torch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSACPolicy:
    def _make_policy(self, warmup: int = 0) -> "SACPolicy":  # noqa: F821
        from src.policies.sac_policy import SACPolicy
        return SACPolicy(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            action_scale=10.0,
            action_bias=0.0,
            hidden_sizes=[32, 32],
            lr=1e-3,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            auto_alpha=True,
            batch_size=8,
            buffer_capacity=200,
            warmup_steps=warmup,
        )

    def test_get_action_returns_float(self):
        policy = self._make_policy()
        action = policy.get_action(SAMPLE_STATE)
        assert isinstance(action, float)

    def test_get_action_within_range(self):
        policy = self._make_policy()
        for _ in range(20):
            action = policy.get_action(SAMPLE_STATE)
            assert -10.0 <= action <= 10.0

    def test_get_action_train_returns_float(self):
        policy = self._make_policy(warmup=0)
        action = policy.get_action_train(SAMPLE_STATE)
        assert isinstance(action, float)

    def test_warmup_random_action(self):
        """During warmup (total_pushes < warmup_steps) actions are random."""
        policy = self._make_policy(warmup=1000)
        # Should still return a float in valid range
        for _ in range(10):
            action = policy.get_action_train(SAMPLE_STATE)
            assert -10.0 <= action <= 10.0

    def test_push_increases_buffer(self):
        policy = self._make_policy()
        assert len(policy.buffer) == 0
        policy.push(SAMPLE_STATE, 1.0, 1.0, NEXT_STATE, False)
        assert len(policy.buffer) == 1

    def test_update_returns_none_when_buffer_small(self):
        policy = self._make_policy()
        assert policy.update() is None

    def test_update_returns_dict_after_enough_transitions(self):
        policy = self._make_policy()
        for _ in range(20):
            policy.push(SAMPLE_STATE, 1.0, 1.0, NEXT_STATE, False)
        result = policy.update()
        assert result is not None
        assert "critic1_loss" in result
        assert "critic2_loss" in result
        assert "actor_loss" in result
        assert "alpha" in result

    def test_losses_are_non_negative(self):
        policy = self._make_policy()
        for _ in range(20):
            policy.push(SAMPLE_STATE, 1.0, 1.0, NEXT_STATE, False)
        stats = policy.update()
        assert stats["critic1_loss"] >= 0.0
        assert stats["critic2_loss"] >= 0.0

    def test_alpha_auto_tuning(self):
        policy = self._make_policy()
        initial_alpha = policy.alpha
        for _ in range(20):
            policy.push(SAMPLE_STATE, 1.0, 1.0, NEXT_STATE, False)
        for _ in range(5):
            policy.update()
        # alpha should change (auto-tuning is on)
        # (it's non-deterministic which direction, just verify it's positive)
        assert policy.alpha > 0.0

    def test_soft_target_update(self):
        import torch
        policy = self._make_policy()
        # Force critics to ones, target critics to zeros
        for p in policy.critic1.parameters():
            p.data.fill_(1.0)
        for p in policy.target_critic1.parameters():
            p.data.fill_(0.0)

        policy._soft_update_targets()

        for p in policy.target_critic1.parameters():
            expected = policy.tau * 1.0 + (1 - policy.tau) * 0.0
            assert torch.allclose(p, torch.full_like(p, expected), atol=1e-5)

    def test_get_set_params_roundtrip(self):
        policy = self._make_policy()
        params = policy.get_params()
        assert "actor_state" in params
        # Set back (smoke test)
        policy.set_params(params)

    def test_get_stats(self):
        policy = self._make_policy()
        stats = policy.get_stats()
        assert "alpha" in stats
        assert "step_count" in stats
        assert "buffer_size" in stats

    def test_step_count_increments(self):
        policy = self._make_policy()
        for _ in range(20):
            policy.push(SAMPLE_STATE, 1.0, 1.0, NEXT_STATE, False)
        policy.update()
        assert policy._step_count == 1

    def test_repr(self):
        policy = self._make_policy()
        r = repr(policy)
        assert "SACPolicy" in r

    def test_get_num_params(self):
        policy = self._make_policy()
        n = policy.get_num_params()
        assert n > 0
