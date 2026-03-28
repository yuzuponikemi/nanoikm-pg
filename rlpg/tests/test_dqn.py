"""
Tests for DQN components

Covers:
- ReplayBuffer: push / sample / capacity eviction
- QNetwork: forward pass shape
- DQNPolicy: get_action, get_action_train, push, update, decay_epsilon, get/set_params
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests._torch_available import TORCH_AVAILABLE
from src.policies.dqn_policy import ReplayBuffer

# Canonical state for testing
SAMPLE_STATE      = np.array([0.1, 0.2, 0.05, -0.1], dtype=np.float32)
NEXT_STATE        = np.array([0.2, 0.1, 0.03, -0.2], dtype=np.float32)
STATE_DIM         = 4
N_ACTIONS         = 11


# ---------------------------------------------------------------------------
# ReplayBuffer (no torch dependency)
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_push_increases_length(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0
        buf.push(SAMPLE_STATE, 3, 1.0, NEXT_STATE, False)
        assert len(buf) == 1

    def test_sample_returns_correct_batch_size(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(SAMPLE_STATE, i % N_ACTIONS, float(i), NEXT_STATE, False)
        batch = buf.sample(16)
        assert len(batch) == 16

    def test_capacity_eviction(self):
        cap = 10
        buf = ReplayBuffer(capacity=cap)
        for i in range(20):
            buf.push(SAMPLE_STATE, 0, float(i), NEXT_STATE, False)
        assert len(buf) == cap  # FIFO eviction keeps capacity

    def test_transition_fields(self):
        buf = ReplayBuffer(capacity=10)
        buf.push(SAMPLE_STATE, 5, 2.5, NEXT_STATE, True)
        t = buf.sample(1)[0]
        np.testing.assert_array_equal(t.state, SAMPLE_STATE.astype(np.float32))
        assert t.action_idx == 5
        assert t.reward == pytest.approx(2.5)
        np.testing.assert_array_equal(t.next_state, NEXT_STATE.astype(np.float32))
        assert t.done is True

    def test_repr(self):
        buf = ReplayBuffer(capacity=50)
        r = repr(buf)
        assert 'ReplayBuffer' in r


# ---------------------------------------------------------------------------
# QNetwork and DQNPolicy (require torch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestQNetwork:
    def test_output_shape(self):
        import torch
        from src.policies.dqn_policy import QNetwork
        net = QNetwork(state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden_sizes=[32, 32])
        x = torch.FloatTensor(SAMPLE_STATE).unsqueeze(0)
        out = net(x)
        assert out.shape == (1, N_ACTIONS)

    def test_batch_output_shape(self):
        import torch
        from src.policies.dqn_policy import QNetwork
        net = QNetwork(state_dim=STATE_DIM, n_actions=N_ACTIONS)
        batch = torch.FloatTensor(np.tile(SAMPLE_STATE, (8, 1)))
        out = net(batch)
        assert out.shape == (8, N_ACTIONS)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDQNPolicy:
    def _make_policy(self) -> "DQNPolicy":  # noqa: F821
        from src.policies.dqn_policy import DQNPolicy
        return DQNPolicy(
            n_actions=N_ACTIONS,
            hidden_sizes=[32, 32],
            lr=1e-3,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.9,
            batch_size=8,
            buffer_capacity=100,
            target_update_freq=10,
        )

    def test_get_action_returns_float(self):
        policy = self._make_policy()
        action = policy.get_action(SAMPLE_STATE)
        assert isinstance(action, float)

    def test_get_action_within_action_values(self):
        policy = self._make_policy()
        # greedy action must be one of the discrete action_values
        for _ in range(20):
            action = policy.get_action(SAMPLE_STATE)
            assert action in policy.action_values

    def test_get_action_train_returns_tuple(self):
        policy = self._make_policy()
        result = policy.get_action_train(SAMPLE_STATE)
        assert isinstance(result, tuple)
        action, action_idx = result
        assert isinstance(action, float)
        assert 0 <= action_idx < N_ACTIONS

    def test_epsilon_greedy_exploration(self):
        """With epsilon=1.0 every action should be random."""
        policy = self._make_policy()
        policy.epsilon = 1.0
        actions = [policy.get_action_train(SAMPLE_STATE)[1] for _ in range(50)]
        # All indices should be in valid range
        assert all(0 <= a < N_ACTIONS for a in actions)

    def test_push_increases_buffer(self):
        policy = self._make_policy()
        assert len(policy.buffer) == 0
        policy.push(SAMPLE_STATE, 3, 1.0, NEXT_STATE, False)
        assert len(policy.buffer) == 1

    def test_update_returns_none_when_buffer_small(self):
        policy = self._make_policy()
        # Buffer is empty → update should return None
        result = policy.update()
        assert result is None

    def test_update_returns_loss_after_enough_transitions(self):
        policy = self._make_policy()
        # Fill buffer beyond batch_size
        for _ in range(20):
            policy.push(SAMPLE_STATE, 0, 1.0, NEXT_STATE, False)
        loss = policy.update()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_decay_epsilon(self):
        policy = self._make_policy()
        initial_eps = policy.epsilon
        policy.decay_epsilon()
        assert policy.epsilon < initial_eps
        # Repeated decay should not go below epsilon_min
        for _ in range(1000):
            policy.decay_epsilon()
        assert policy.epsilon >= policy.epsilon_min

    def test_target_network_syncs(self):
        import torch
        policy = self._make_policy()
        # Modify online network
        for p in policy.online_net.parameters():
            p.data.fill_(1.0)
        # Fill buffer and force enough steps to trigger target sync
        for _ in range(20):
            policy.push(SAMPLE_STATE, 0, 1.0, NEXT_STATE, False)
        for _ in range(policy.target_update_freq + 1):
            policy.update()
        # After sync target should match (within float tolerance)
        for p_on, p_tg in zip(policy.online_net.parameters(), policy.target_net.parameters()):
            assert torch.allclose(p_on, p_tg, atol=1e-5)

    def test_get_params_set_params_roundtrip(self):
        policy = self._make_policy()
        params = policy.get_params()
        # Modify epsilon
        policy.epsilon = 0.5
        policy.set_params({"epsilon": 0.99})
        assert policy.epsilon == pytest.approx(0.99)

    def test_get_stats(self):
        policy = self._make_policy()
        stats = policy.get_stats()
        assert "epsilon" in stats
        assert "step_count" in stats
        assert "buffer_size" in stats

    def test_repr(self):
        policy = self._make_policy()
        r = repr(policy)
        assert "DQNPolicy" in r
