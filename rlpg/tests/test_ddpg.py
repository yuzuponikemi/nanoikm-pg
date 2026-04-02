"""
Unit tests for DDPG policy.

Tests cover:
- Ornstein-Uhlenbeck noise generation
- Replay buffer operations
- Actor and Critic network shapes
- DDPG policy action selection and training
"""

import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from src.policies import (
        DDPGPolicy, DDPGActorNet, DDPGCriticNet,
        DDPGReplayBuffer, OUNoise
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestOUNoise:
    """Test Ornstein-Uhlenbeck noise generation."""

    def test_ou_noise_initialization(self):
        """Test OU noise initialization."""
        noise = OUNoise(action_dim=1, mu=0.0, theta=0.15, sigma=0.3)
        assert noise.action_dim == 1
        assert np.allclose(noise.x, [0.0])

    def test_ou_noise_sample_shape(self):
        """Test OU noise sample output shape."""
        noise = OUNoise(action_dim=1)
        sample = noise.sample()
        assert sample.shape == (1,)

    def test_ou_noise_reset(self):
        """Test OU noise reset functionality."""
        noise = OUNoise(action_dim=1, mu=0.0)
        noise.x = np.array([10.0])
        noise.reset()
        assert np.allclose(noise.x, [0.0])

    def test_ou_noise_mean_reversion(self):
        """Test that OU noise reverts to mean over time."""
        noise = OUNoise(action_dim=1, mu=0.0, theta=1.0, sigma=0.0)  # No noise
        noise.reset()
        # With sigma=0, noise should decay toward mu
        for _ in range(100):
            sample = noise.sample()
        assert np.abs(sample[0]) < 0.01  # Should be close to 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDDPGReplayBuffer:
    """Test DDPG replay buffer operations."""

    def test_replay_buffer_initialization(self):
        """Test replay buffer creation."""
        buffer = DDPGReplayBuffer(capacity=100)
        assert len(buffer) == 0
        assert buffer.capacity == 100

    def test_replay_buffer_push(self):
        """Test pushing transitions to the buffer."""
        buffer = DDPGReplayBuffer(capacity=10)
        state = np.array([1.0, 2.0, 3.0, 4.0])
        action = 5.0
        reward = 10.0
        next_state = np.array([1.1, 2.1, 3.1, 4.1])
        done = False

        buffer.push(state, action, reward, next_state, done)
        assert len(buffer) == 1

    def test_replay_buffer_sample(self):
        """Test sampling from the buffer."""
        buffer = DDPGReplayBuffer(capacity=100)
        for i in range(10):
            state = np.array([float(i)] * 4)
            buffer.push(state, float(i), float(i * 10), state + 0.1, False)

        batch = buffer.sample(5)
        assert len(batch) == 5
        # Check fields
        assert hasattr(batch[0], 'state')
        assert hasattr(batch[0], 'action')
        assert hasattr(batch[0], 'reward')
        assert hasattr(batch[0], 'next_state')
        assert hasattr(batch[0], 'done')

    def test_replay_buffer_capacity_eviction(self):
        """Test FIFO eviction when buffer reaches capacity."""
        buffer = DDPGReplayBuffer(capacity=5)
        for i in range(10):
            state = np.array([float(i)] * 4)
            buffer.push(state, float(i), float(i), state + 0.1, False)

        assert len(buffer) == 5  # Only last 5 stored
        batch = buffer.sample(5)
        # Should contain transitions with indices 5-9 (FIFO eviction)
        actions = [t.action for t in batch]
        assert all(5 <= a < 10 for a in actions)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDDPGActorNet:
    """Test DDPG Actor network."""

    def test_actor_net_output_shape(self):
        """Test actor network output shape."""
        actor = DDPGActorNet(state_dim=4, action_dim=1, hidden_sizes=[64, 64])
        state = torch.randn(8, 4)  # Batch of 8
        action = actor(state)
        assert action.shape == (8, 1)

    def test_actor_net_output_range(self):
        """Test actor network output is in [-1, 1]."""
        actor = DDPGActorNet(state_dim=4, action_dim=1, hidden_sizes=[64, 64])
        state = torch.randn(100, 4)
        action = actor(state)
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0)

    def test_actor_net_deterministic(self):
        """Test actor is deterministic (same input → same output)."""
        actor = DDPGActorNet(state_dim=4, action_dim=1, hidden_sizes=[64, 64])
        state = torch.randn(1, 4)
        with torch.no_grad():
            action1 = actor(state)
            action2 = actor(state)
        assert torch.allclose(action1, action2)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDDPGCriticNet:
    """Test DDPG Critic network."""

    def test_critic_net_output_shape(self):
        """Test critic network output shape."""
        critic = DDPGCriticNet(state_dim=4, action_dim=1, hidden_sizes=[64, 64])
        state = torch.randn(8, 4)  # Batch of 8
        action = torch.randn(8, 1)
        q_value = critic(state, action)
        assert q_value.shape == (8, 1)

    def test_critic_net_forward_pass(self):
        """Test critic network forward pass."""
        critic = DDPGCriticNet(state_dim=4, action_dim=1, hidden_sizes=[64, 64])
        state = torch.randn(1, 4)
        action = torch.randn(1, 1)
        q_value = critic(state, action)
        assert q_value.requires_grad  # Should be differentiable


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDDPGPolicy:
    """Test DDPG policy."""

    def test_ddpg_policy_initialization(self):
        """Test DDPG policy initialization."""
        policy = DDPGPolicy(
            hidden_sizes=[64, 64],
            action_low=-10.0,
            action_high=10.0,
        )
        assert policy.action_low == -10.0
        assert policy.action_high == 10.0
        assert policy.gamma == 0.99
        assert policy.step_count == 0

    def test_ddpg_policy_get_action(self):
        """Test deterministic action selection."""
        policy = DDPGPolicy()
        state = np.array([0.0, 0.0, 0.1, 0.0])
        action = policy.get_action(state)
        assert isinstance(action, (float, np.floating))
        assert -10.0 <= action <= 10.0

    def test_ddpg_policy_get_action_train_warmup(self):
        """Test action selection during warmup (random exploration)."""
        policy = DDPGPolicy(warmup_steps=100)
        state = np.array([0.0, 0.0, 0.1, 0.0])

        # During warmup, actions should be random
        actions = [policy.get_action_train(state) for _ in range(5)]
        assert all(-10.0 <= a <= 10.0 for a in actions)
        # Should have some variance during warmup
        assert len(set(actions)) > 1 or len(actions) == 1

    def test_ddpg_policy_get_action_train_with_noise(self):
        """Test action selection with OU noise (post-warmup)."""
        policy = DDPGPolicy(warmup_steps=0)  # No warmup
        state = np.array([0.0, 0.0, 0.1, 0.0])

        # After warmup, actions should have noise
        actions = [policy.get_action_train(state) for _ in range(10)]
        assert all(-10.0 <= a <= 10.0 for a in actions)

    def test_ddpg_policy_push(self):
        """Test storing transitions."""
        policy = DDPGPolicy()
        state = np.array([0.0, 0.0, 0.1, 0.0])
        action = 5.0
        reward = 10.0
        next_state = np.array([0.0, 0.0, 0.15, 0.1])
        done = False

        policy.push(state, action, reward, next_state, done)
        assert len(policy.replay_buffer) == 1
        assert policy.step_count == 1

    def test_ddpg_policy_update_insufficient_buffer(self):
        """Test update with insufficient buffer samples."""
        policy = DDPGPolicy()
        state = np.array([0.0, 0.0, 0.1, 0.0])

        # Not enough samples to train
        policy.push(state, 5.0, 10.0, state + 0.01, False)
        stats = policy.update(batch_size=32)

        assert stats["actor_loss"] == 0.0
        assert stats["critic_loss"] == 0.0

    def test_ddpg_policy_update_with_sufficient_buffer(self):
        """Test update with sufficient buffer samples."""
        policy = DDPGPolicy(warmup_steps=0)
        state = np.array([0.0, 0.0, 0.1, 0.0])

        # Fill buffer with more than batch_size samples
        for i in range(100):
            next_state = state + np.random.randn(4) * 0.01
            action = policy.get_action(state)
            policy.push(state, action, 10.0 - i * 0.1, next_state, i >= 99)
            state = next_state

        stats = policy.update(batch_size=32, n_updates=1)
        assert "actor_loss" in stats
        assert "critic_loss" in stats
        # Losses should be finite
        assert np.isfinite(stats["actor_loss"])
        assert np.isfinite(stats["critic_loss"])

    def test_ddpg_policy_soft_target_update(self):
        """Test soft target network updates."""
        policy = DDPGPolicy(tau=0.005, warmup_steps=0)
        state = np.array([0.0, 0.0, 0.1, 0.0])

        # Fill buffer and update
        for i in range(100):
            action = policy.get_action(state)
            next_state = state + np.random.randn(4) * 0.01
            policy.push(state, action, 1.0, next_state, False)
            state = next_state

        policy.update(batch_size=32, n_updates=5)

        # Check that target networks have been updated
        # (parameters should be different from main networks after updates)
        actor_params = list(policy.actor.parameters())
        actor_target_params = list(policy.actor_target.parameters())

        # With soft updates, should not be exactly equal (tau = 0.005)
        for p1, p2 in zip(actor_params, actor_target_params):
            # They should be close but not identical
            assert not torch.allclose(p1, p2, atol=1e-6)

    def test_ddpg_policy_get_set_params(self):
        """Test parameter serialization."""
        policy = DDPGPolicy()
        params = policy.get_params()

        assert "actor" in params
        assert "critic" in params
        assert "actor_target" in params
        assert "critic_target" in params

        # Load params into a new policy
        policy2 = DDPGPolicy()
        policy2.set_params(params)

        # Policies should give same action for same state
        state = np.array([0.0, 0.0, 0.1, 0.0])
        with torch.no_grad():
            action1 = policy.get_action(state)
            action2 = policy2.get_action(state)
        assert np.isclose(action1, action2)

    def test_ddpg_policy_get_stats(self):
        """Test statistics tracking."""
        policy = DDPGPolicy(warmup_steps=100)
        state = np.array([0.0, 0.0, 0.1, 0.0])

        # Push some transitions
        for i in range(50):
            policy.push(state, 5.0, 1.0, state + 0.01, False)

        stats = policy.get_stats()
        assert stats["replay_buffer_size"] == 50
        assert stats["step_count"] == 50
        assert stats["warmup_steps"] == 100
        assert stats["in_warmup"] is True

    def test_ddpg_policy_integration(self):
        """Integration test: collect episode and train."""
        policy = DDPGPolicy(warmup_steps=50, hidden_sizes=[32, 32])

        # Mock environment step
        state = np.array([0.0, 0.0, 0.0, 0.0])
        episode_reward = 0.0

        for step in range(100):
            action = policy.get_action_train(state)
            # Simulate next state and reward
            next_state = state + np.random.randn(4) * 0.1
            reward = 1.0 - 0.01 * np.sum(np.abs(state))
            done = step >= 99
            episode_reward += reward

            policy.push(state, action, reward, next_state, done)

            # Train after warmup
            if policy.step_count > 50:
                stats = policy.update(batch_size=32, n_updates=1)
                assert np.isfinite(stats["actor_loss"])
                assert np.isfinite(stats["critic_loss"])

            state = next_state

        assert episode_reward > 0  # Should accumulate some reward


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
