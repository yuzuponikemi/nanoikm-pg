"""Tests for InvertedPendulumEnv."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environments.pendulum import InvertedPendulumEnv


class TestInvertedPendulumEnvReset:
    """Test reset() behavior."""

    def test_reset_returns_ndarray(self):
        env = InvertedPendulumEnv()
        state = env.reset()
        assert isinstance(state, np.ndarray)

    def test_reset_state_dimension(self):
        env = InvertedPendulumEnv()
        state = env.reset()
        assert state.shape == (4,)

    def test_reset_state_dtype(self):
        env = InvertedPendulumEnv()
        state = env.reset()
        assert state.dtype == np.float64

    def test_reset_state_within_range(self):
        env = InvertedPendulumEnv()
        state = env.reset(seed=42)
        assert np.all(np.abs(state) <= 0.05)

    def test_reset_with_seed_reproducible(self):
        env = InvertedPendulumEnv()
        s1 = env.reset(seed=123)
        s2 = env.reset(seed=123)
        np.testing.assert_array_equal(s1, s2)

    def test_reset_with_initial_state(self):
        env = InvertedPendulumEnv()
        initial = np.array([0.1, 0.2, 0.05, -0.1])
        state = env.reset(initial_state=initial)
        np.testing.assert_array_almost_equal(state, initial)

    def test_reset_clears_step_counter(self):
        env = InvertedPendulumEnv()
        env.reset()
        env.step(0.0)
        env.step(0.0)
        env.reset()
        assert env.steps == 0

    def test_reset_clears_history(self):
        env = InvertedPendulumEnv()
        env.reset()
        env.step(0.0)
        env.reset()
        assert len(env.history["actions"]) == 0
        assert len(env.history["rewards"]) == 0
        assert len(env.history["states"]) == 1  # initial state


class TestInvertedPendulumEnvStep:
    """Test step() behavior."""

    def test_step_returns_tuple_of_four(self):
        env = InvertedPendulumEnv()
        env.reset(seed=0)
        result = env.step(0.0)
        assert len(result) == 4

    def test_step_state_shape(self):
        env = InvertedPendulumEnv()
        env.reset(seed=0)
        state, _, _, _ = env.step(0.0)
        assert state.shape == (4,)

    def test_step_state_dtype(self):
        env = InvertedPendulumEnv()
        env.reset(seed=0)
        state, _, _, _ = env.step(0.0)
        assert state.dtype == np.float64

    def test_step_reward_is_float(self):
        env = InvertedPendulumEnv()
        env.reset(seed=0)
        _, reward, _, _ = env.step(0.0)
        assert isinstance(reward, float)

    def test_step_done_is_bool(self):
        env = InvertedPendulumEnv()
        env.reset(seed=0)
        _, _, done, _ = env.step(0.0)
        assert isinstance(done, bool)

    def test_step_info_is_dict(self):
        env = InvertedPendulumEnv()
        env.reset(seed=0)
        _, _, _, info = env.step(0.0)
        assert isinstance(info, dict)

    def test_step_reward_one_when_not_done(self):
        env = InvertedPendulumEnv()
        env.reset(seed=0)
        _, reward, done, _ = env.step(0.0)
        if not done:
            assert reward == 1.0

    def test_step_increments_counter(self):
        env = InvertedPendulumEnv()
        env.reset(seed=0)
        env.step(0.0)
        assert env.steps == 1
        env.step(0.0)
        assert env.steps == 2

    def test_step_discrete_action(self):
        env = InvertedPendulumEnv()
        env.reset(seed=0)
        state, _, _, info = env.step(1)
        assert info["force"] == env.force_mag

    def test_step_discrete_action_left(self):
        env = InvertedPendulumEnv()
        env.reset(seed=0)
        _, _, _, info = env.step(0)
        assert info["force"] == -env.force_mag

    def test_step_continuous_action_clipped(self):
        env = InvertedPendulumEnv()
        env.reset(seed=0)
        _, _, _, info = env.step(100.0)
        assert info["force"] == env.force_mag

    def test_step_without_reset_raises(self):
        env = InvertedPendulumEnv()
        with pytest.raises(RuntimeError):
            env.step(0.0)

    def test_step_state_changes(self):
        env = InvertedPendulumEnv()
        s0 = env.reset(initial_state=np.array([0.0, 0.0, 0.1, 0.0]))
        s1, _, _, _ = env.step(0.0)
        assert not np.array_equal(s0, s1)

    def test_step_returns_copy(self):
        env = InvertedPendulumEnv()
        env.reset(seed=0)
        s1, _, _, _ = env.step(0.0)
        s1_orig = s1.copy()
        env.step(0.0)
        np.testing.assert_array_equal(s1, s1_orig)


class TestInvertedPendulumEnvDone:
    """Test done flag termination conditions."""

    def test_done_by_angle_positive(self):
        env = InvertedPendulumEnv()
        env.reset(initial_state=np.array([0.0, 0.0, env.theta_threshold + 0.01, 0.0]))
        _, _, done, info = env.step(0.0)
        assert done
        assert info["terminated_by_angle"]

    def test_done_by_angle_negative(self):
        env = InvertedPendulumEnv()
        env.reset(initial_state=np.array([0.0, 0.0, -(env.theta_threshold + 0.01), 0.0]))
        _, _, done, info = env.step(0.0)
        assert done
        assert info["terminated_by_angle"]

    def test_done_by_position_positive(self):
        env = InvertedPendulumEnv()
        env.reset(initial_state=np.array([env.x_threshold + 0.01, 0.0, 0.0, 0.0]))
        _, _, done, info = env.step(0.0)
        assert done
        assert info["terminated_by_position"]

    def test_done_by_position_negative(self):
        env = InvertedPendulumEnv()
        env.reset(initial_state=np.array([-(env.x_threshold + 0.01), 0.0, 0.0, 0.0]))
        _, _, done, info = env.step(0.0)
        assert done
        assert info["terminated_by_position"]

    def test_done_by_max_steps(self):
        env = InvertedPendulumEnv(max_steps=3)
        env.reset(initial_state=np.array([0.0, 0.0, 0.0, 0.0]))
        for _ in range(2):
            _, _, done, _ = env.step(0.0)
        assert not done
        _, _, done, info = env.step(0.0)
        assert done
        assert info["terminated_by_time"]

    def test_not_done_within_bounds(self):
        env = InvertedPendulumEnv()
        env.reset(initial_state=np.array([0.0, 0.0, 0.0, 0.0]))
        _, _, done, _ = env.step(0.0)
        assert not done

    def test_reward_zero_when_done(self):
        env = InvertedPendulumEnv()
        env.reset(initial_state=np.array([0.0, 0.0, env.theta_threshold + 0.1, 0.0]))
        _, reward, done, _ = env.step(0.0)
        assert done
        assert reward == 0.0
