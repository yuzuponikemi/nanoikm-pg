"""
Tests for InvertedPendulumEnv

Covers:
- reset() initializes state correctly
- step() returns the expected tuple shape and types
- State dimension and dtype
- done flag firing conditions (angle, position, max_steps)
"""

import numpy as np
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environments.pendulum import InvertedPendulumEnv


class TestInvertedPendulumEnvReset:
    """Tests for the reset() method."""

    def test_reset_returns_numpy_array(self):
        env = InvertedPendulumEnv()
        state = env.reset()
        assert isinstance(state, np.ndarray)

    def test_reset_state_shape(self):
        env = InvertedPendulumEnv()
        state = env.reset()
        assert state.shape == (4,)

    def test_reset_state_dtype(self):
        env = InvertedPendulumEnv()
        state = env.reset()
        assert state.dtype == np.float64

    def test_reset_default_state_within_small_range(self):
        """Default reset samples from [-0.05, 0.05]."""
        env = InvertedPendulumEnv()
        for _ in range(20):
            state = env.reset()
            assert np.all(np.abs(state) <= 0.05 + 1e-10), (
                f"State {state} exceeds [-0.05, 0.05]"
            )

    def test_reset_with_seed_is_reproducible(self):
        env = InvertedPendulumEnv()
        s1 = env.reset(seed=42)
        s2 = env.reset(seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_reset_with_custom_initial_state(self):
        env = InvertedPendulumEnv()
        custom = np.array([0.1, 0.2, 0.05, -0.1])
        state = env.reset(initial_state=custom)
        np.testing.assert_array_almost_equal(state, custom)

    def test_reset_resets_step_counter(self):
        env = InvertedPendulumEnv()
        env.reset()
        for _ in range(10):
            env.step(0.0)
        env.reset()
        assert env.steps == 0

    def test_reset_clears_history(self):
        env = InvertedPendulumEnv()
        env.reset()
        env.step(1.0)
        env.reset()
        assert len(env.history["actions"]) == 0
        assert len(env.history["rewards"]) == 0


class TestInvertedPendulumEnvStep:
    """Tests for the step() method."""

    def test_step_returns_four_element_tuple(self):
        env = InvertedPendulumEnv()
        env.reset()
        result = env.step(0.0)
        assert len(result) == 4

    def test_step_state_shape(self):
        env = InvertedPendulumEnv()
        env.reset()
        state, reward, done, info = env.step(0.0)
        assert isinstance(state, np.ndarray)
        assert state.shape == (4,)

    def test_step_state_dtype(self):
        env = InvertedPendulumEnv()
        env.reset()
        state, _, _, _ = env.step(0.0)
        assert state.dtype == np.float64

    def test_step_reward_is_float(self):
        env = InvertedPendulumEnv()
        env.reset()
        _, reward, _, _ = env.step(0.0)
        assert isinstance(reward, float)

    def test_step_done_is_bool(self):
        env = InvertedPendulumEnv()
        env.reset()
        _, _, done, _ = env.step(0.0)
        assert isinstance(done, bool)

    def test_step_info_is_dict(self):
        env = InvertedPendulumEnv()
        env.reset()
        _, _, _, info = env.step(0.0)
        assert isinstance(info, dict)

    def test_step_info_keys(self):
        env = InvertedPendulumEnv()
        env.reset()
        _, _, _, info = env.step(0.0)
        expected_keys = {
            "x", "x_dot", "theta", "theta_dot", "force", "steps",
            "terminated_by_angle", "terminated_by_position", "terminated_by_time",
        }
        assert expected_keys.issubset(info.keys())

    def test_step_without_reset_raises(self):
        env = InvertedPendulumEnv()
        with pytest.raises(RuntimeError):
            env.step(0.0)

    def test_step_reward_positive_when_not_done(self):
        env = InvertedPendulumEnv()
        env.reset(initial_state=[0.0, 0.0, 0.0, 0.0])
        _, reward, done, _ = env.step(0.0)
        if not done:
            assert reward == 1.0

    def test_step_increments_step_counter(self):
        env = InvertedPendulumEnv()
        env.reset()
        for i in range(5):
            env.step(0.0)
        assert env.steps == 5

    def test_step_continuous_action(self):
        env = InvertedPendulumEnv()
        env.reset()
        state, _, _, info = env.step(5.0)
        assert state.shape == (4,)
        assert info["force"] == pytest.approx(5.0)

    def test_step_discrete_action_left(self):
        env = InvertedPendulumEnv(force_mag=10.0)
        env.reset()
        _, _, _, info = env.step(0)
        assert info["force"] == pytest.approx(-10.0)

    def test_step_discrete_action_right(self):
        env = InvertedPendulumEnv(force_mag=10.0)
        env.reset()
        _, _, _, info = env.step(1)
        assert info["force"] == pytest.approx(10.0)

    def test_step_action_clipped_to_force_mag(self):
        env = InvertedPendulumEnv(force_mag=10.0)
        env.reset()
        _, _, _, info = env.step(999.0)
        assert info["force"] == pytest.approx(10.0)


class TestInvertedPendulumEnvDoneConditions:
    """Tests for episode termination conditions."""

    def test_done_by_angle_positive(self):
        """Episode ends when pole angle exceeds theta_threshold."""
        env = InvertedPendulumEnv(theta_threshold=0.2095)
        # Start close to threshold, apply force to tip it over
        env.reset(initial_state=[0.0, 0.0, 0.2090, 0.5])
        done = False
        for _ in range(50):
            _, _, done, info = env.step(10.0)
            if done:
                break
        assert done, "Expected done=True when pole angle exceeds threshold"

    def test_done_by_angle_negative(self):
        env = InvertedPendulumEnv(theta_threshold=0.2095)
        env.reset(initial_state=[0.0, 0.0, -0.2090, -0.5])
        done = False
        for _ in range(50):
            _, _, done, info = env.step(-10.0)
            if done:
                break
        assert done

    def test_done_by_position(self):
        """Episode ends when cart goes beyond x_threshold."""
        env = InvertedPendulumEnv(x_threshold=2.4)
        env.reset(initial_state=[2.3, 5.0, 0.0, 0.0])
        done = False
        for _ in range(50):
            _, _, done, info = env.step(10.0)
            if done:
                break
        assert done, "Expected done=True when cart position exceeds threshold"

    def test_done_by_max_steps(self):
        """Episode ends when max_steps is reached."""
        env = InvertedPendulumEnv(max_steps=5)
        env.reset(initial_state=[0.0, 0.0, 0.0, 0.0])
        states, actions, rewards, info = [], [], [], {}
        done = False
        steps = 0
        while not done:
            _, reward, done, info = env.step(0.0)
            steps += 1
        assert steps == 5
        assert info["terminated_by_time"]

    def test_not_done_in_balanced_state(self):
        """Balanced pole should not terminate immediately."""
        env = InvertedPendulumEnv()
        env.reset(initial_state=[0.0, 0.0, 0.0, 0.0])
        _, _, done, _ = env.step(0.0)
        assert not done

    def test_info_flags_angle_termination(self):
        env = InvertedPendulumEnv(theta_threshold=0.01)
        env.reset(initial_state=[0.0, 0.0, 0.02, 0.0])
        _, _, done, info = env.step(0.0)
        if done:
            assert info["terminated_by_angle"]

    def test_info_flags_position_termination(self):
        env = InvertedPendulumEnv(x_threshold=0.1)
        env.reset(initial_state=[0.09, 5.0, 0.0, 0.0])
        done = False
        info = {}
        for _ in range(10):
            _, _, done, info = env.step(10.0)
            if done:
                break
        if done:
            assert info["terminated_by_position"]


class TestInvertedPendulumEnvStateLabels:
    """Tests for helper methods."""

    def test_get_state_labels_returns_four_strings(self):
        env = InvertedPendulumEnv()
        labels = env.get_state_labels()
        assert len(labels) == 4
        assert all(isinstance(label, str) for label in labels)

    def test_get_normalized_state_shape(self):
        env = InvertedPendulumEnv()
        env.reset()
        normalized = env.get_normalized_state()
        assert normalized.shape == (4,)

    def test_get_energy_keys(self):
        env = InvertedPendulumEnv()
        env.reset()
        energy = env.get_energy()
        assert "kinetic" in energy
        assert "potential" in energy
        assert "total" in energy

    def test_get_energy_total_equals_sum(self):
        env = InvertedPendulumEnv()
        env.reset()
        energy = env.get_energy()
        assert energy["total"] == pytest.approx(
            energy["kinetic"] + energy["potential"], rel=1e-6
        )

    def test_render_ascii_returns_string(self):
        env = InvertedPendulumEnv()
        env.reset()
        result = env.render_ascii()
        assert isinstance(result, str)

    def test_get_history_keys(self):
        env = InvertedPendulumEnv()
        env.reset()
        env.step(0.0)
        history = env.get_history()
        assert "states" in history
        assert "actions" in history
        assert "rewards" in history
