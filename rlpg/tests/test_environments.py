"""
Unit tests for InvertedPendulumEnv

Tests cover:
- reset() / step() の基本動作
- 状態の次元数・型の確認
- doneフラグの発火条件
- 物理量の妥当性
"""

import numpy as np
import pytest
from src.environments.pendulum import InvertedPendulumEnv


class TestInvertedPendulumReset:
    """reset() の動作テスト"""

    def test_reset_returns_array(self, env):
        state = env.reset()
        assert isinstance(state, np.ndarray)

    def test_reset_state_shape(self, env):
        state = env.reset()
        assert state.shape == (4,), f"Expected shape (4,), got {state.shape}"

    def test_reset_state_dtype(self, env):
        state = env.reset()
        assert state.dtype == np.float64

    def test_reset_state_labels(self, env):
        labels = env.get_state_labels()
        assert len(labels) == 4

    def test_reset_default_near_zero(self, env):
        """デフォルトのリセットは原点付近に初期化される"""
        for _ in range(20):
            state = env.reset()
            assert np.all(np.abs(state) < 0.5), (
                f"Initial state should be small, got {state}"
            )

    def test_reset_with_seed_reproducible(self, env):
        state1 = env.reset(seed=0)
        state2 = env.reset(seed=0)
        np.testing.assert_array_equal(state1, state2)

    def test_reset_with_different_seeds(self, env):
        state1 = env.reset(seed=1)
        state2 = env.reset(seed=2)
        assert not np.array_equal(state1, state2)

    def test_reset_with_explicit_initial_state(self, env):
        target = np.array([0.1, 0.0, 0.05, 0.0])
        state = env.reset(initial_state=target)
        np.testing.assert_array_almost_equal(state, target)

    def test_reset_clears_step_counter(self, env):
        env.reset()
        for _ in range(10):
            env.step(0.0)
        env.reset()
        assert env.steps == 0

    def test_reset_clears_history(self, env):
        env.reset()
        for _ in range(5):
            env.step(0.0)
        env.reset()
        # history は reset 直後に初期状態の1エントリのみ
        assert len(env.history['states']) == 1
        assert len(env.history['actions']) == 0
        assert len(env.history['rewards']) == 0


class TestInvertedPendulumStep:
    """step() の動作テスト"""

    def test_step_returns_tuple(self, env):
        env.reset()
        result = env.step(0.0)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_step_state_shape(self, env):
        env.reset()
        state, reward, done, info = env.step(0.0)
        assert state.shape == (4,)

    def test_step_state_dtype(self, env):
        env.reset()
        state, reward, done, info = env.step(0.0)
        assert state.dtype == np.float64

    def test_step_reward_type(self, env):
        env.reset()
        _, reward, _, _ = env.step(0.0)
        assert isinstance(reward, float)

    def test_step_done_type(self, env):
        env.reset()
        _, _, done, _ = env.step(0.0)
        assert isinstance(done, bool)

    def test_step_info_type(self, env):
        env.reset()
        _, _, _, info = env.step(0.0)
        assert isinstance(info, dict)

    def test_step_info_keys(self, env):
        env.reset()
        _, _, _, info = env.step(0.0)
        expected_keys = {'x', 'x_dot', 'theta', 'theta_dot', 'force', 'steps'}
        assert expected_keys.issubset(info.keys())

    def test_step_reward_positive_when_not_done(self, env):
        """ポールが倒れていない間は報酬 1.0"""
        env.reset(initial_state=[0.0, 0.0, 0.0, 0.0])
        _, reward, done, _ = env.step(0.0)
        if not done:
            assert reward == 1.0

    def test_step_reward_zero_when_done(self, env):
        """エピソード終了時は報酬 0.0"""
        # 大きな角度で初期化してすぐ終了させる
        env.reset(initial_state=[0.0, 0.0, 0.5, 0.0])  # theta > theta_threshold
        _, reward, done, _ = env.step(0.0)
        if done:
            assert reward == 0.0

    def test_step_increments_counter(self, env):
        env.reset()
        for i in range(5):
            env.step(0.0)
            assert env.steps == i + 1

    def test_step_without_reset_raises(self):
        """reset()前にstep()を呼ぶとRuntimeErrorが発生する"""
        fresh_env = InvertedPendulumEnv()
        with pytest.raises(RuntimeError):
            fresh_env.step(0.0)

    def test_step_discrete_action_left(self, env):
        env.reset(initial_state=[0.0, 0.0, 0.0, 0.0])
        _, _, _, info = env.step(0)
        assert info['force'] == pytest.approx(-env.force_mag)

    def test_step_discrete_action_right(self, env):
        env.reset(initial_state=[0.0, 0.0, 0.0, 0.0])
        _, _, _, info = env.step(1)
        assert info['force'] == pytest.approx(env.force_mag)

    def test_step_continuous_action_clipped(self, env):
        """force_magを超えた連続アクションはクリップされる"""
        env.reset()
        large_action = 1000.0
        _, _, _, info = env.step(large_action)
        assert info['force'] == pytest.approx(env.force_mag)

    def test_step_state_changes(self, env):
        """ステップ後に状態が変化する（静止状態でも物理が動く）"""
        state0 = env.reset(initial_state=[0.0, 0.0, 0.01, 0.0])
        state1, _, _, _ = env.step(0.0)
        assert not np.allclose(state0, state1), "State should change after step"


class TestDoneConditions:
    """doneフラグの発火条件テスト"""

    def test_done_by_angle_positive(self):
        """theta > theta_threshold でエピソード終了"""
        env = InvertedPendulumEnv(theta_threshold=0.2095)
        env.reset(initial_state=[0.0, 0.0, 0.3, 0.0])  # theta=0.3 > 0.2095
        _, _, done, info = env.step(0.0)
        assert done
        assert info['terminated_by_angle']

    def test_done_by_angle_negative(self):
        """theta < -theta_threshold でエピソード終了"""
        env = InvertedPendulumEnv(theta_threshold=0.2095)
        env.reset(initial_state=[0.0, 0.0, -0.3, 0.0])
        _, _, done, info = env.step(0.0)
        assert done
        assert info['terminated_by_angle']

    def test_done_by_position_positive(self):
        """x > x_threshold でエピソード終了"""
        env = InvertedPendulumEnv(x_threshold=2.4)
        env.reset(initial_state=[3.0, 0.0, 0.0, 0.0])  # x=3.0 > 2.4
        _, _, done, info = env.step(0.0)
        assert done
        assert info['terminated_by_position']

    def test_done_by_position_negative(self):
        """x < -x_threshold でエピソード終了"""
        env = InvertedPendulumEnv(x_threshold=2.4)
        env.reset(initial_state=[-3.0, 0.0, 0.0, 0.0])
        _, _, done, info = env.step(0.0)
        assert done
        assert info['terminated_by_position']

    def test_done_by_max_steps(self):
        """max_steps に達したらエピソード終了"""
        env = InvertedPendulumEnv(max_steps=5, theta_threshold=10.0, x_threshold=100.0)
        env.reset(initial_state=[0.0, 0.0, 0.0, 0.0])
        for i in range(4):
            _, _, done, info = env.step(0.0)
            assert not done, f"Should not be done at step {i+1}"
        # 5ステップ目
        _, _, done, info = env.step(0.0)
        assert done
        assert info['terminated_by_time']

    def test_not_done_at_equilibrium(self):
        """平衡点付近では終了しない"""
        env = InvertedPendulumEnv(max_steps=500)
        env.reset(initial_state=[0.0, 0.0, 0.0, 0.0])
        _, _, done, _ = env.step(0.0)
        # 完全静止に近い状態では1ステップ目は終了しないはず
        assert not done


class TestPhysicsConsistency:
    """物理量の妥当性テスト"""

    def test_energy_returns_dict(self, env):
        env.reset()
        energy = env.get_energy()
        assert isinstance(energy, dict)
        assert set(energy.keys()) == {'kinetic', 'potential', 'total'}

    def test_energy_total_consistent(self, env):
        env.reset(initial_state=[0.0, 0.0, 0.0, 0.0])
        energy = env.get_energy()
        assert energy['total'] == pytest.approx(
            energy['kinetic'] + energy['potential'], rel=1e-6
        )

    def test_normalized_state_shape(self, env):
        env.reset()
        norm_state = env.get_normalized_state()
        assert norm_state.shape == (4,)

    def test_get_history_returns_arrays(self, env):
        env.reset()
        for _ in range(5):
            env.step(1.0)
        hist = env.get_history()
        assert isinstance(hist['states'], np.ndarray)
        assert isinstance(hist['actions'], np.ndarray)
        assert isinstance(hist['rewards'], np.ndarray)

    def test_get_history_length_consistent(self, env):
        env.reset()
        n_steps = 10
        for _ in range(n_steps):
            env.step(0.5)
        hist = env.get_history()
        # states: n_steps+1 (初期状態含む), actions/rewards: n_steps
        assert hist['states'].shape[0] == n_steps + 1
        assert hist['actions'].shape[0] == n_steps
        assert hist['rewards'].shape[0] == n_steps

    def test_custom_physics_params(self):
        """カスタム物理パラメータが正しく設定される"""
        env = InvertedPendulumEnv(
            gravity=1.62,       # 月面重力
            cart_mass=2.0,
            pole_mass=0.2,
            pole_length=1.0
        )
        assert env.gravity == pytest.approx(1.62)
        assert env.cart_mass == pytest.approx(2.0)
        assert env.pole_mass == pytest.approx(0.2)
        assert env.pole_length == pytest.approx(1.0)
        assert env.total_mass == pytest.approx(2.2)
