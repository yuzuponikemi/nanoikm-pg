"""
Unit tests for training utilities

Tests cover:
- collect_episode() が正しく動作するか
- evaluate_policy() が収束指標を返すか
- train_policy() が収束指標を返すか
"""

import numpy as np
import pytest
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


@pytest.fixture
def simple_env():
    """テスト用の短いエピソードの環境"""
    return InvertedPendulumEnv(max_steps=50)


@pytest.fixture
def random_policy():
    return RandomPolicy(seed=0)


@pytest.fixture
def linear_policy():
    return LinearPolicy()


class TestCollectEpisode:
    """collect_episode() のテスト"""

    def test_returns_tuple_of_four(self, simple_env, random_policy):
        result = collect_episode(simple_env, random_policy)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_states_is_list(self, simple_env, random_policy):
        states, actions, rewards, info = collect_episode(simple_env, random_policy)
        assert isinstance(states, list)

    def test_actions_is_list(self, simple_env, random_policy):
        states, actions, rewards, info = collect_episode(simple_env, random_policy)
        assert isinstance(actions, list)

    def test_rewards_is_list(self, simple_env, random_policy):
        states, actions, rewards, info = collect_episode(simple_env, random_policy)
        assert isinstance(rewards, list)

    def test_info_is_dict(self, simple_env, random_policy):
        states, actions, rewards, info = collect_episode(simple_env, random_policy)
        assert isinstance(info, dict)

    def test_states_shape_consistent(self, simple_env, random_policy):
        states, actions, rewards, _ = collect_episode(simple_env, random_policy)
        # states はアクション数+1（初期状態含む）
        assert len(states) == len(actions) + 1

    def test_rewards_length_equals_actions(self, simple_env, random_policy):
        _, actions, rewards, _ = collect_episode(simple_env, random_policy)
        assert len(rewards) == len(actions)

    def test_each_state_is_4d(self, simple_env, random_policy):
        states, _, _, _ = collect_episode(simple_env, random_policy)
        for state in states:
            assert state.shape == (4,)

    def test_rewards_are_non_negative(self, simple_env, random_policy):
        _, _, rewards, _ = collect_episode(simple_env, random_policy)
        for r in rewards:
            assert r >= 0.0

    def test_episode_ends_within_max_steps(self, simple_env, random_policy):
        _, actions, _, _ = collect_episode(simple_env, random_policy)
        assert len(actions) <= simple_env.max_steps

    def test_at_least_one_step(self, simple_env, random_policy):
        """エピソードは少なくとも1ステップ以上ある"""
        _, actions, _, _ = collect_episode(simple_env, random_policy)
        assert len(actions) >= 1

    def test_custom_max_steps_honored(self, simple_env, random_policy):
        """custom max_steps が守られる"""
        _, actions, _, _ = collect_episode(
            simple_env, random_policy, max_steps=5
        )
        assert len(actions) <= 5


class TestEvaluatePolicy:
    """evaluate_policy() のテスト"""

    def test_returns_dict(self, simple_env, random_policy):
        result = evaluate_policy(simple_env, random_policy, n_episodes=3)
        assert isinstance(result, dict)

    def test_required_keys_present(self, simple_env, random_policy):
        result = evaluate_policy(simple_env, random_policy, n_episodes=3)
        required_keys = {
            'mean_reward', 'std_reward', 'mean_length',
            'episode_rewards', 'episode_lengths'
        }
        assert required_keys.issubset(result.keys())

    def test_mean_reward_is_float(self, simple_env, random_policy):
        result = evaluate_policy(simple_env, random_policy, n_episodes=3)
        assert isinstance(result['mean_reward'], float)

    def test_std_reward_is_non_negative(self, simple_env, random_policy):
        result = evaluate_policy(simple_env, random_policy, n_episodes=3)
        assert result['std_reward'] >= 0.0

    def test_mean_length_is_positive(self, simple_env, random_policy):
        result = evaluate_policy(simple_env, random_policy, n_episodes=3)
        assert result['mean_length'] > 0

    def test_episode_rewards_length(self, simple_env, random_policy):
        n = 5
        result = evaluate_policy(simple_env, random_policy, n_episodes=n)
        assert len(result['episode_rewards']) == n

    def test_episode_lengths_length(self, simple_env, random_policy):
        n = 5
        result = evaluate_policy(simple_env, random_policy, n_episodes=n)
        assert len(result['episode_lengths']) == n

    def test_mean_reward_matches_episodes(self, simple_env, random_policy):
        result = evaluate_policy(simple_env, random_policy, n_episodes=10, seed=42)
        expected_mean = np.mean(result['episode_rewards'])
        assert result['mean_reward'] == pytest.approx(expected_mean)

    def test_reproducible_with_seed(self, simple_env, random_policy):
        r1 = evaluate_policy(simple_env, random_policy, n_episodes=5, seed=99)
        r2 = evaluate_policy(simple_env, random_policy, n_episodes=5, seed=99)
        assert r1['mean_reward'] == pytest.approx(r2['mean_reward'])

    def test_rewards_are_non_negative(self, simple_env, random_policy):
        result = evaluate_policy(simple_env, random_policy, n_episodes=5)
        for r in result['episode_rewards']:
            assert r >= 0

    def test_good_policy_scores_higher_than_random(self, simple_env):
        """良い方策はランダム方策より高スコア（確率的だが概ね成立）"""
        # theta と theta_dot に大きな重みを持つ調整済みポリシー
        good_policy = LinearPolicy(weights=np.array([0.0, 0.0, 50.0, 5.0]))
        random_pol = RandomPolicy(seed=0)

        good_result = evaluate_policy(simple_env, good_policy, n_episodes=10, seed=42)
        rand_result = evaluate_policy(simple_env, random_pol, n_episodes=10, seed=42)

        # 良いポリシーの方が平均スコアが高いはず
        assert good_result['mean_reward'] >= rand_result['mean_reward'], (
            f"Good policy ({good_result['mean_reward']:.1f}) should outperform "
            f"random ({rand_result['mean_reward']:.1f})"
        )


class TestTrainPolicy:
    """train_policy() のテスト"""

    @pytest.fixture
    def quick_train_env(self):
        """短いエピソードで素早くテストするための環境"""
        return InvertedPendulumEnv(max_steps=30)

    def test_train_returns_dict(self, quick_train_env, linear_policy):
        result = train_policy(
            quick_train_env, linear_policy,
            algorithm='hill_climbing',
            n_iterations=3,
            n_episodes_per_eval=2,
            verbose=False
        )
        assert isinstance(result, dict)

    def test_train_returns_required_keys(self, quick_train_env, linear_policy):
        result = train_policy(
            quick_train_env, linear_policy,
            algorithm='hill_climbing',
            n_iterations=3,
            n_episodes_per_eval=2,
            verbose=False
        )
        required_keys = {'best_params', 'best_reward', 'reward_history'}
        assert required_keys.issubset(result.keys())

    def test_best_params_is_array(self, quick_train_env, linear_policy):
        result = train_policy(
            quick_train_env, linear_policy,
            algorithm='hill_climbing',
            n_iterations=3,
            n_episodes_per_eval=2,
            verbose=False
        )
        assert isinstance(result['best_params'], np.ndarray)

    def test_best_reward_is_number(self, quick_train_env, linear_policy):
        result = train_policy(
            quick_train_env, linear_policy,
            algorithm='hill_climbing',
            n_iterations=3,
            n_episodes_per_eval=2,
            verbose=False
        )
        assert isinstance(result['best_reward'], (int, float, np.floating))

    def test_reward_history_length(self, quick_train_env, linear_policy):
        n_iter = 5
        result = train_policy(
            quick_train_env, linear_policy,
            algorithm='hill_climbing',
            n_iterations=n_iter,
            n_episodes_per_eval=2,
            verbose=False
        )
        assert len(result['reward_history']) == n_iter + 1  # 初期評価含む

    def test_hill_climbing_algorithm(self, quick_train_env, linear_policy):
        result = train_policy(
            quick_train_env, linear_policy,
            algorithm='hill_climbing',
            n_iterations=3,
            n_episodes_per_eval=2,
            verbose=False
        )
        assert result['best_reward'] >= 0

    def test_random_search_algorithm(self, quick_train_env, linear_policy):
        result = train_policy(
            quick_train_env, linear_policy,
            algorithm='random_search',
            n_iterations=3,
            n_episodes_per_eval=2,
            verbose=False
        )
        assert 'best_reward' in result

    def test_evolutionary_algorithm(self, quick_train_env, linear_policy):
        result = train_policy(
            quick_train_env, linear_policy,
            algorithm='evolutionary',
            n_iterations=2,
            population_size=5,
            n_episodes_per_eval=2,
            verbose=False
        )
        assert 'best_reward' in result

    def test_unknown_algorithm_raises(self, quick_train_env, linear_policy):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            train_policy(
                quick_train_env, linear_policy,
                algorithm='unknown_algo',
                n_iterations=1,
                verbose=False
            )

    def test_policy_without_flat_params_raises(self, quick_train_env):
        """get_flat_params を持たないポリシーはエラー"""
        bad_policy = RandomPolicy()  # RandomPolicyにはget_flat_paramsがない
        with pytest.raises(ValueError, match="get_flat_params"):
            train_policy(
                quick_train_env, bad_policy,
                algorithm='hill_climbing',
                n_iterations=1,
                verbose=False
            )

    def test_best_params_shape_matches_policy(self, quick_train_env, linear_policy):
        """best_params の形状がポリシーパラメータと一致する"""
        expected_shape = linear_policy.get_flat_params().shape
        result = train_policy(
            quick_train_env, linear_policy,
            algorithm='hill_climbing',
            n_iterations=3,
            n_episodes_per_eval=2,
            verbose=False
        )
        assert result['best_params'].shape == expected_shape

    def test_reward_history_non_decreasing_hill_climbing(self, quick_train_env, linear_policy):
        """ヒルクライミングのreward_historyは単調非減少（ベスト追跡）"""
        result = train_policy(
            quick_train_env, linear_policy,
            algorithm='hill_climbing',
            n_iterations=10,
            n_episodes_per_eval=2,
            verbose=False,
            seed=42
        )
        history = result['reward_history']
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 1e-9, (
                f"Hill climbing history should be non-decreasing, "
                f"got {history[i-1]:.2f} -> {history[i]:.2f} at step {i}"
            )


class TestComputeReturns:
    """compute_returns() のテスト"""

    def test_single_reward(self):
        rewards = [1.0]
        returns = compute_returns(rewards, gamma=0.99)
        assert returns.shape == (1,)
        assert returns[0] == pytest.approx(1.0)

    def test_multiple_rewards_no_discount(self):
        rewards = [1.0, 1.0, 1.0]
        returns = compute_returns(rewards, gamma=1.0)
        # G_0 = 3, G_1 = 2, G_2 = 1
        np.testing.assert_allclose(returns, [3.0, 2.0, 1.0])

    def test_multiple_rewards_with_discount(self):
        rewards = [1.0, 0.0, 0.0]
        returns = compute_returns(rewards, gamma=0.9)
        # G_0 = 1, G_1 = 0, G_2 = 0
        assert returns[0] == pytest.approx(1.0)
        assert returns[1] == pytest.approx(0.0)
        assert returns[2] == pytest.approx(0.0)

    def test_returns_shape(self):
        rewards = [1.0] * 10
        returns = compute_returns(rewards, gamma=0.99)
        assert returns.shape == (10,)

    def test_returns_are_non_negative_for_non_negative_rewards(self):
        rewards = [1.0] * 5
        returns = compute_returns(rewards, gamma=0.99)
        assert np.all(returns >= 0)

    def test_discount_reduces_later_rewards(self):
        """割引により遠い報酬ほど価値が低くなる"""
        rewards = [0.0, 0.0, 1.0]
        returns = compute_returns(rewards, gamma=0.9)
        # G_0 = 0.9^2 * 1.0 = 0.81
        assert returns[0] == pytest.approx(0.81, rel=1e-5)

    def test_gamma_zero(self):
        """gamma=0 は即時報酬のみ"""
        rewards = [1.0, 2.0, 3.0]
        returns = compute_returns(rewards, gamma=0.0)
        np.testing.assert_allclose(returns, [1.0, 2.0, 3.0])


class TestNormalizeReturns:
    """normalize_returns() のテスト"""

    def test_normalized_mean_near_zero(self):
        returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_returns(returns)
        assert abs(np.mean(normalized)) < 1e-10

    def test_normalized_std_near_one(self):
        returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_returns(returns)
        assert abs(np.std(normalized) - 1.0) < 1e-10

    def test_constant_returns_handled(self):
        """全て同じ値の場合でもエラーにならない"""
        returns = np.array([5.0, 5.0, 5.0])
        normalized = normalize_returns(returns)
        # std=0 の場合は mean を引くだけ
        np.testing.assert_allclose(normalized, [0.0, 0.0, 0.0])

    def test_shape_preserved(self):
        returns = np.array([1.0, 2.0, 3.0])
        normalized = normalize_returns(returns)
        assert normalized.shape == returns.shape
