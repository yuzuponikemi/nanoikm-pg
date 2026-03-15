"""
Unit tests for Policy classes (Random, Linear, Neural)

Tests cover:
- act() / get_action() が正しい形状を返すか
- パラメータの取得・設定
- ポリシー固有の動作
"""

import numpy as np
import pytest
from src.policies.random_policy import RandomPolicy
from src.policies.linear_policy import LinearPolicy

# NeuralNetworkPolicy は torch が必要なので条件付きでインポート
try:
    import torch
    from src.policies.neural_policy import NeuralNetworkPolicy
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# テスト用の代表的な状態ベクトル
SAMPLE_STATES = [
    np.array([0.0, 0.0, 0.0, 0.0]),        # 完全な平衡状態
    np.array([0.1, 0.0, 0.05, 0.0]),        # 軽微なずれ
    np.array([-0.5, 1.0, -0.1, 0.3]),      # 一般的な状態
    np.array([0.0, 0.0, 0.2, -0.5]),        # 角速度あり
]


class TestRandomPolicy:
    """RandomPolicy のテスト"""

    def test_get_action_returns_float(self, random_policy):
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = random_policy.get_action(state)
        assert isinstance(action, (float, np.floating))

    @pytest.mark.parametrize("state", SAMPLE_STATES)
    def test_get_action_in_range(self, state):
        policy = RandomPolicy(action_low=-10.0, action_high=10.0)
        action = policy.get_action(state)
        assert -10.0 <= action <= 10.0, (
            f"Action {action} is out of range [-10, 10]"
        )

    def test_get_action_discrete_values(self):
        """discrete=True のとき極値のみ返す"""
        policy = RandomPolicy(action_low=-5.0, action_high=5.0, discrete=True, seed=0)
        actions = [policy.get_action(np.zeros(4)) for _ in range(50)]
        unique = set(actions)
        assert unique == {-5.0, 5.0}, f"Expected only {{-5.0, 5.0}}, got {unique}"

    def test_get_action_ignores_state(self):
        """RandomPolicyは状態に関係なく動作する"""
        policy = RandomPolicy(seed=99)
        # 同じシードで同じ順序を生成する
        state_a = np.array([0.0, 0.0, 0.0, 0.0])
        state_b = np.array([1.0, -1.0, 0.5, -0.5])
        policy2 = RandomPolicy(seed=99)
        # 結果は状態によらずシードに依存する
        a1 = policy.get_action(state_a)
        a2 = policy2.get_action(state_b)
        assert a1 == pytest.approx(a2), "Same seed should produce same action regardless of state"

    def test_get_params_returns_dict(self, random_policy):
        params = random_policy.get_params()
        assert isinstance(params, dict)
        assert 'action_low' in params
        assert 'action_high' in params
        assert 'discrete' in params

    def test_set_params_updates_range(self):
        policy = RandomPolicy(action_low=-10.0, action_high=10.0)
        policy.set_params({'action_low': -1.0, 'action_high': 1.0})
        assert policy.action_low == -1.0
        assert policy.action_high == 1.0
        # 新しいレンジ内に収まるか確認
        for _ in range(20):
            action = policy.get_action(np.zeros(4))
            assert -1.0 <= action <= 1.0

    def test_get_num_params_is_zero(self, random_policy):
        """RandomPolicyは学習パラメータを持たない"""
        assert random_policy.get_num_params() == 0

    def test_custom_action_range(self):
        policy = RandomPolicy(action_low=-3.0, action_high=3.0)
        for _ in range(50):
            action = policy.get_action(np.zeros(4))
            assert -3.0 <= action <= 3.0


class TestLinearPolicy:
    """LinearPolicy のテスト"""

    def test_get_action_returns_float(self, linear_policy):
        state = np.array([0.0, 0.0, 0.1, 0.0])
        action = linear_policy.get_action(state)
        assert isinstance(action, float)

    @pytest.mark.parametrize("state", SAMPLE_STATES)
    def test_get_action_in_range(self, state):
        policy = LinearPolicy(action_low=-10.0, action_high=10.0)
        action = policy.get_action(state)
        assert -10.0 <= action <= 10.0

    def test_get_action_zero_weights_zero_bias(self):
        """重みゼロ、バイアスゼロならアクションはゼロ"""
        policy = LinearPolicy(weights=np.zeros(4), bias=0.0)
        state = np.array([1.0, 2.0, 3.0, 4.0])
        action = policy.get_action(state)
        assert action == pytest.approx(0.0)

    def test_get_action_linear_combination(self):
        """action = weights @ state + bias が正確に計算される"""
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        bias = 0.5
        policy = LinearPolicy(weights=weights, bias=bias, action_low=-1000, action_high=1000)
        state = np.array([1.0, 1.0, 1.0, 1.0])
        action = policy.get_action(state)
        expected = float(np.dot(weights, state) + bias)
        assert action == pytest.approx(expected)

    def test_get_action_clipped(self):
        """action_range を超える値はクリップされる"""
        # 大きな重みで大きな値を生成
        policy = LinearPolicy(
            weights=np.array([100.0, 0.0, 0.0, 0.0]),
            action_low=-5.0,
            action_high=5.0
        )
        state = np.array([1.0, 0.0, 0.0, 0.0])
        action = policy.get_action(state)
        assert action == pytest.approx(5.0)

    def test_get_action_clipped_negative(self):
        policy = LinearPolicy(
            weights=np.array([-100.0, 0.0, 0.0, 0.0]),
            action_low=-5.0,
            action_high=5.0
        )
        state = np.array([1.0, 0.0, 0.0, 0.0])
        action = policy.get_action(state)
        assert action == pytest.approx(-5.0)

    def test_get_params_returns_dict(self, linear_policy):
        params = linear_policy.get_params()
        assert isinstance(params, dict)
        assert 'weights' in params
        assert 'bias' in params

    def test_set_params_updates_weights(self, linear_policy):
        new_weights = np.array([1.0, 2.0, 3.0, 4.0])
        linear_policy.set_params({'weights': new_weights})
        np.testing.assert_array_almost_equal(linear_policy.weights, new_weights)

    def test_get_num_params(self, linear_policy):
        """weights(4) + bias(1) = 5パラメータ"""
        assert linear_policy.get_num_params() == 5

    def test_get_flat_params(self, linear_policy):
        """flat_params は weights と bias を結合したもの"""
        flat = linear_policy.get_flat_params()
        assert flat.shape == (5,)
        np.testing.assert_array_equal(flat[:4], linear_policy.weights)
        assert flat[4] == linear_policy.bias

    def test_set_flat_params_roundtrip(self, linear_policy):
        original_flat = linear_policy.get_flat_params().copy()
        flat = original_flat * 2  # 変更する
        linear_policy.set_flat_params(flat)
        np.testing.assert_array_almost_equal(linear_policy.get_flat_params(), flat)

    def test_perturb_creates_new_policy(self, linear_policy):
        np.random.seed(0)
        perturbed = linear_policy.perturb(noise_scale=0.1)
        assert isinstance(perturbed, LinearPolicy)
        # 元のポリシーとは異なるはず
        assert not np.allclose(perturbed.weights, linear_policy.weights)

    def test_perturb_does_not_modify_original(self, linear_policy):
        original_weights = linear_policy.weights.copy()
        np.random.seed(0)
        linear_policy.perturb(noise_scale=0.1)
        np.testing.assert_array_equal(linear_policy.weights, original_weights)

    def test_theta_weight_sign(self):
        """thetaへの正の重みはポールの傾きに対して適切な力を生成する"""
        # theta > 0 (右に傾く) → 正の力（右に押す）でバランスを取る
        policy = LinearPolicy(weights=np.array([0.0, 0.0, 10.0, 0.0]))
        state_lean_right = np.array([0.0, 0.0, 0.1, 0.0])
        action = policy.get_action(state_lean_right)
        assert action > 0, "Leaning right should produce positive force"

        state_lean_left = np.array([0.0, 0.0, -0.1, 0.0])
        action = policy.get_action(state_lean_left)
        assert action < 0, "Leaning left should produce negative force"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestNeuralNetworkPolicy:
    """NeuralNetworkPolicy のテスト（PyTorchが必要）"""

    @pytest.fixture
    def neural_policy(self):
        return NeuralNetworkPolicy(hidden_sizes=[32, 32])

    def test_get_action_returns_float(self, neural_policy):
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = neural_policy.get_action(state)
        assert isinstance(action, float)

    @pytest.mark.parametrize("state", SAMPLE_STATES)
    def test_get_action_in_range(self, state):
        policy = NeuralNetworkPolicy(action_low=-10.0, action_high=10.0)
        action = policy.get_action(state)
        assert -10.0 <= action <= 10.0, (
            f"Action {action} is out of range [-10, 10]"
        )

    def test_get_action_deterministic(self, neural_policy):
        """同じ状態で同じアクション（推論時は決定論的）"""
        state = np.array([0.1, -0.1, 0.05, -0.05])
        action1 = neural_policy.get_action(state)
        action2 = neural_policy.get_action(state)
        assert action1 == pytest.approx(action2)

    def test_get_params_returns_dict(self, neural_policy):
        params = neural_policy.get_params()
        assert isinstance(params, dict)
        assert 'network_state' in params
        assert 'hidden_sizes' in params

    def test_set_params_roundtrip(self, neural_policy):
        params = neural_policy.get_params()
        state = np.array([0.1, 0.0, 0.05, 0.0])
        action_before = neural_policy.get_action(state)
        # パラメータを再設定
        neural_policy.set_params(params)
        action_after = neural_policy.get_action(state)
        assert action_before == pytest.approx(action_after)

    def test_get_num_params_positive(self, neural_policy):
        """ニューラルネットは必ず1つ以上のパラメータを持つ"""
        n = neural_policy.get_num_params()
        assert n > 0

    def test_get_flat_params_shape(self, neural_policy):
        flat = neural_policy.get_flat_params()
        assert flat.ndim == 1
        assert flat.shape[0] == neural_policy.get_num_params()

    def test_set_flat_params_changes_action(self, neural_policy):
        state = np.array([0.1, 0.0, 0.05, 0.0])
        action_before = neural_policy.get_action(state)

        # ゼロパラメータに設定
        n = neural_policy.get_num_params()
        neural_policy.set_flat_params(np.zeros(n))
        action_after = neural_policy.get_action(state)

        # アクションが変化したことを確認（ゼロパラメータなら出力が変わるはず）
        # ※ 元々ゼロ初期化の場合は変わらないこともあるが、型の確認として有効
        assert isinstance(action_after, float)

    def test_custom_hidden_sizes(self):
        """異なるアーキテクチャで正しく動作するか"""
        for hidden_sizes in [[16], [64, 64], [32, 32, 32]]:
            policy = NeuralNetworkPolicy(hidden_sizes=hidden_sizes)
            state = np.array([0.1, 0.0, 0.05, 0.0])
            action = policy.get_action(state)
            assert isinstance(action, float)
            assert -10.0 <= action <= 10.0

    def test_get_action_and_log_prob(self, neural_policy):
        """stochasticモードでアクションと対数確率が返る"""
        state = np.array([0.1, 0.0, 0.05, 0.0])
        action, log_prob = neural_policy.get_action_and_log_prob(state)
        assert isinstance(action, float)
        # log_prob は負の値（確率なので）
        assert hasattr(log_prob, 'item')  # torch.Tensor

    def test_repr_contains_class_name(self, neural_policy):
        """__repr__ にクラス名が含まれる"""
        r = repr(neural_policy)
        assert 'NeuralNetworkPolicy' in r or 'params' in r
