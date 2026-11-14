import numpy as np

class NumpyMLPPolicy:
    """
    Bộ suy luận mạng nơ-ron thuần Numpy (Zero-Dependency) tương thích hoàn toàn với
    kiến trúc actor của Stable-Baselines3 (PPO / MlpPolicy):
      - Tầng 1: Linear(input_dim -> 64) + Tanh
      - Tầng 2: Linear(64 -> 64) + Tanh
      - Tầng 3: Linear(64 -> 1) + Clip([-1.0, 1.0])
    Hỗ trợ vector quan sát 9D [s_t, s_{t-1}, a_{t-1}] hoặc 4D [s_t]
    và chế độ đối xứng lẻ odd_symmetric: a = 0.5 * (pi(obs) - pi(-obs)).
    """
    def __init__(self, input_dim=9, hidden_dims=(64, 64), output_dim=1, odd_symmetric=False):
        self.input_dim = input_dim
        self.h1_dim, self.h2_dim = hidden_dims
        self.output_dim = output_dim
        self.odd_symmetric = bool(odd_symmetric)

        # Khởi tạo ma trận trọng số (Xavier)
        self.w1 = np.random.randn(self.h1_dim, self.input_dim).astype(np.float32) * np.sqrt(2.0 / (input_dim + self.h1_dim))
        self.b1 = np.zeros(self.h1_dim, dtype=np.float32)

        self.w2 = np.random.randn(self.h2_dim, self.h1_dim).astype(np.float32) * np.sqrt(2.0 / (self.h1_dim + self.h2_dim))
        self.b2 = np.zeros(self.h2_dim, dtype=np.float32)

        self.w3 = np.random.randn(self.output_dim, self.h2_dim).astype(np.float32) * np.sqrt(2.0 / (self.h2_dim + self.output_dim))
        self.b3 = np.zeros(self.output_dim, dtype=np.float32)

    def _forward_unclipped_single(self, x):
        x_arr = np.asarray(x, dtype=np.float32).flatten()
        # Tầng ẩn 1: Linear + Tanh
        h1 = np.tanh(np.dot(self.w1, x_arr) + self.b1)
        # Tầng ẩn 2: Linear + Tanh
        h2 = np.tanh(np.dot(self.w2, h1) + self.b2)
        # Tầng đầu ra: Linear (chưa bão hòa)
        out = np.dot(self.w3, h2) + self.b3
        return float(out[0])

    def _forward_raw_single(self, x):
        return float(np.clip(self._forward_unclipped_single(x), -1.0, 1.0))

    def forward(self, obs):
        """
        Lan truyền tiến actor PPO: Tanh -> Tanh -> Linear -> Clip([-1.0, 1.0]).
        Khi odd_symmetric=True, tính 0.5 * (pi(obs) - pi(-obs)) để triệt tiêu lệch tâm tại obs = 0.
        """
        x = np.asarray(obs, dtype=np.float32).flatten()
        if self.odd_symmetric:
            return 0.5 * (self._forward_raw_single(x) - self._forward_raw_single(-x))
        return self._forward_raw_single(x)

    def load_from_sb3(self, model, odd_symmetric=None):
        """Trích xuất và kiểm tra tính toàn vẹn của trọng số từ mô hình Stable-Baselines3 PPO."""
        try:
            policy_net = model.policy.mlp_extractor.policy_net
            action_net = model.policy.action_net

            # Lớp Linear 1
            w1 = policy_net[0].weight.detach().cpu().numpy().copy()
            b1 = policy_net[0].bias.detach().cpu().numpy().copy()

            # Lớp Linear 2
            w2 = policy_net[2].weight.detach().cpu().numpy().copy()
            b2 = policy_net[2].bias.detach().cpu().numpy().copy()

            # Lớp Action Linear
            w3 = action_net.weight.detach().cpu().numpy().copy()
            b3 = action_net.bias.detach().cpu().numpy().copy()

            # Kiểm tra kích thước và tính hợp lệ (finite check)
            for name, arr in [("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2), ("w3", w3), ("b3", b3)]:
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"Trọng số {name} chứa giá trị NaN hoặc Inf!")

            if w1.shape[0] != 64 or w2.shape[0] != 64 or w3.shape[0] != 1:
                raise ValueError(f"Kích thước mạng không khớp: w1={w1.shape}, w2={w2.shape}, w3={w3.shape}")

            self.w1 = w1
            self.b1 = b1
            self.input_dim = self.w1.shape[1]
            self.w2 = w2
            self.b2 = b2
            self.w3 = w3
            self.b3 = b3
            if odd_symmetric is not None:
                self.odd_symmetric = bool(odd_symmetric)

            print(
                f"[Model] Trích xuất thành công: input_dim={self.input_dim}, "
                f"h1={self.w1.shape[0]}, h2={self.w2.shape[0]}, odd_symmetric={self.odd_symmetric}"
            )
            return True
        except Exception as e:
            print(f"[Error] Lỗi trích xuất SB3: {e}")
            return False

    def save_weights(self, path):
        np.savez(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            w3=self.w3,
            b3=self.b3,
            input_dim=self.input_dim,
            odd_symmetric=np.bool_(self.odd_symmetric),
        )

    def load_weights(self, path):
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            self.w1 = data['w1']
            self.b1 = data['b1']
            self.w2 = data['w2']
            self.b2 = data['b2']
            self.w3 = data['w3']
            self.b3 = data['b3']
            if 'input_dim' in data:
                self.input_dim = int(data['input_dim'])
            else:
                self.input_dim = self.w1.shape[1]
            self.odd_symmetric = bool(data['odd_symmetric']) if 'odd_symmetric' in data else False
        else:
            d = data.item()
            self.w1 = d['w1']
            self.b1 = d['b1']
            self.w2 = d['w2']
            self.b2 = d['b2']
            self.w3 = d['w3']
            self.b3 = d['b3']
            self.input_dim = self.w1.shape[1]
            self.odd_symmetric = bool(d.get('odd_symmetric', False))


try:
    import torch as _torch
    from stable_baselines3.common.policies import ActorCriticPolicy as _SB3ActorCriticPolicy

    class OddSymmetricActorCriticPolicy(_SB3ActorCriticPolicy):
        """
        Actor đối xứng lẻ cho SB3 PPO với vị trí chặn [-1, 1] từng nhánh trước khi lấy hiệu:
            a(obs) = 0.5 * (clamp(raw(obs), -1, 1) - clamp(raw(-obs), -1, 1))
        Đồng nhất 100% giữa PyTorch (SB3), NumPy và C ở cả vùng tuyến tính lẫn khi một nhánh chạm trần.
        Critic V(obs) giữ nguyên theo value_net chuẩn.
        """

        def _extract_pi_vf_features(self, obs):
            features = self.extract_features(obs)
            if isinstance(features, tuple):
                return features
            return features, features

        def _odd_sym_distribution(self, pi_features):
            raw_pos = _torch.clamp(self.action_net(self.mlp_extractor.forward_actor(pi_features)), -1.0, 1.0)
            raw_neg = _torch.clamp(self.action_net(self.mlp_extractor.forward_actor(-pi_features)), -1.0, 1.0)
            mean_actions = 0.5 * (raw_pos - raw_neg)
            return self.action_dist.proba_distribution(mean_actions, self.log_std)

        def forward(self, obs, deterministic=False):
            pi_features, vf_features = self._extract_pi_vf_features(obs)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            values = self.value_net(latent_vf)
            distribution = self._odd_sym_distribution(pi_features)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            actions = actions.reshape((-1, *self.action_space.shape))
            return actions, values, log_prob

        def get_distribution(self, obs):
            pi_features, _ = self._extract_pi_vf_features(obs)
            return self._odd_sym_distribution(pi_features)

        def evaluate_actions(self, obs, actions):
            pi_features, vf_features = self._extract_pi_vf_features(obs)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            distribution = self._odd_sym_distribution(pi_features)
            log_prob = distribution.log_prob(actions)
            values = self.value_net(latent_vf)
            entropy = distribution.entropy()
            return values, log_prob, entropy

except ImportError:
    OddSymmetricActorCriticPolicy = None
