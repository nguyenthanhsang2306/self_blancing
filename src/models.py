import numpy as np

class NumpyMLPPolicy:
    """
    Bộ suy luận mạng nơ-ron thuần Numpy (Zero-Dependency) tương thích hoàn toàn với
    kiến trúc actor của Stable-Baselines3 (PPO / MlpPolicy):
      - Tầng 1: Linear(input_dim -> 64) + Tanh
      - Tầng 2: Linear(64 -> 64) + Tanh
      - Tầng 3: Linear(64 -> 1) + Clip([-1.0, 1.0])
    Hỗ trợ vector quan sát 9D [s_t, s_{t-1}, a_{t-1}] hoặc 4D [s_t].
    """
    def __init__(self, input_dim=9, hidden_dims=(64, 64), output_dim=1):
        self.input_dim = input_dim
        self.h1_dim, self.h2_dim = hidden_dims
        self.output_dim = output_dim

        # Khởi tạo ma trận trọng số (Xavier)
        self.w1 = np.random.randn(self.h1_dim, self.input_dim).astype(np.float32) * np.sqrt(2.0 / (input_dim + self.h1_dim))
        self.b1 = np.zeros(self.h1_dim, dtype=np.float32)

        self.w2 = np.random.randn(self.h2_dim, self.h1_dim).astype(np.float32) * np.sqrt(2.0 / (self.h1_dim + self.h2_dim))
        self.b2 = np.zeros(self.h2_dim, dtype=np.float32)

        self.w3 = np.random.randn(self.output_dim, self.h2_dim).astype(np.float32) * np.sqrt(2.0 / (self.h2_dim + self.output_dim))
        self.b3 = np.zeros(self.output_dim, dtype=np.float32)

    def forward(self, obs):
        """
        Lan truyền tiến actor PPO: Tanh -> Tanh -> Linear -> Clip([-1.0, 1.0]).
        Đồng nhất 100% với model.predict(obs, deterministic=True) của Stable-Baselines3.
        """
        x = np.asarray(obs, dtype=np.float32).flatten()
        # Tầng ẩn 1: Linear + Tanh
        h1 = np.tanh(np.dot(self.w1, x) + self.b1)
        # Tầng ẩn 2: Linear + Tanh
        h2 = np.tanh(np.dot(self.w2, h1) + self.b2)
        # Tầng đầu ra: Linear + Bão hòa [-1.0, 1.0]
        out = np.dot(self.w3, h2) + self.b3
        action = float(np.clip(out[0], -1.0, 1.0))
        return action

    def load_from_sb3(self, model):
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

            print(f"[Model] Trích xuất thành công: input_dim={self.input_dim}, h1={self.w1.shape[0]}, h2={self.w2.shape[0]}")
            return True
        except Exception as e:
            print(f"[Error] Lỗi trích xuất SB3: {e}")
            return False

    def save_weights(self, path):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, w3=self.w3, b3=self.b3, input_dim=self.input_dim)

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
        else:
            d = data.item()
            self.w1 = d['w1']
            self.b1 = d['b1']
            self.w2 = d['w2']
            self.b2 = d['b2']
            self.w3 = d['w3']
            self.b3 = d['b3']
            self.input_dim = self.w1.shape[1]
