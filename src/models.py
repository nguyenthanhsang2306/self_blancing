import numpy as np

class NumpyMLPPolicy:
    """
    Bộ suy luận mạng nơ-ron thuần Numpy (Zero-Dependency) tương thích với kiến trúc
    net_arch=[64, 64] của Stable-Baselines3 (PPO / MlpPolicy):
      - Tầng 1: Linear(4 -> 64) + ReLU / Tanh
      - Tầng 2: Linear(64 -> 64) + ReLU / Tanh
      - Tầng 3: Linear(64 -> 1) + Tanh
    Dùng để mô phỏng, kiểm thử độ trễ và xuất trực tiếp sang code C cho STM32.
    """
    def __init__(self, input_dim=4, hidden_dims=(64, 64), output_dim=1):
        self.input_dim = input_dim
        self.h1_dim, self.h2_dim = hidden_dims
        self.output_dim = output_dim

        # Khởi tạo trọng số mặc định (Xavier / He)
        self.w1 = np.random.randn(self.h1_dim, self.input_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(self.h1_dim, dtype=np.float32)

        self.w2 = np.random.randn(self.h2_dim, self.h1_dim).astype(np.float32) * np.sqrt(2.0 / self.h1_dim)
        self.b2 = np.zeros(self.h2_dim, dtype=np.float32)

        self.w3 = np.random.randn(self.output_dim, self.h2_dim).astype(np.float32) * 0.1
        self.b3 = np.zeros(self.output_dim, dtype=np.float32)

    def forward(self, obs):
        """
        Thực hiện lan truyền tiến (Forward Pass) cho vector 4D: [theta, dtheta, x, v]
        Trả về: action trong [-1.0, 1.0]
        """
        x = np.asarray(obs, dtype=np.float32).flatten()
        # Tầng ẩn 1 (ReLU)
        h1 = np.maximum(0.0, np.dot(self.w1, x) + self.b1)
        # Tầng ẩn 2 (ReLU)
        h2 = np.maximum(0.0, np.dot(self.w2, h1) + self.b2)
        # Tầng đầu ra (Tanh để giới hạn mô-men trong [-1, 1])
        out = np.dot(self.w3, h2) + self.b3
        action = float(np.tanh(out[0]))
        return action

    def load_from_sb3(self, model):
        """Trích xuất trực tiếp trọng số từ mô hình Stable-Baselines3 PPO."""
        try:
            policy_net = model.policy.mlp_extractor.policy_net
            action_net = model.policy.action_net

            # Lớp Linear 1
            self.w1 = policy_net[0].weight.detach().cpu().numpy().copy()
            self.b1 = policy_net[0].bias.detach().cpu().numpy().copy()

            # Lớp Linear 2
            self.w2 = policy_net[2].weight.detach().cpu().numpy().copy()
            self.b2 = policy_net[2].bias.detach().cpu().numpy().copy()

            # Lớp Action Linear
            self.w3 = action_net.weight.detach().cpu().numpy().copy()
            self.b3 = action_net.bias.detach().cpu().numpy().copy()
            return True
        except Exception as e:
            print(f"[Warning] Không thể trích xuất tự động từ SB3: {e}")
            return False

    def save_weights(self, path):
        """Lưu toàn bộ ma trận trọng số vào file .npz"""
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, w3=self.w3, b3=self.b3)

    def load_weights(self, path):
        """Nạp ma trận trọng số từ file .npz hoặc .npy"""
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            self.w1 = data['w1']
            self.b1 = data['b1']
            self.w2 = data['w2']
            self.b2 = data['b2']
            self.w3 = data['w3']
            self.b3 = data['b3']
        else:
            d = data.item()
            self.w1 = d['w1']
            self.b1 = d['b1']
            self.w2 = d['w2']
            self.b2 = d['b2']
            self.w3 = d['w3']
            self.b3 = d['b3']
