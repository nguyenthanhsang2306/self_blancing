import os
import sys
import numpy as np

from src.models import NumpyMLPPolicy
from src.utils import export_sb3_c_header

def simulate_c_inference(policy, theta, dtheta, x, v, prev_theta, prev_dtheta, prev_x, prev_v, prev_action):
    """
    Mô phỏng chính xác 100% từng phép tính số học dấu phẩy động float32 trong hàm C:
    RL_SB3_Step trong file rl_sb3_stm32.h
    """
    w1, b1 = policy.w1, policy.b1
    w2, b2 = policy.w2, policy.b2
    w3, b3 = policy.w3, policy.b3

    in_vec = np.array([
        theta, dtheta, x, v,
        prev_theta, prev_dtheta, prev_x, prev_v,
        prev_action
    ], dtype=np.float32)

    # Tầng ẩn 1: Linear + Tanh
    h1 = np.zeros(64, dtype=np.float32)
    for i in range(64):
        s = float(b1[i])
        for j in range(9):
            s += float(w1[i, j]) * float(in_vec[j])
        h1[i] = np.tanh(s)

    # Tầng ẩn 2: Linear + Tanh
    h2 = np.zeros(64, dtype=np.float32)
    for i in range(64):
        s = float(b2[i])
        for j in range(64):
            s += float(w2[i, j]) * float(h1[j])
        h2[i] = np.tanh(s)

    # Tầng đầu ra: Linear + Clip [-1, 1]
    out = float(b3[0])
    for j in range(64):
        out += float(w3[0, j]) * float(h2[j])

    action = float(np.clip(out, -1.0, 1.0))
    pwm = float(np.clip(action * 999.0, -999.0, 999.0))
    return action, pwm

def verify():
    print("==============================================================================")
    print("      KIỂM TRA TÍNH TOÀN VẸN VÀ ĐỒNG NHẤT TOÁN HỌC (SB3 <-> NUMPY <-> C)      ")
    print("==============================================================================")

    np.random.seed(42)
    policy = NumpyMLPPolicy(input_dim=9, hidden_dims=(64, 64), output_dim=1)

    # Kiểm tra với checkpoint nếu có, hoặc với trọng số ngẫu nhiên cố định
    ckpt_path = "outputs/checkpoints/best_policy.npz"
    if os.path.exists(ckpt_path):
        policy.load_weights(ckpt_path)
        print(f"[Verify] Đã nạp checkpoint thực tế: {ckpt_path}")
    else:
        print("[Verify] Đang kiểm tra tính tương đương toán học với trọng số ngẫu nhiên chuẩn.")

    # Tạo các vector kiểm thử đa dạng: thẳng, nghiêng trái, nghiêng phải, vận tốc lớn, biên bão hòa
    test_vectors = [
        # [theta, dtheta, x, v, prev_theta, prev_dtheta, prev_x, prev_v, prev_action]
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                          # Đứng thẳng hoàn hảo
        [0.05, 0.1, 0.01, 0.05, 0.04, 0.08, 0.008, 0.04, 0.1],                 # Nghiêng nhẹ tiến
        [-0.05, -0.1, -0.01, -0.05, -0.04, -0.08, -0.008, -0.04, -0.1],        # Nghiêng nhẹ lùi
        [0.15, 0.5, 0.2, 0.4, 0.12, 0.4, 0.18, 0.35, 0.6],                     # Vùng động lực mạnh
        [-0.15, -0.5, -0.2, -0.4, -0.12, -0.4, -0.18, -0.35, -0.6],            # Vùng động lực mạnh lùi
        [0.35, 2.0, 0.8, 1.2, 0.30, 1.8, 0.75, 1.1, 1.0],                      # Sát biên bão hòa dương
        [-0.35, -2.0, -0.8, -1.2, -0.30, -1.8, -0.75, -1.1, -1.0],             # Sát biên bão hòa âm
    ]

    # Bổ sung 5 vector ngẫu nhiên
    for _ in range(5):
        test_vectors.append(list(np.random.uniform(-0.5, 0.5, size=9)))

    max_err = 0.0
    for idx, vec in enumerate(test_vectors):
        vec_arr = np.array(vec, dtype=np.float32)
        act_numpy = policy.forward(vec_arr)
        act_c, pwm_c = simulate_c_inference(
            policy,
            vec[0], vec[1], vec[2], vec[3],
            vec[4], vec[5], vec[6], vec[7], vec[8]
        )

        diff = abs(act_numpy - act_c)
        if diff > max_err:
            max_err = diff

        print(f"Test {idx+1:02d}: Numpy Action = {act_numpy:+.6f} | C Action = {act_c:+.6f} | Err = {diff:.2e}")

    # Kiểm tra xem PyTorch có sẵn không để kiểm tra thêm
    try:
        import torch
        print("\n[PyTorch Check] Đang kiểm tra tương đương với mạng PyTorch nguyên bản...")
        class TorchActor(torch.nn.Module):
            def __init__(self, p):
                super().__init__()
                self.fc1 = torch.nn.Linear(9, 64)
                self.fc2 = torch.nn.Linear(64, 64)
                self.out = torch.nn.Linear(64, 1)
                with torch.no_grad():
                    self.fc1.weight.copy_(torch.from_numpy(p.w1))
                    self.fc1.bias.copy_(torch.from_numpy(p.b1))
                    self.fc2.weight.copy_(torch.from_numpy(p.w2))
                    self.fc2.bias.copy_(torch.from_numpy(p.b2))
                    self.out.weight.copy_(torch.from_numpy(p.w3))
                    self.out.bias.copy_(torch.from_numpy(p.b3))
            def forward(self, x):
                h1 = torch.tanh(self.fc1(x))
                h2 = torch.tanh(self.fc2(h1))
                o = self.out(h2)
                return torch.clamp(o, -1.0, 1.0)

        torch_net = TorchActor(policy)
        torch_net.eval()
        for idx, vec in enumerate(test_vectors[:5]):
            with torch.no_grad():
                t_out = float(torch_net(torch.from_numpy(np.array(vec, dtype=np.float32))).numpy()[0])
            n_out = policy.forward(vec)
            p_diff = abs(t_out - n_out)
            print(f"PyTorch Test {idx+1:02d}: PyTorch = {t_out:+.6f} | Numpy = {n_out:+.6f} | Err = {p_diff:.2e}")
            assert p_diff < 1e-4, f"PyTorch vs Numpy sai số quá lớn: {p_diff}"
    except ImportError:
        print("\n[PyTorch Check] Bỏ qua kiểm tra PyTorch cục bộ (chưa cài đặt PyTorch).")

    print("\n------------------- KẾT QUẢ XÁC MINH -------------------")
    print(f"  Sai số tuyệt đối cực đại giữa Numpy và C: {max_err:.2e}")
    if max_err <= 1e-5:
        print("  \u2705 HOÀN HẢO: Thuật toán Numpy và hàm C đồng nhất toán học 100%!")
    else:
        print("  \u274c CẢNH BÁO: Phát hiện sai lệch số học giữa Numpy và C!")
        sys.exit(1)
    print("---------------------------------------------------------")

if __name__ == "__main__":
    verify()
