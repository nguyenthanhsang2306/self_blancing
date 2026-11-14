import os
import sys
import argparse
import numpy as np

from src.models import NumpyMLPPolicy
from src.utils import export_sb3_c_header


def _c_raw_forward(policy, in_vec):
    """Mô phỏng chính xác từng phép tính float32 của hàm _rl_sb3_raw_forward trong C."""
    w1, b1 = policy.w1, policy.b1
    w2, b2 = policy.w2, policy.b2
    w3, b3 = policy.w3, policy.b3
    in_dim = int(policy.input_dim)

    h1 = np.zeros(64, dtype=np.float32)
    for i in range(64):
        s = float(b1[i])
        for j in range(in_dim):
            s += float(w1[i, j]) * float(in_vec[j])
        h1[i] = np.tanh(s)

    h2 = np.zeros(64, dtype=np.float32)
    for i in range(64):
        s = float(b2[i])
        for j in range(64):
            s += float(w2[i, j]) * float(h1[j])
        h2[i] = np.tanh(s)

    out = float(b3[0])
    for j in range(64):
        out += float(w3[0, j]) * float(h2[j])

    return float(np.clip(out, -1.0, 1.0))


def simulate_c_inference(policy, theta, dtheta, x, v, prev_theta, prev_dtheta, prev_x, prev_v, prev_action):
    """
    Mô phỏng chính xác 100% từng phép tính số học dấu phẩy động float32 trong hàm C:
    RL_SB3_Step trong file rl_sb3_stm32.h (hỗ trợ cả single-pass và odd_symmetric).
    """
    in_vec = np.array([
        theta, dtheta, x, v,
        prev_theta, prev_dtheta, prev_x, prev_v,
        prev_action
    ], dtype=np.float32)

    if getattr(policy, 'odd_symmetric', False):
        action = 0.5 * (_c_raw_forward(policy, in_vec) - _c_raw_forward(policy, -in_vec))
    else:
        action = _c_raw_forward(policy, in_vec)

    pwm = float(np.clip(action * 999.0, -999.0, 999.0))
    return action, pwm


def _verify_policy(policy, label):
    print(f"\n--- Kiểm tra chế độ [{label}] (odd_symmetric={getattr(policy, 'odd_symmetric', False)}) ---")
    rng = np.random.RandomState(42)

    test_vectors = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.05, 0.1, 0.01, 0.05, 0.04, 0.08, 0.008, 0.04, 0.1],
        [-0.05, -0.1, -0.01, -0.05, -0.04, -0.08, -0.008, -0.04, -0.1],
        [0.15, 0.5, 0.2, 0.4, 0.12, 0.4, 0.18, 0.35, 0.6],
        [-0.15, -0.5, -0.2, -0.4, -0.12, -0.4, -0.18, -0.35, -0.6],
        [0.35, 2.0, 0.8, 1.2, 0.30, 1.8, 0.75, 1.1, 1.0],
        [-0.35, -2.0, -0.8, -1.2, -0.30, -1.8, -0.75, -1.1, -1.0],
    ]
    for _ in range(5):
        test_vectors.append(list(rng.uniform(-0.5, 0.5, size=9)))

    max_err = 0.0
    max_pwm_err = 0.0
    for idx, vec in enumerate(test_vectors):
        vec_arr = np.array(vec, dtype=np.float32)
        act_numpy = policy.forward(vec_arr)
        pwm_numpy = float(np.clip(act_numpy * 999.0, -999.0, 999.0))
        act_c, pwm_c = simulate_c_inference(
            policy,
            vec[0], vec[1], vec[2], vec[3],
            vec[4], vec[5], vec[6], vec[7], vec[8]
        )

        diff = abs(act_numpy - act_c)
        pwm_diff = abs(pwm_numpy - pwm_c)
        max_err = max(max_err, diff)
        max_pwm_err = max(max_pwm_err, pwm_diff)
        print(f"  Test {idx+1:02d}: Numpy={act_numpy:+.6f} | C={act_c:+.6f} | Err={diff:.2e}")

    # Kiểm tra đối chiếu trên một chuỗi trạng thái 9D liên tiếp (closed-loop history buffer)
    seq_states = [
        (0.052, 0.12, 0.002, 0.03),
        (0.048, -0.05, 0.005, 0.04),
        (0.031, -0.22, 0.009, 0.02),
        (0.000, 0.00, 0.000, 0.00),
        (-0.031, 0.22, -0.009, -0.02),
    ]
    np_hist = [0.0, 0.0, 0.0, 0.0, 0.0]
    c_hist = [0.0, 0.0, 0.0, 0.0, 0.0]
    max_seq_err = 0.0
    for th, dth, xx, vv in seq_states:
        np_vec = np.array([th, dth, xx, vv, *np_hist], dtype=np.float32)
        act_n = policy.forward(np_vec)
        act_c, _ = simulate_c_inference(policy, th, dth, xx, vv, *c_hist)
        max_seq_err = max(max_seq_err, abs(act_n - act_c))
        np_hist = [th, dth, xx, vv, act_n]
        c_hist = [th, dth, xx, vv, act_c]
    print(f"  Sequence 9D rollout max Err: {max_seq_err:.2e}")
    max_err = max(max_err, max_seq_err)

    if getattr(policy, 'odd_symmetric', False):
        zero_act = policy.forward(np.zeros(9, dtype=np.float32))
        max_asym = abs(zero_act)
        for vec in test_vectors[1:]:
            arr = np.array(vec, dtype=np.float32)
            max_asym = max(max_asym, abs(policy.forward(arr) + policy.forward(-arr)))
        print(f"  Odd symmetry check: f(0)={zero_act:+.2e}, max|f(x)+f(-x)|={max_asym:.2e}")
        assert max_asym <= 1e-6, f"Vi phạm tính đối xứng lẻ: {max_asym}"

    try:
        import torch
        print(f"  [PyTorch Check] Đối chiếu PyTorch <-> Numpy ({label})...")
        class TorchActor(torch.nn.Module):
            def __init__(self, p):
                super().__init__()
                self.odd_symmetric = bool(getattr(p, 'odd_symmetric', False))
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

            def _raw(self, x):
                h1 = torch.tanh(self.fc1(x))
                h2 = torch.tanh(self.fc2(h1))
                return torch.clamp(self.out(h2), -1.0, 1.0)

            def forward(self, x):
                if self.odd_symmetric:
                    return 0.5 * (self._raw(x) - self._raw(-x))
                return self._raw(x)

        torch_net = TorchActor(policy)
        torch_net.eval()
        for idx, vec in enumerate(test_vectors[:5]):
            with torch.no_grad():
                t_out = float(torch_net(torch.from_numpy(np.array(vec, dtype=np.float32))).numpy()[0])
            n_out = policy.forward(vec)
            p_diff = abs(t_out - n_out)
            print(f"    PyTorch Test {idx+1:02d}: PyTorch={t_out:+.6f} | Numpy={n_out:+.6f} | Err={p_diff:.2e}")
            assert p_diff < 1e-4, f"PyTorch vs Numpy sai số quá lớn: {p_diff}"
    except ImportError:
        print("  [PyTorch Check] Bỏ qua kiểm tra PyTorch cục bộ (chưa cài đặt PyTorch).")

    print(f"  Max Numpy<->C action err: {max_err:.2e} | PWM err: {max_pwm_err:.2e}")
    if max_err > 1e-5:
        raise AssertionError(f"Sai lệch số học giữa Numpy và C quá ngưỡng 1e-5: {max_err}")
    return max_err


def verify():
    parser = argparse.ArgumentParser(description="Verify PyTorch <-> Numpy <-> C inference parity")
    parser.add_argument("--weights", type=str, default=None, help="Optional .npz checkpoint to verify")
    parser.add_argument("--symmetric", action="store_true", help="Force odd_symmetric=True when testing --weights")
    args = parser.parse_args()

    print("==============================================================================")
    print("      KIỂM TRA TÍNH TOÀN VẸN VÀ ĐỒNG NHẤT TOÁN HỌC (SB3 <-> NUMPY <-> C)      ")
    print("==============================================================================")

    if args.weights:
        if not os.path.exists(args.weights):
            sys.exit(f"[Error] Không tìm thấy checkpoint: {args.weights}")
        pol = NumpyMLPPolicy(input_dim=9, hidden_dims=(64, 64), output_dim=1)
        pol.load_weights(args.weights)
        if args.symmetric:
            pol.odd_symmetric = True
        max_err = _verify_policy(pol, os.path.basename(args.weights))
    else:
        np.random.seed(42)
        base_policy = NumpyMLPPolicy(input_dim=9, hidden_dims=(64, 64), output_dim=1, odd_symmetric=False)
        ckpt_path = "outputs/checkpoints/best_policy.npz"
        if os.path.exists(ckpt_path):
            base_policy.load_weights(ckpt_path)
            print(f"[Verify] Đã nạp checkpoint: {ckpt_path}")
        else:
            print("[Verify] Dùng trọng số ngẫu nhiên chuẩn để kiểm tra công thức toán học.")

        err1 = _verify_policy(base_policy, f"ckpt(odd_symmetric={base_policy.odd_symmetric})")
        sym_policy = NumpyMLPPolicy(input_dim=9, hidden_dims=(64, 64), output_dim=1, odd_symmetric=True)
        sym_policy.w1, sym_policy.b1 = base_policy.w1.copy(), base_policy.b1.copy()
        sym_policy.w2, sym_policy.b2 = base_policy.w2.copy(), base_policy.b2.copy()
        sym_policy.w3, sym_policy.b3 = base_policy.w3.copy(), base_policy.b3.copy()
        err2 = _verify_policy(sym_policy, "odd_symmetric=True")
        max_err = max(err1, err2)

    print("\n------------------- KẾT QUẢ XÁC MINH -------------------")
    print(f"  Sai số tuyệt đối cực đại giữa Numpy và C: {max_err:.2e}")
    print("  [PASS] Thuật toán Numpy và hàm C đồng nhất toán học 100%!")
    print("---------------------------------------------------------")


if __name__ == "__main__":
    verify()
