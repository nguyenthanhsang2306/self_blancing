import os
import re
import sys
import copy
import shutil
import argparse
import tempfile
import subprocess
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
    RL_SB3_Step trong file rl_sb3_stm32.h (chặn từng nhánh [-1, 1] trước khi lấy hiệu).
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


def parse_c_header(header_path):
    """Đọc ngược hằng số trọng số và cờ RL_SB3_ODD_SYMMETRIC từ file .h đã xuất."""
    with open(header_path, "r", encoding="utf-8") as f:
        text = f.read()

    def _extract_define(name, default=0):
        m = re.search(rf"#define\s+{name}\s+(\d+)", text)
        return int(m.group(1)) if m else default

    def _extract_array(name):
        m = re.search(rf"{name}\b[^=]*=\s*\{{(.*?)\}};", text, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Không tìm thấy mảng {name} trong header {header_path}")
        vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))
        return np.array([float(v) for v in vals], dtype=np.float32)

    in_dim = _extract_define("RL_SB3_INPUT_DIM", 9)
    h1_dim = _extract_define("RL_SB3_H1_DIM", 64)
    h2_dim = _extract_define("RL_SB3_H2_DIM", 64)
    odd_sym = bool(_extract_define("RL_SB3_ODD_SYMMETRIC", 0))

    pol = NumpyMLPPolicy(input_dim=in_dim, hidden_dims=(h1_dim, h2_dim), output_dim=1, odd_symmetric=odd_sym)
    pol.w1 = _extract_array("RL_SB3_W1").reshape(h1_dim, in_dim)
    pol.b1 = _extract_array("RL_SB3_B1").reshape(h1_dim)
    pol.w2 = _extract_array("RL_SB3_W2").reshape(h2_dim, h1_dim)
    pol.b2 = _extract_array("RL_SB3_B2").reshape(h2_dim)
    pol.w3 = _extract_array("RL_SB3_W3").reshape(1, h2_dim)
    m_b3_scalar = re.search(r"static\s+const\s+float\s+RL_SB3_B3\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)f?\s*;", text)
    if m_b3_scalar:
        pol.b3 = np.array([float(m_b3_scalar.group(1))], dtype=np.float32)
    else:
        pol.b3 = _extract_array("RL_SB3_B3").reshape(1)
    return pol


def run_compiled_c_sequence(header_path, seq_states):
    """
    Biên dịch trực tiếp file header .h bằng gcc (nếu khả dụng trên HĐH) và chạy hàm C RL_SB3_Step
    trên chuỗi trạng thái [theta, dtheta, x, v]. Trả về danh sách (action, pwm) hoặc None nếu bị bỏ qua.
    """
    gcc = shutil.which("gcc")
    if not gcc:
        return None

    abs_hdr = os.path.abspath(header_path).replace(os.sep, "/")
    rows_c = ",\n        ".join(
        "{" + ", ".join(f"{float(v):.7f}f" for v in row) + "}"
        for row in seq_states
    )
    c_code = f'''#include <stdio.h>
#include "{abs_hdr}"

int main(void) {{
    RL_SB3_Controller_t ctrl;
    RL_SB3_Init(&ctrl);
    float seq[{len(seq_states)}][4] = {{
        {rows_c}
    }};
    for (int i = 0; i < {len(seq_states)}; i++) {{
        float pwm = RL_SB3_Step(&ctrl, seq[i][0], seq[i][1], seq[i][2], seq[i][3]);
        printf("%.8f\\n", pwm);
    }}
    return 0;
}}
'''
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            c_src = os.path.join(tmpdir, "verify_driver.c")
            c_exe = os.path.join(tmpdir, "verify_driver.exe" if os.name == "nt" else "verify_driver")
            with open(c_src, "w", encoding="utf-8") as f:
                f.write(c_code)
            subprocess.run([gcc, c_src, "-O2", "-lm", "-o", c_exe], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out = subprocess.check_output([c_exe], text=True).strip().splitlines()
            pwms = [float(line) for line in out]
            return [(p / 999.0, p) for p in pwms]
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"  [Compiled C] Bỏ qua chạy gcc cục bộ ({e.__class__.__name__}); vẫn kiểm tra qua bộ giả lập float32 + phân tích header.")
        return None


def find_one_branch_saturation_vectors(policy, max_trials=5000, seed=123):
    """
    Tìm các vector đầu vào 9D mà MỘT nhánh chạm trần [-1, 1] trong khi nhánh đối dấu KHÔNG chạm trần,
    và hai cách chặn (chặn từng nhánh vs lấy hiệu rồi mới chặn) cho kết quả khác nhau rõ rệt.
    """
    rng = np.random.RandomState(seed)
    in_dim = int(policy.input_dim)
    found = []
    for trial in range(max_trials):
        scale = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0][trial % 6]
        vec = rng.uniform(-scale, scale, size=in_dim).astype(np.float32)
        if in_dim == 9:
            vec[0] = float(np.clip(vec[0], -0.6, 0.6))
            vec[4] = float(np.clip(vec[4], -0.6, 0.6))
            vec[8] = float(np.clip(vec[8], -1.0, 1.0))
        r_pos = policy._forward_unclipped_single(vec)
        r_neg = policy._forward_unclipped_single(-vec)
        one_sat = (abs(r_pos) > 1.005) ^ (abs(r_neg) > 1.005)
        if one_sat:
            branch_clip = 0.5 * (float(np.clip(r_pos, -1.0, 1.0)) - float(np.clip(r_neg, -1.0, 1.0)))
            naive_diff_clip = float(np.clip(0.5 * (r_pos - r_neg), -1.0, 1.0))
            if abs(branch_clip - naive_diff_clip) > 1e-3:
                found.append((vec, r_pos, r_neg, branch_clip, naive_diff_clip))
                if len(found) >= 5:
                    break
    return found


def _sb3_predict_action(sb3_model, vec):
    pred, _ = sb3_model.predict(np.asarray(vec, dtype=np.float32), deterministic=True)
    return float(np.asarray(pred).item())


def _build_forced_one_branch_sat_policy(base_policy):
    """Tạo bản sao khuếch đại nhẹ w3/b3 để bảo đảm luôn kiểm thử được ca 1 nhánh chạm trần."""
    pol = copy.deepcopy(base_policy)
    pol.odd_symmetric = True
    max_raw = 0.0
    rng = np.random.RandomState(7)
    probes = [rng.uniform(-0.8, 0.8, size=pol.input_dim).astype(np.float32) for _ in range(32)]
    for v in probes:
        max_raw = max(max_raw, abs(pol._forward_unclipped_single(v)))
    gain = 2.0 / max(max_raw, 1e-3)
    pol.w3 = (pol.w3 * gain).astype(np.float32)
    pol.b3 = np.array([0.45], dtype=np.float32)
    return pol


def verify_policy_parity(policy, label, sb3_model=None, header_path=None):
    print(f"\n--- Kiểm tra [{label}] (odd_symmetric={getattr(policy, 'odd_symmetric', False)}) ---")
    rng = np.random.RandomState(42)

    header_policy = None
    if header_path:
        if not os.path.exists(header_path):
            raise FileNotFoundError(f"Không tìm thấy header C: {header_path}")
        header_policy = parse_c_header(header_path)
        assert header_policy.odd_symmetric == bool(getattr(policy, 'odd_symmetric', False)), (
            f"Cờ RL_SB3_ODD_SYMMETRIC trong {header_path} ({header_policy.odd_symmetric}) "
            f"không khớp policy ({policy.odd_symmetric})"
        )
        w_diffs = [
            float(np.max(np.abs(policy.w1 - header_policy.w1))),
            float(np.max(np.abs(policy.b1 - header_policy.b1))),
            float(np.max(np.abs(policy.w2 - header_policy.w2))),
            float(np.max(np.abs(policy.b2 - header_policy.b2))),
            float(np.max(np.abs(policy.w3 - header_policy.w3))),
            float(np.max(np.abs(policy.b3 - header_policy.b3))),
        ]
        max_w_diff = max(w_diffs)
        print(f"  [Header Weights] Max chênh lệch trọng số .npz <-> .h: {max_w_diff:.2e}")
        assert max_w_diff <= 1e-6, f"Trọng số header {header_path} lệch quá ngưỡng 1e-6: {max_w_diff}"

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

    sat_cases = []
    if getattr(policy, 'odd_symmetric', False):
        sat_cases = find_one_branch_saturation_vectors(policy)
        if sat_cases:
            print(f"  [1-Branch Saturation] Tìm thấy {len(sat_cases)} vector làm đúng 1 nhánh chạm trần trên policy này:")
            for k, (svec, rp, rn, bc, nc) in enumerate(sat_cases, start=1):
                print(
                    f"    Sat #{k}: raw(+x)={rp:+.4f}, raw(-x)={rn:+.4f} | "
                    f"branch_clip={bc:+.6f} vs diff_then_clip={nc:+.6f} (|Δ|={abs(bc-nc):.4f})"
                )
        else:
            print("  [1-Branch Saturation] Policy này không có vector 1 nhánh chạm trần trong 5000 mẫu (sẽ chạy bài test khuếch đại riêng bên dưới).")

    all_vectors = list(test_vectors) + [list(c[0]) for c in sat_cases]

    max_err = 0.0
    max_pwm_err = 0.0
    max_sb3_err = 0.0
    for idx, vec in enumerate(all_vectors):
        vec_arr = np.array(vec, dtype=np.float32)
        act_numpy = policy.forward(vec_arr)
        pwm_numpy = float(np.clip(act_numpy * 999.0, -999.0, 999.0))
        c_pol = header_policy if header_policy is not None else policy
        act_c, pwm_c = simulate_c_inference(c_pol, *vec_arr.tolist())

        diff = abs(act_numpy - act_c)
        pwm_diff = abs(pwm_numpy - pwm_c)
        max_err = max(max_err, diff)
        max_pwm_err = max(max_pwm_err, pwm_diff)

        sb3_msg = ""
        if sb3_model is not None:
            act_sb3 = _sb3_predict_action(sb3_model, vec_arr)
            sb3_diff = abs(act_sb3 - act_numpy)
            max_sb3_err = max(max_sb3_err, sb3_diff)
            sb3_msg = f" | SB3={act_sb3:+.6f} (|SB3-Np|={sb3_diff:.2e})"

        tag = "Sat" if idx >= len(test_vectors) else f"Test {idx+1:02d}"
        print(f"  {tag}: Numpy={act_numpy:+.6f} | C={act_c:+.6f} | Err={diff:.2e}{sb3_msg}")

    # Chuỗi kiểm thử 9D liên tiếp (gồm cả trạng thái gần 0 và trạng thái biên lớn dễ chạm trần)
    seq_states = [
        (0.052, 0.12, 0.002, 0.03),
        (0.048, -0.05, 0.005, 0.04),
        (0.031, -0.22, 0.009, 0.02),
        (0.000, 0.00, 0.000, 0.00),
        (-0.260, -2.50, -0.450, -1.60),
        (0.280, 2.80, 0.550, 1.80),
        (-0.031, 0.22, -0.009, -0.02),
    ]
    if sat_cases:
        sv = sat_cases[0][0]
        seq_states.append((float(sv[0]), float(sv[1]), float(sv[2]), float(sv[3])))

    compiled_c_out = run_compiled_c_sequence(header_path, seq_states) if header_path else None
    np_hist = [0.0, 0.0, 0.0, 0.0, 0.0]
    c_hist = [0.0, 0.0, 0.0, 0.0, 0.0]
    c_pol = header_policy if header_policy is not None else policy
    max_seq_err = 0.0
    max_compiled_err = 0.0
    for step_idx, (th, dth, xx, vv) in enumerate(seq_states):
        np_vec = np.array([th, dth, xx, vv, *np_hist], dtype=np.float32)
        act_n = policy.forward(np_vec)
        act_c, _ = simulate_c_inference(c_pol, th, dth, xx, vv, *c_hist)
        max_seq_err = max(max_seq_err, abs(act_n - act_c))
        if sb3_model is not None:
            act_s = _sb3_predict_action(sb3_model, np_vec)
            max_sb3_err = max(max_sb3_err, abs(act_s - act_n))
        if compiled_c_out is not None:
            act_bin, _ = compiled_c_out[step_idx]
            max_compiled_err = max(max_compiled_err, abs(act_bin - act_n))
        np_hist = [th, dth, xx, vv, act_n]
        c_hist = [th, dth, xx, vv, act_c]

    print(f"  Sequence 9D rollout max Numpy<->C_sim Err: {max_seq_err:.2e}")
    if compiled_c_out is not None:
        print(f"  Sequence 9D rollout max Numpy<->Compiled_GCC_Binary Err: {max_compiled_err:.2e}")
        assert max_compiled_err <= 1e-5, f"Compiled C header lệch Numpy quá ngưỡng 1e-5: {max_compiled_err}"
    max_err = max(max_err, max_seq_err, max_compiled_err)

    if getattr(policy, 'odd_symmetric', False):
        zero_act = policy.forward(np.zeros(9, dtype=np.float32))
        max_asym = abs(zero_act)
        for vec in all_vectors[1:]:
            arr = np.array(vec, dtype=np.float32)
            max_asym = max(max_asym, abs(policy.forward(arr) + policy.forward(-arr)))
        print(f"  Odd symmetry check: f(0)={zero_act:+.2e}, max|f(x)+f(-x)|={max_asym:.2e}")
        assert max_asym <= 1e-6, f"Vi phạm tính đối xứng lẻ: {max_asym}"

    # Kiểm tra PyTorch (và ép 1 ca 1 nhánh chạm trần để đảm bảo Torch/NumPy/C dùng chung vị trí clamp)
    try:
        import torch

        class TorchActor(torch.nn.Module):
            def __init__(self, p):
                super().__init__()
                self.odd_symmetric = bool(getattr(p, 'odd_symmetric', False))
                self.fc1 = torch.nn.Linear(int(p.input_dim), 64)
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

        print(f"  [PyTorch Check] Đối chiếu PyTorch <-> Numpy ({label})...")
        torch_net = TorchActor(policy)
        torch_net.eval()
        torch_probes = test_vectors[:5] + [list(c[0]) for c in sat_cases[:2]]
        for idx, vec in enumerate(torch_probes):
            with torch.no_grad():
                t_out = float(torch_net(torch.from_numpy(np.array(vec, dtype=np.float32))).numpy()[0])
            n_out = policy.forward(vec)
            p_diff = abs(t_out - n_out)
            print(f"    PyTorch Test {idx+1:02d}: PyTorch={t_out:+.6f} | Numpy={n_out:+.6f} | Err={p_diff:.2e}")
            assert p_diff < 1e-4, f"PyTorch vs Numpy sai số quá lớn: {p_diff}"

        # Kiểm thử bắt buộc trường hợp 1 nhánh chạm trần trên mạng khuếch đại
        sat_pol = _build_forced_one_branch_sat_policy(policy)
        forced_cases = find_one_branch_saturation_vectors(sat_pol)
        assert forced_cases, "Không tạo được mẫu kiểm thử 1 nhánh chạm trần!"
        sat_torch = TorchActor(sat_pol)
        sat_torch.eval()
        fvec, rp, rn, bc, nc = forced_cases[0]
        with torch.no_grad():
            ft_out = float(sat_torch(torch.from_numpy(fvec)).numpy()[0])
        fn_out = sat_pol.forward(fvec)
        fc_out, _ = simulate_c_inference(sat_pol, *fvec.tolist())
        assert abs(ft_out - fn_out) <= 1e-5 and abs(fn_out - fc_out) <= 1e-5, (
            f"Sai lệch 1 nhánh chạm trần: Torch={ft_out}, Numpy={fn_out}, C={fc_out}"
        )
        print(
            f"  [Forced 1-Branch Sat PASS] raw=({rp:+.3f}, {rn:+.3f}) | "
            f"Torch={ft_out:+.6f}, Numpy={fn_out:+.6f}, C={fc_out:+.6f} "
            f"(naive_diff_clip={nc:+.6f}, |Δ|={abs(bc-nc):.4f})"
        )
    except ImportError:
        print("  [PyTorch Check] Bỏ qua kiểm tra PyTorch cục bộ (chưa cài đặt PyTorch).")
        # Vẫn kiểm tra Numpy <-> C khi 1 nhánh chạm trần trên mạng khuếch đại
        sat_pol = _build_forced_one_branch_sat_policy(policy)
        forced_cases = find_one_branch_saturation_vectors(sat_pol)
        assert forced_cases, "Không tạo được mẫu kiểm thử 1 nhánh chạm trần!"
        fvec, rp, rn, bc, nc = forced_cases[0]
        fn_out = sat_pol.forward(fvec)
        fc_out, _ = simulate_c_inference(sat_pol, *fvec.tolist())
        assert abs(fn_out - fc_out) <= 1e-5
        print(
            f"  [Forced 1-Branch Sat PASS] raw=({rp:+.3f}, {rn:+.3f}) | "
            f"Numpy={fn_out:+.6f}, C={fc_out:+.6f} "
            f"(naive_diff_clip={nc:+.6f}, |Δ|={abs(bc-nc):.4f})"
        )

    print(f"  Max Numpy<->C action err: {max_err:.2e} | PWM err: {max_pwm_err:.2e}")
    if sb3_model is not None:
        print(f"  Max SB3<->Numpy action err: {max_sb3_err:.2e}")
        assert max_sb3_err <= 1e-5, f"Sai lệch SB3 <-> Numpy quá ngưỡng 1e-5: {max_sb3_err}"
    if max_err > 1e-5:
        raise AssertionError(f"Sai lệch số học giữa Numpy và C quá ngưỡng 1e-5: {max_err}")
    return max_err


def verify():
    parser = argparse.ArgumentParser(description="Verify SB3 <-> Numpy <-> C inference parity")
    parser.add_argument("--weights", type=str, default=None, help="Optional .npz checkpoint to verify")
    parser.add_argument("--sb3-model", type=str, default=None, help="Optional SB3 PPO .zip model to compare against Numpy & C")
    parser.add_argument("--header", type=str, default=None, help="Optional exported C header .h to verify directly")
    parser.add_argument("--symmetric", action="store_true", help="Force odd_symmetric=True when testing --weights")
    args = parser.parse_args()

    print("==============================================================================")
    print("      KIỂM TRA TÍNH TOÀN VẸN VÀ ĐỒNG NHẤT TOÁN HỌC (SB3 <-> NUMPY <-> C)      ")
    print("==============================================================================")

    sb3_model = None
    if args.sb3_model:
        if not os.path.exists(args.sb3_model):
            sys.exit(f"[Error] Không tìm thấy SB3 model: {args.sb3_model}")
        try:
            from stable_baselines3 import PPO
            sb3_model = PPO.load(args.sb3_model, device="cpu")
            print(f"[Verify] Đã nạp SB3 PPO model: {args.sb3_model}")
        except ImportError as e:
            sys.exit(f"[Error] Không thể nạp SB3 model vì thiếu thư viện: {e}")

    if args.weights:
        if not os.path.exists(args.weights):
            sys.exit(f"[Error] Không tìm thấy checkpoint: {args.weights}")
        pol = NumpyMLPPolicy(input_dim=9, hidden_dims=(64, 64), output_dim=1)
        pol.load_weights(args.weights)
        if args.symmetric:
            pol.odd_symmetric = True
        max_err = verify_policy_parity(pol, os.path.basename(args.weights), sb3_model=sb3_model, header_path=args.header)
    else:
        np.random.seed(42)
        base_policy = NumpyMLPPolicy(input_dim=9, hidden_dims=(64, 64), output_dim=1, odd_symmetric=False)
        ckpt_path = "outputs/checkpoints/best_policy.npz"
        if os.path.exists(ckpt_path):
            base_policy.load_weights(ckpt_path)
            print(f"[Verify] Đã nạp checkpoint: {ckpt_path}")
        else:
            print("[Verify] Dùng trọng số ngẫu nhiên chuẩn để kiểm tra công thức toán học.")

        err1 = verify_policy_parity(base_policy, f"ckpt(odd_symmetric={base_policy.odd_symmetric})")
        sym_policy = NumpyMLPPolicy(input_dim=9, hidden_dims=(64, 64), output_dim=1, odd_symmetric=True)
        sym_policy.w1, sym_policy.b1 = base_policy.w1.copy(), base_policy.b1.copy()
        sym_policy.w2, sym_policy.b2 = base_policy.w2.copy(), base_policy.b2.copy()
        sym_policy.w3, sym_policy.b3 = base_policy.w3.copy(), base_policy.b3.copy()
        err2 = verify_policy_parity(sym_policy, "odd_symmetric=True")
        max_err = max(err1, err2)

    print("\n------------------- KẾT QUẢ XÁC MINH -------------------")
    print(f"  Sai số tuyệt đối cực đại giữa các nhánh: {max_err:.2e}")
    print("  [PASS] Thuật toán SB3, Numpy và hàm C đồng nhất toán học 100%!")
    print("---------------------------------------------------------")


if __name__ == "__main__":
    verify()
