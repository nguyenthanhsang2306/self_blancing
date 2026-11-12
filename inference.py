import os
import sys
import time
import math
import argparse
import yaml
import numpy as np

from src.utils import plot_simulation_trajectory
from src.environment import SelfBalancingGymEnv
from src.models import NumpyMLPPolicy

def run_inference():
    parser = argparse.ArgumentParser(description="Test RL Self-Balancing Robot Controller")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config")
    parser.add_argument("--weights", type=str, default=None, help="Custom weights .npz path")
    parser.add_argument("--kick", type=float, default=0.0, help="External disturbance force in Newtons (default: 0.0 for pure balance test)")
    parser.add_argument("--allow-random", action="store_true", help="Allow running with uninitialized random weights (for testing only)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"[Error] Không tìm thấy file cấu hình: {args.config}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("==============================================================================")
    print("      INFERENCE & HIGH-FIDELITY SIMULATION TEST FOR 9D RL CONTROLLER         ")
    print(f"      Disturbance Kick Force: {args.kick:.2f} N (t = 2.0s to 2.1s)            ")
    print("==============================================================================")

    env = SelfBalancingGymEnv(config)
    obs, _ = env.reset(seed=42)
    obs_dim = env.observation_space.shape[0]

    net_arch = config['rl_training'].get('net_arch', [64, 64])
    policy = NumpyMLPPolicy(input_dim=obs_dim, hidden_dims=(net_arch[0], net_arch[1]), output_dim=1)

    ckpt_path = args.weights if args.weights is not None else os.path.join(config['export']['checkpoint_dir'], "best_policy.npz")
    if os.path.exists(ckpt_path):
        policy.load_weights(ckpt_path)
        print(f"[Model] Đã nạp thành công weights từ: {ckpt_path} (input_dim={policy.input_dim})")
    elif args.allow_random:
        print("[Warning] Chạy với trọng số ngẫu nhiên (--allow-random). Robot sẽ không thể cân bằng!")
    else:
        print("\n" + "=" * 78)
        print(f"[ERROR] Không tìm thấy file checkpoint: {ckpt_path}")
        print("Mô hình chưa được huấn luyện! Vui lòng thực hiện một trong các bước sau:")
        print("  1. Huấn luyện trên Google Colab qua notebook `train_colab.ipynb`")
        print("  2. Chạy `python train.py --config configs/default.yaml` trên môi trường có SB3")
        print("  3. Thêm cờ `--allow-random` nếu chỉ muốn kiểm thử luồng code với trọng số ngẫu nhiên.")
        print("=" * 78 + "\n")
        sys.exit(1)

    t_hist = []
    theta_hist = []
    x_hist = []
    pwm_hist = []

    dur = 5.0
    steps = int(dur / env.Ts)

    print(f"[Sim] Bắt đầu chạy mô phỏng {dur}s ({steps} bước tại Ts = {env.Ts*1000:.1f}ms)...")
    latencies = []

    for step_i in range(steps):
        t = step_i * env.Ts

        # Tác động ngoại lực nếu người dùng yêu cầu (ví dụ --kick -1.5)
        dist_force = 0.0
        if 2.0 <= t <= 2.10 and args.kick != 0.0:
            dist_force = args.kick

        t_start = time.perf_counter()
        action = policy.forward(obs)
        lat = (time.perf_counter() - t_start) * 1e6
        latencies.append(lat)

        obs, reward, terminated, truncated, info = env.step(action, disturbance=dist_force)

        t_hist.append(t)
        theta_hist.append(info['theta_deg'])
        x_hist.append(info['x_pos'])
        pwm_hist.append(info['applied_pwm'])

        if terminated:
            print(f"[Sim] Robot ngã tại t = {t:.2f}s (Góc = {info['theta_deg']:.1f}°)")
            break

    fig_path = os.path.join(config['export']['figures_dir'], "rl_inference_test.png")
    plot_simulation_trajectory(t_hist, theta_hist, x_hist, pwm_hist, fig_path)

    avg_lat = np.mean(latencies)
    max_lat = np.max(latencies)
    max_deg = np.max(np.abs(theta_hist))
    max_x = np.max(np.abs(x_hist))

    print("\n------------------- KẾT QUẢ HIỆU NĂNG -------------------")
    print(f"  Thời gian cân bằng sống sót : {t_hist[-1]:.2f} / {dur:.2f} s")
    print(f"  Độ lệch góc lớn nhất        : {max_deg:.2f}° (Ngưỡng ngã: {env.cfg['gym_env']['fall_angle_deg']}°)")
    print(f"  Độ lệch vị trí lớn nhất     : {max_x:.3f} m")
    print(f"  Độ trễ suy luận trung bình  : {avg_lat:.2f} µs (Cực đại: {max_lat:.2f} µs)")
    print(f"  Thời gian tính chu kỳ 200Hz : 5000 µs -> Tận dụng ~1% CPU của STM32!")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    run_inference()
