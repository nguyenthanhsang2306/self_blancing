import os
import sys
import time
import argparse
import yaml
import numpy as np

from src.utils import plot_simulation_trajectory
from src.environment import SelfBalancingGymEnv
from src.models import NumpyMLPPolicy
from src.evaluator import PolicyEvaluator

def run_inference():
    parser = argparse.ArgumentParser(description="Test & Evaluate RL Self-Balancing Robot Controller")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config")
    parser.add_argument("--weights", type=str, default=None, help="Custom weights .npz path")
    parser.add_argument("--kick", type=float, default=0.0, help="External disturbance force in Newtons (default: 0.0)")
    parser.add_argument("--angle", type=float, default=None, help="Specific initial pitch angle in degrees (e.g. -3.0 or +5.0)")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration in seconds (default: 10.0s)")
    parser.add_argument("--eval-matrix", action="store_true", help="Run full evaluation matrix across angles, kicks, and battery levels")
    parser.add_argument("--allow-random", action="store_true", help="Allow running with uninitialized random weights")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"[Error] Không tìm thấy file cấu hình: {args.config}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    ckpt_path = args.weights if args.weights is not None else os.path.join(config['export']['checkpoint_dir'], "best_policy.npz")

    env = SelfBalancingGymEnv(config)
    obs_dim = env.observation_space.shape[0]
    net_arch = config['rl_training'].get('net_arch', [64, 64])
    policy = NumpyMLPPolicy(input_dim=obs_dim, hidden_dims=(net_arch[0], net_arch[1]), output_dim=1)

    if os.path.exists(ckpt_path):
        policy.load_weights(ckpt_path)
        print(f"[Model] Đã nạp thành công weights từ: {ckpt_path} (input_dim={policy.input_dim})")
    elif args.allow_random:
        print("[Warning] Chạy với trọng số ngẫu nhiên (--allow-random).")
    else:
        sys.exit(f"[Error] Không tìm thấy checkpoint: {ckpt_path}. Vui lòng huấn luyện hoặc chỉ định --weights.")

    evaluator = PolicyEvaluator(config)

    if args.eval_matrix:
        print("==============================================================================")
        print("      CHẠY MA TRẬN ĐÁNH GIÁ ĐỘC LẬP TOÀN DIỆN CHO CONTROLLER (9D PPO RL)     ")
        print("      Ma trận: Góc ±1°, ±3°, ±5° | Xung nhiễu 0N, ±1N, ±2N | Pin 7.2V-8.4V   ")
        print(f"      Thời gian chạy: {args.duration:.1f}s cho mỗi kịch bản                   ")
        print("==============================================================================")
        summary = evaluator.evaluate(policy, quick_eval=False)
        report = evaluator.format_report(summary)
        print("\n" + report + "\n")
        return

    # Chạy đơn kịch bản và vẽ đồ thị
    print("==============================================================================")
    print("      CHẠY MÔ PHỎNG ĐỒNG BỘ CAO ĐỘ (10.0S SIM-TO-REAL RESPONSE)              ")
    print(f"      Disturbance Kick Force: {args.kick:.2f} N (t = 2.0s to 2.1s)            ")
    print(f"      Thời gian mô phỏng:     {args.duration:.1f} s                            ")
    print("==============================================================================")

    reset_opts = {'v_bat': config['robot_physics'].get('V_bat', 7.4), 'fixed_physics': True}
    if args.angle is not None:
        reset_opts['init_theta_deg'] = args.angle

    obs, _ = env.reset(seed=42, options=reset_opts)

    t_hist = []
    theta_hist = []
    x_hist = []
    pwm_hist = []

    steps = int(args.duration / env.Ts)
    latencies = []

    for step_i in range(steps):
        t = step_i * env.Ts
        dist_force = args.kick if (2.0 <= t <= 2.10 and args.kick != 0.0) else 0.0

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

    fig_dir = config['export']['figures_dir']
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "rl_inference_test.png")
    plot_simulation_trajectory(t_hist, theta_hist, x_hist, pwm_hist, fig_path)

    avg_lat = float(np.mean(latencies))
    max_lat = float(np.max(latencies))
    max_deg = float(np.max(np.abs(theta_hist)))
    max_x = float(np.max(np.abs(x_hist)))
    rms_pwm = float(np.sqrt(np.mean(np.square(pwm_hist))))

    print("\n------------------- KẾT QUẢ HIỆU NĂNG -------------------")
    print(f"  Thời gian cân bằng sống sót : {t_hist[-1]:.2f} / {args.duration:.2f} s")
    print(f"  Độ lệch góc lớn nhất        : {max_deg:.2f}° (Ngưỡng ngã: {env.cfg['gym_env']['fall_angle_deg']}°)")
    print(f"  Độ lệch vị trí lớn nhất     : {max_x*100:.2f} cm")
    print(f"  RMS PWM                     : {rms_pwm:.1f} / 999")
    print(f"  Độ trễ suy luận trung bình  : {avg_lat:.2f} µs (Cực đại: {max_lat:.2f} µs)")
    print(f"  Thời gian tính chu kỳ 200Hz : 5000 µs -> Tận dụng ~1% CPU của STM32!")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    run_inference()
