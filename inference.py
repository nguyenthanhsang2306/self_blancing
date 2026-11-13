import os
import sys
import time
import argparse
import yaml
import numpy as np

from src.utils import plot_simulation_trajectory, export_sb3_c_header
from src.environment import SelfBalancingGymEnv
from src.models import NumpyMLPPolicy
from src.evaluator import PolicyEvaluator

def _load_policy(ckpt_path, obs_dim, net_arch, allow_random=False):
    policy = NumpyMLPPolicy(input_dim=obs_dim, hidden_dims=(net_arch[0], net_arch[1]), output_dim=1)
    if os.path.exists(ckpt_path):
        policy.load_weights(ckpt_path)
        print(f"[Model] Đã nạp thành công weights từ: {ckpt_path} (input_dim={policy.input_dim})")
    elif allow_random:
        print("[Warning] Chạy với trọng số ngẫu nhiên (--allow-random).")
    else:
        sys.exit(f"[Error] Không tìm thấy checkpoint: {ckpt_path}. Vui lòng huấn luyện hoặc chỉ định --weights.")
    return policy

def run_inference():
    parser = argparse.ArgumentParser(description="Test & Evaluate RL Self-Balancing Robot Controller")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config")
    parser.add_argument("--weights", type=str, default=None, help="Custom weights .npz path")
    parser.add_argument("--compare", nargs="+", default=None, help="Compare multiple .npz checkpoints on the exact same evaluation suite")
    parser.add_argument("--export-winner", action="store_true", help="Export best compared checkpoint to best_policy.npz and C header")
    parser.add_argument("--kick", type=float, default=0.0, help="External disturbance force in Newtons (default: 0.0)")
    parser.add_argument("--angle", type=float, default=None, help="Specific initial pitch angle in degrees (e.g. -3.0 or +5.0)")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration in seconds (default: 10.0s)")
    parser.add_argument("--eval-matrix", action="store_true", help="Run full evaluation matrix across angles, kicks, battery levels, and 20-30s drift tests")
    parser.add_argument("--allow-random", action="store_true", help="Allow running with uninitialized random weights")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"[Error] Không tìm thấy file cấu hình: {args.config}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    env = SelfBalancingGymEnv(config)
    obs_dim = env.observation_space.shape[0]
    net_arch = config['rl_training'].get('net_arch', [64, 64])
    evaluator = PolicyEvaluator(config)

    if args.compare:
        print("==============================================================================")
        print("      SO SÁNH CÁC CHECKPOINT TRÊN CÙNG BỘ TEST ĐỘC LẬP (90x10s + 6x20-30s)    ")
        print("      Xếp hạng: Ít ngã nhất -> Giữ VT (2s cuối) nhiều nhất -> Final X nhỏ     ")
        print("==============================================================================")
        compared = []
        for path in args.compare:
            pol = _load_policy(path, obs_dim, net_arch, allow_random=False)
            summ = evaluator.evaluate(pol, quick_eval=False)
            compared.append((path, pol, summ))

        compared.sort(key=lambda item: item[2]['ranking_key'], reverse=True)
        print("\n| Hạng | Checkpoint | Ngã | Giữ VT (2s cuối) | Final X TB | Max X TB | Settle TB (đạt) | RMS PWM |")
        print("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for rank, (path, _, s) in enumerate(compared, start=1):
            settle_str = f"{s['mean_settle_s']:.2f}s" if s['mean_settle_s'] is not None else "Không đạt"
            print(
                f"| #{rank} | {os.path.basename(path)} | {s['fall_count']}/{s['total_runs']} ({s['fall_rate_pct']:.1f}%) | "
                f"{s['hold_count']}/{s['total_runs']} ({s['hold_rate_pct']:.1f}%) | "
                f"{s['mean_final_x_m']*100:.2f} cm | {s['mean_max_x_m']*100:.2f} cm | "
                f"{settle_str} | {s['mean_rms_pwm']:.1f} |"
            )

        best_path, best_pol, best_summ = compared[0]
        print(f"\n[Winner] Bản tốt nhất: {best_path}")
        print("\n" + evaluator.format_report(best_summ) + "\n")

        if args.export_winner:
            best_dst = os.path.join(config['export']['checkpoint_dir'], "best_policy.npz")
            best_pol.save_weights(best_dst)
            hdr_dir = config['export']['header_output_dir']
            hdr_name = config['export'].get('header_filename', 'rl_sb3_stm32.h')
            export_sb3_c_header(best_pol, os.path.join(hdr_dir, hdr_name))
            stm32_inc = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Project_self", "Core", "Inc", hdr_name))
            if os.path.isdir(os.path.dirname(stm32_inc)):
                export_sb3_c_header(best_pol, stm32_inc)
                print(f"[Sync] Đã đồng bộ winner sang STM32 Header: {stm32_inc}")
        return

    ckpt_path = args.weights if args.weights is not None else os.path.join(config['export']['checkpoint_dir'], "best_policy.npz")
    policy = _load_policy(ckpt_path, obs_dim, net_arch, allow_random=args.allow_random)

    if args.eval_matrix:
        print("==============================================================================")
        print("      CHẠY MA TRẬN ĐÁNH GIÁ ĐỘC LẬP TOÀN DIỆN CHO CONTROLLER (9D PPO RL)     ")
        print("      90 ca 10s (Góc ±1°, ±3°, ±5° | Nhiễu 0, ±1, ±2N | Pin 7.2-8.4V) + 6 ca 20-30s ")
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
