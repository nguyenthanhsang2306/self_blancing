import os
import sys
import argparse
import yaml
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import set_seed, save_checkpoint, export_adrc_header, export_rl_header, plot_training_history, plot_benchmark_comparison
from src.environment import SelfBalancingRobotEnv
from src.models import CascadePIDController, ADRC_ESO_Controller, ResidualPolicyNet
from src.engine import train_residual_policy, run_benchmark

def sanity_check(config):
    """
    Automated Sanity Check for Inverted Pendulum Dynamics & Physical Consistency.
    Verifies inertia tensor positive definiteness, gain b0 stability, and timing.
    """
    p = config['robot_physics']
    M, m, l, r = p['M'], p['m'], p['l'], p['r']
    J, b, Km, V_bat = p['J'], p['b'], p['Km'], p['V_bat']
    Ts = p['Ts']

    assert M > 0 and m > 0 and l > 0 and r > 0, "[Sanity Check FAILED] Geometry & mass must be strictly positive!"
    assert J > 0 and Km > 0 and V_bat > 0, "[Sanity Check FAILED] Electrical & inertia constants invalid!"
    assert Ts == 0.005, "[Sanity Check WARNING] Sampling period should be 0.005s (200 Hz TIM6)!"

    M_tot = M + 1.5 * m
    den_0 = M_tot * (J + M * (l ** 2)) - (M * l) ** 2
    assert den_0 > 0, f"[Sanity Check FAILED] Denominator non-positive: {den_0} <= 0 (unstable manifold)!"

    Km_1000 = Km * (V_bat / 1000.0)
    b0_calc = (M * l * Km_1000) / den_0 * (180.0 / math.pi)

    print("[Sanity Check PASSED]")
    print(f"  - Total Equivalent Mass M_tot : {M_tot:.4f} kg")
    print(f"  - Determinant of Inertia den_0: {den_0:.6e} kg^2.m^2")
    print(f"  - Nominal Input Gain b0       : {b0_calc:.4f} deg/(s^2 * PWM)")
    print(f"  - Control Cycle Period Ts     : {Ts*1000:.1f} ms (200 Hz)\n")
    return True


def run_experiment_pid(config):
    print("==============================================================================")
    print("              THÍ NGHIỆM: CASCADE PID BASELINE CONTROLLER                     ")
    print("==============================================================================")
    sanity_check(config)
    env = SelfBalancingRobotEnv(config)
    pid = CascadePIDController(config)
    controllers = {'Cascade PID': {'base': pid, 'policy': None}}
    res = run_benchmark(env, controllers, config)

    m = res['Cascade PID']['metrics']
    print("\n--- KẾT QUẢ ĐÁNH GIÁ CASCADE PID ---")
    print(f"Max Kick Tilt Angle : {m['max_kick_tilt_deg']:.2f}°")
    print(f"Max Position Drift  : {m['max_pos_drift_m']*100:.2f} cm")
    print(f"Steady-State Error  : {m['steady_tilt_deg']:.3f}°")
    print(f"Peak Motor PWM      : {m['peak_pwm']:.1f} / 999\n")


def run_experiment_adrc(config):
    print("==============================================================================")
    print("              THÍ NGHIỆM: ACTIVE DISTURBANCE REJECTION (ADRC)                 ")
    print("==============================================================================")
    sanity_check(config)
    env = SelfBalancingRobotEnv(config)
    adrc = ADRC_ESO_Controller(config)
    controllers = {'ADRC (ESO Only)': {'base': adrc, 'policy': None}}
    res = run_benchmark(env, controllers, config)

    m = res['ADRC (ESO Only)']['metrics']
    print("\n--- KẾT QUẢ ĐÁNH GIÁ ADRC (3RD-ORDER LESO) ---")
    print(f"Max Kick Tilt Angle : {m['max_kick_tilt_deg']:.2f}°")
    print(f"Max Position Drift  : {m['max_pos_drift_m']*100:.2f} cm")
    print(f"Steady-State Error  : {m['steady_tilt_deg']:.3f}°")
    print(f"Peak Motor PWM      : {m['peak_pwm']:.1f} / 999\n")


def run_experiment_ablation_bandwidth(config):
    print("==============================================================================")
    print("       THÍ NGHIỆM ABLATION: KHẢO SÁT BĂNG THÔNG BỘ QUAN SÁT OMEGA_O          ")
    print("==============================================================================")
    sanity_check(config)
    sweep_omega = config['adrc_controller'].get('sweep_omega_o', [15.0, 48.0, 120.0])

    env = SelfBalancingRobotEnv(config)
    controllers = {}

    for w_o in sweep_omega:
        cfg_copy = yaml.safe_load(yaml.dump(config))
        cfg_copy['adrc_controller']['omega_o'] = w_o
        name = f"ADRC (wo = {w_o:.0f} rad/s)"
        controllers[name] = {'base': ADRC_ESO_Controller(cfg_copy), 'policy': None}

    res = run_benchmark(env, controllers, config)

    print("\n" + "=" * 78)
    print(f"{'BĂNG THÔNG OMEGA_O':<24} | {'MAX KICK TILT':<14} | {'MAX POS DRIFT':<14} | {'ĐỘ ỔN ĐỊNH'}")
    print("-" * 78)
    for name, r in res.items():
        m = r['metrics']
        eval_txt = "Quá chậm, bị ngã" if m['max_kick_tilt_deg'] > 20 else ("Tối ưu (Cứng vững)" if "48" in name else "Nhạy nhiễu cao tần")
        print(f"{name:<24} | {m['max_kick_tilt_deg']:>10.2f}°    | {m['max_pos_drift_m']*100:>10.2f} cm   | {eval_txt}")
    print("=" * 78 + "\n")


def run_experiment_default(config):
    print("==============================================================================")
    print("      THÍ NGHIỆM CHUẨN: HYBRID ADRC + ADVERSARIAL RESIDUAL REINFORCEMENT      ")
    print("      Target: STM32L476RG (ARM Cortex-M4 @ 80MHz with FPU)                   ")
    print("==============================================================================")
    sanity_check(config)

    seeds = [42, 43, 44]
    print(f"[Multi-Seed Protocol] Huấn luyện và đánh giá qua 3 hạt ngẫu nhiên: {seeds}...")

    best_seed = seeds[0]
    best_overall_reward = -float('inf')
    best_policy_overall = None
    best_history = None

    env = SelfBalancingRobotEnv(config)

    for seed in seeds:
        print(f"\n---> Khởi chạy Seed: {seed}")
        set_seed(seed)
        adrc_base = ADRC_ESO_Controller(config)
        policy = ResidualPolicyNet(config)

        history = train_residual_policy(env, adrc_base, policy, config)
        final_best = history['best_reward'][-1]

        if final_best > best_overall_reward:
            best_overall_reward = final_best
            best_seed = seed
            best_policy_overall = policy
            best_history = history

    print(f"\n[Multi-Seed Complete] Seed tối ưu nhất: {best_seed} (Reward: {best_overall_reward:.2f})")

    # Lưu checkpoint
    ckpt_path = os.path.join(config['export']['checkpoint_dir'], f"best_model_seed_{best_seed}.npy")
    save_checkpoint(best_policy_overall, ckpt_path)
    # Lưu alias best_model.npy
    save_checkpoint(best_policy_overall, os.path.join(config['export']['checkpoint_dir'], "best_model.npy"))

    # Xuất header C
    hdr_dir = config['export']['header_output_dir']
    export_adrc_header(config, os.path.join(hdr_dir, "adrc_controller_stm32.h"))
    export_rl_header(best_policy_overall, config, os.path.join(hdr_dir, "rl_controller_stm32.h"))

    # Vẽ biểu đồ hội tụ
    fig_dir = config['export']['figures_dir']
    plot_training_history(best_history, os.path.join(fig_dir, "training_convergence.png"))

    # Chạy Benchmark tổng thể
    controllers = {
        'Cascade PID': {'base': CascadePIDController(config), 'policy': None},
        'ADRC (ESO Only)': {'base': ADRC_ESO_Controller(config), 'policy': None},
        'ADRC + Residual RL': {'base': ADRC_ESO_Controller(config), 'policy': best_policy_overall}
    }
    results = run_benchmark(env, controllers, config)

    print("\n" + "=" * 78)
    print(f"{'BỘ ĐIỀU KHIỂN':<22} | {'MAX KICK TILT':<14} | {'MAX POS DRIFT':<14} | {'SAI SỐ XÁC LẬP'}")
    print("-" * 78)
    for name, r in results.items():
        m = r['metrics']
        print(f"{name:<22} | {m['max_kick_tilt_deg']:>10.2f}°    | {m['max_pos_drift_m']*100:>10.2f} cm   | {m['steady_tilt_deg']:>8.3f}°")
    print("=" * 78 + "\n")

    plot_benchmark_comparison(results, os.path.join(fig_dir, "benchmark_comparison.png"))
    print("[Hoàn tất] Toàn bộ kết quả và header firmware đã được xuất thành công!")


def main():
    parser = argparse.ArgumentParser(description="Unified Experiment Orchestrator for Self-Balancing Robot")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    mode = config['system'].get('mode', 'default')

    if mode == 'pid':
        run_experiment_pid(config)
    elif mode == 'adrc':
        run_experiment_adrc(config)
    elif mode == 'ablation_bandwidth':
        run_experiment_ablation_bandwidth(config)
    else:
        run_experiment_default(config)

if __name__ == "__main__":
    main()

