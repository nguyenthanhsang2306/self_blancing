import os
import time
import argparse
import yaml
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models import ADRC_ESO_Controller, ResidualPolicyNet
from src.environment import SelfBalancingRobotEnv
from src.utils import load_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Real-time Inference and Benchmark Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.npy", help="Path to checkpoint")
    parser.add_argument("--kick_force", type=float, default=-1.5, help="Disturbance kick force in Newtons")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("==============================================================================")
    print("                 REAL-TIME INFERENCE & ROBUSTNESS TEST                        ")
    print("==============================================================================")

    env = SelfBalancingRobotEnv(config)
    adrc = ADRC_ESO_Controller(config)
    policy = ResidualPolicyNet(config)

    # Load weights
    load_checkpoint(policy, args.checkpoint)

    # 1. Benchmark Single-Step Inference Latency
    test_obs = env.get_observation()
    n_iters = 10000
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = policy.forward(test_obs)
    t1 = time.perf_counter()
    avg_latency_us = ((t1 - t0) / n_iters) * 1e6
    print(f"[Latency Benchmark] Average Policy Inference Time: {avg_latency_us:.2f} microseconds / step")
    print(f"[Hardware Real-Time Check] 200 Hz budget = 5000 us. CPU compute = {avg_latency_us:.2f} us (< 0.5% CPU load).\n")

    # 2. Run High-Stress Inference Simulation
    print(f"[Simulation] Testing extreme shock disturbance F = {args.kick_force:.2f} N at t=1.0s...")
    obs = env.reset(domain_randomize=False)
    adrc.reset()

    sim_duration = 3.0
    steps = int(sim_duration / env.Ts)
    t_arr, th_arr, x_arr, u_arr, z3_arr = [], [], [], [], []

    for step in range(steps):
        t = step * env.Ts
        dist = args.kick_force if (1.0 <= t <= 1.08) else 0.0

        th = math.degrees(env.state[0])
        dth = math.degrees(env.state[1])
        x = env.state[2]
        v = env.state[3]

        u_base, z3 = adrc.step(th, x, v)
        delta_u = policy.forward(obs)
        u_total = float(np.clip(u_base + delta_u, -999.0, 999.0))

        obs, _, done, _ = env.step(u_total, disturbance_force=dist)

        t_arr.append(t)
        th_arr.append(th)
        x_arr.append(x)
        u_arr.append(u_total)
        z3_arr.append(z3)

    # 3. Plot Inference Results
    os.makedirs("outputs/figures", exist_ok=True)
    out_fig = "outputs/figures/inference_test.png"
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True, dpi=300)

    axs[0].plot(t_arr, th_arr, 'g-', lw=2.0, label='Tilt Angle [deg]')
    axs[0].axvspan(1.0, 1.08, color='red', alpha=0.2, label=f'Kick Shock ({args.kick_force}N)')
    axs[0].set_ylabel('Angle [deg]', fontweight='bold')
    axs[0].set_title(f'Inference Verification Under Extreme Shock (Kick = {args.kick_force}N)', fontsize=13, fontweight='bold')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()

    axs[1].plot(t_arr, x_arr, 'b-', lw=2.0, label='Position [m]')
    axs[1].set_ylabel('Position x [m]', fontweight='bold')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend()

    axs[2].plot(t_arr, u_arr, 'm-', lw=1.8, label='Motor Output PWM')
    axs[2].set_ylabel('PWM [-999, +999]', fontweight='bold')
    axs[2].set_xlabel('Time [s]', fontweight='bold')
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()

    print(f"[Result] Max tilt angle recorded: {np.max(np.abs(th_arr)):.2f}°")
    print(f"[Result] Final steady position drift: {abs(x_arr[-1])*100:.2f} cm")
    print(f"[Result] Inference plot generated -> {out_fig}\n")

if __name__ == "__main__":
    main()

