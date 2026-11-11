import os
import argparse
import yaml
import numpy as np

from src.utils import set_seed, save_checkpoint, export_adrc_header, export_rl_header, plot_training_history, plot_benchmark_comparison
from src.environment import SelfBalancingRobotEnv
from src.models import CascadePIDController, ADRC_ESO_Controller, ResidualPolicyNet
from src.engine import train_residual_policy, run_benchmark

def main():
    parser = argparse.ArgumentParser(description="Train Ultra-Robust RL + ADRC Controller for Self-Balancing Robot")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load Configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("==============================================================================")
    print("      ULTRA-ROBUST SELF-BALANCING ROBOT: RL + ADRC TRAINING PIPELINE          ")
    print("      Target: STM32L476RG (ARM Cortex-M4 @ 80MHz with Hardware FPU)           ")
    print("==============================================================================")

    # 1. Deterministic Seeding
    seed = config['system']['seed']
    set_seed(seed)
    print(f"[Setup] Random seed set to: {seed}")

    # 2. Instantiate Environment & Models
    env = SelfBalancingRobotEnv(config)
    adrc_base = ADRC_ESO_Controller(config)
    residual_policy = ResidualPolicyNet(config)

    # 3. Train Residual Policy under Adversarial Disturbance & Domain Randomization
    history = train_residual_policy(env, adrc_base, residual_policy, config)

    # 4. Save Best Checkpoint
    ckpt_dir = config['export']['checkpoint_dir']
    ckpt_path = os.path.join(ckpt_dir, "best_model.npy")
    save_checkpoint(residual_policy, ckpt_path)

    # 5. Export Embedded C Headers for STM32
    hdr_dir = config['export']['header_output_dir']
    adrc_hdr = os.path.join(hdr_dir, "adrc_controller_stm32.h")
    rl_hdr = os.path.join(hdr_dir, "rl_controller_stm32.h")
    export_adrc_header(config, adrc_hdr)
    export_rl_header(residual_policy, config, rl_hdr)

    # 6. Plot Training Convergence
    fig_dir = config['export']['figures_dir']
    train_plot_path = os.path.join(fig_dir, "training_convergence.png")
    plot_training_history(history, train_plot_path)

    # 7. Run Rigorous Comparative Benchmark
    print("\n[Benchmark] Executing Multi-Controller Extreme Stress Test...")
    controllers = {
        'Cascade PID': {'base': CascadePIDController(config), 'policy': None},
        'ADRC (ESO Only)': {'base': ADRC_ESO_Controller(config), 'policy': None},
        'ADRC + Residual RL': {'base': ADRC_ESO_Controller(config), 'policy': residual_policy}
    }

    results = run_benchmark(env, controllers, config)

    # 8. Output Performance Scorecard
    print("\n" + "=" * 78)
    print(f"{'CONTROLLER':<22} | {'MAX KICK TILT':<14} | {'MAX POS DRIFT':<14} | {'STEADY ERROR':<12}")
    print("-" * 78)
    for name, r in results.items():
        m = r['metrics']
        print(f"{name:<22} | {m['max_kick_tilt_deg']:>10.2f}°    | {m['max_pos_drift_m']*100:>10.2f} cm   | {m['steady_tilt_deg']:>8.3f}°")
    print("=" * 78 + "\n")

    # 9. Plot Comparative Benchmark Results
    bench_plot_path = os.path.join(fig_dir, "benchmark_comparison.png")
    plot_benchmark_comparison(results, bench_plot_path)

    print(f"[Done] All artifacts successfully generated in '{config['system']['output_dir']}/'!")

if __name__ == "__main__":
    main()

