import os
import sys
import argparse
import yaml
import numpy as np

from src.utils import set_seed, export_sb3_c_header
from src.environment import SelfBalancingGymEnv
from src.models import NumpyMLPPolicy
from src.evaluator import PolicyEvaluator

def make_env_fn(config, env_rank, base_seed):
    def _init():
        return SelfBalancingGymEnv(config, env_rank=env_rank, seed=base_seed + env_rank * 1009)
    return _init

def train():
    parser = argparse.ArgumentParser(description="Train RL Self-Balancing Robot via Stable-Baselines3 (PPO)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to default.yaml")
    parser.add_argument("--timesteps", type=int, default=None, help="Override training timesteps")
    parser.add_argument("--w-pos", type=float, default=None, help="Override reward_weights.w_pos")
    parser.add_argument("--recenter-prob", type=float, default=None, help="Override gym_env.recenter_init_prob (e.g. 0.0 to disable, 0.30 for 30%%)")
    parser.add_argument("--tag", type=str, default=None, help="Optional suffix tag for checkpoint and header outputs")
    parser.add_argument("--deploy-gate", type=str, default="real_robot", choices=["real_robot", "no_kick", "all"], help="Gate criteria before updating canonical best_policy.npz / STM32 header")
    parser.add_argument("--update-stm32-header", action="store_true", help="Explicitly sync canonical STM32 header (still blocked unless deploy gate passes and untagged)")
    parser.add_argument("--demo-sample", action="store_true", help="Generate mock weights for testing without training")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"[Error] Không tìm thấy file cấu hình: {args.config}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.w_pos is not None:
        config.setdefault('reward_weights', {})['w_pos'] = float(args.w_pos)
        print(f"[Config Override] reward_weights.w_pos = {config['reward_weights']['w_pos']:.4f}")
    if args.recenter_prob is not None:
        if not 0.0 <= args.recenter_prob <= 1.0:
            parser.error("--recenter-prob must be between 0 and 1")
        config.setdefault('gym_env', {})['recenter_init_prob'] = float(args.recenter_prob)
        print(f"[Config Override] gym_env.recenter_init_prob = {config['gym_env']['recenter_init_prob']:.2f}")

    print("==============================================================================")
    print("      REINFORCEMENT LEARNING FOR SELF-BALANCING ROBOT (SB3 PPO + STM32)      ")
    print("      Curriculum: Full Range (±1°, ±3°, ±5°) + Bidirectional Kicks             ")
    print("      Target Hardware: STM32L476RG (Cortex-M4 @ 80MHz with FPU)                ")
    print("==============================================================================")

    seed = config['system']['seed']
    set_seed(seed)

    env = SelfBalancingGymEnv(config, env_rank=0, seed=seed)
    obs_dim = env.observation_space.shape[0]
    print(f"[Env] Observation space dimension: {obs_dim}D ({config['gym_env'].get('obs_mode', '9d')})")
    print(f"[Env] Action space:                {env.action_space}")

    rl_cfg = config['rl_training']
    total_timesteps = args.timesteps if args.timesteps is not None else rl_cfg['total_timesteps']
    algo_name = rl_cfg.get('algorithm', 'PPO').upper()
    net_arch = rl_cfg.get('net_arch', [64, 64])

    numpy_policy = NumpyMLPPolicy(input_dim=obs_dim, hidden_dims=(net_arch[0], net_arch[1]), output_dim=1)
    evaluator = PolicyEvaluator(config)

    if args.demo_sample:
        print("[Notice] Chế độ --demo-sample: Tạo trọng số mẫu để kiểm tra pipeline xuất C.")
        ckpt_dir = config['export']['checkpoint_dir']
        os.makedirs(ckpt_dir, exist_ok=True)
        demo_npz = os.path.join(ckpt_dir, "demo_sample_policy.npz")
        numpy_policy.save_weights(demo_npz)
        print(f"[Demo] Đã lưu trọng số mẫu sang: {demo_npz}")
        return

    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as e:
        print("\n" + "=" * 78)
        print("[ERROR] Thiếu thư viện huấn luyện Reinforcement Learning:")
        print(f"        {e}")
        print("Huấn luyện PPO yêu cầu: pip install torch stable-baselines3 gymnasium")
        print("Để huấn luyện đầy đủ và xuất checkpoint thực tế, vui lòng chạy trên Google Colab")
        print("(sử dụng train_colab.ipynb) hoặc môi trường có GPU/PyTorch.")
        print("=" * 78 + "\n")
        sys.exit(1)

    if algo_name != 'PPO':
        sys.exit(f"[Error] Thuật toán '{algo_name}' không được hỗ trợ trong pipeline xuất C trực tiếp. Vui lòng chọn 'PPO'.")

    print(f"[Training] Khởi tạo PPO với kiến trúc actor Tanh: [{obs_dim} -> {net_arch[0]} (Tanh) -> {net_arch[1]} (Tanh) -> 1 (Linear)]...")
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(pi=net_arch, vf=net_arch)
    )

    n_envs = rl_cfg.get('n_envs', 8)
    try:
        # Khởi tạo từng môi trường với seed và RandomState độc lập
        env_fns = [make_env_fn(config, i, seed) for i in range(n_envs)]
        train_env = DummyVecEnv(env_fns)
        print(f"[Parallel] Kích hoạt Vectorized Environment với {n_envs} môi trường song song (Độc lập RNG, Tăng tốc 8x)!")
    except Exception as env_err:
        print(f"[Notice] Sử dụng môi trường đơn do: {env_err}")
        train_env = env

    n_steps_per_env = max(256, int(2048 / n_envs)) if n_envs > 1 else 2048

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=rl_cfg.get('learning_rate', 0.0003),
        gamma=rl_cfg.get('gamma', 0.99),
        n_steps=n_steps_per_env,
        batch_size=rl_cfg.get('batch_size', 64),
        ent_coef=rl_cfg.get('ent_coef', 0.001),
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1
    )

    ckpt_dir = config['export']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    run_label = args.tag if args.tag else "candidate"
    run_suffix = f"_{run_label}"
    best_npz_path = os.path.join(ckpt_dir, f"best_policy{run_suffix}.npz" if args.tag else "candidate_best_policy.npz")
    final_npz_path = os.path.join(ckpt_dir, f"final_policy{run_suffix}.npz" if args.tag else "candidate_final_policy.npz")
    final_sb3_path = os.path.join(ckpt_dir, f"{algo_name.lower()}_final_model{run_suffix}.zip")

    # Xóa checkpoint cũ của cùng nhãn chạy để chỉ chọn checkpoint trong lần train hiện tại
    if os.path.exists(best_npz_path):
        os.remove(best_npz_path)
        print(f"[Cleanup] Đã xóa checkpoint cũ {best_npz_path} để tránh dùng nhầm kết quả đợt train trước.")

    def _fmt_eval_line(step_label, summary):
        settle_str = f"{summary['mean_settle_s']:.2f}s" if summary['mean_settle_s'] is not None else "N/A"
        return (
            f"[{step_label}] Ngã: {summary['fall_count']}/{summary['total_runs']} ({summary['fall_rate_pct']:.1f}%) | "
            f"Giữ VT: {summary['hold_count']}/{summary['total_runs']} | "
            f"Final X: {summary['mean_final_x_m']*100:.2f}cm | Settle: {settle_str}"
        )

    # Callback đánh giá độc lập để chọn Checkpoint tốt nhất của riêng lần train này
    class BestCheckpointCallback(BaseCallback):
        def __init__(self, eval_freq=20000, verbose=1):
            super().__init__(verbose)
            self.eval_freq = max(1, eval_freq // n_envs)
            self.best_key = None
            self.best_step = 0
            self.has_best = False

        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq == 0:
                numpy_policy.load_from_sb3(self.model)
                summary = evaluator.evaluate(numpy_policy, quick_eval=True)
                key = summary['ranking_key']

                if self.verbose > 0:
                    print("\n" + _fmt_eval_line(f"Eval @ Step {self.num_timesteps}", summary))

                if self.best_key is None or key > self.best_key:
                    self.best_key = key
                    self.best_step = self.num_timesteps
                    self.has_best = True
                    numpy_policy.save_weights(best_npz_path)

                    if self.verbose > 0:
                        print(f"  >>> [NEW BEST CHECKPOINT] Đã cập nhật {best_npz_path} tại step {self.num_timesteps}!\n")

            return True

    eval_callback = BestCheckpointCallback(eval_freq=rl_cfg.get('eval_freq', 20000), verbose=1)

    print(f"[Training] Bắt đầu quá trình huấn luyện {total_timesteps} timesteps (w_pos={config['reward_weights']['w_pos']})...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Lưu model cuối cùng vào final_policy (không ghi đè best_policy nếu chưa đánh giá!)
    model.save(final_sb3_path)
    numpy_policy.load_from_sb3(model)
    numpy_policy.save_weights(final_npz_path)
    print(f"\n[Checkpoint] Đã lưu model cuối đợt train sang: {final_npz_path}")

    # Đánh giá nhanh model cuối để so với checkpoint tốt nhất trong chính lần train hiện tại
    final_quick = evaluator.evaluate(numpy_policy, quick_eval=True)
    print(_fmt_eval_line(f"Eval @ Final ({total_timesteps})", final_quick))
    if (not eval_callback.has_best) or (final_quick['ranking_key'] > eval_callback.best_key):
        eval_callback.best_key = final_quick['ranking_key']
        eval_callback.best_step = total_timesteps
        eval_callback.has_best = True
        numpy_policy.save_weights(best_npz_path)
        print(f"  >>> [NEW BEST CHECKPOINT] Model cuối đợt train đạt hạng cao nhất -> {best_npz_path}")

    # ĐẢM BẢO CHỈ CHỌN CHECKPOINT CỦA LẦN TRAIN HIỆN TẠI
    if eval_callback.has_best and os.path.exists(best_npz_path):
        numpy_policy.load_weights(best_npz_path)
        print(f"[Final Selection] Đã chọn Checkpoint tốt nhất của lần train này (step {eval_callback.best_step}): {best_npz_path}")
    else:
        print("[Notice] Sử dụng final model trong phiên chạy hiện tại.")

    # Đánh giá độc lập toàn diện (90 ca 10s + 2 ca 0° ±1.5N + 6 ca dài 20–30s) trên Checkpoint tốt nhất
    print("\n" + "=" * 78)
    print("      ĐÁNH GIÁ ĐỘC LẬP TOÀN DIỆN TRÊN CHECKPOINT TỐT NHẤT (98 KỊCH BẢN) ")
    print("=" * 78)
    summary = evaluator.evaluate(numpy_policy, quick_eval=False)
    report = evaluator.format_report(summary, deploy_gate_mode=args.deploy_gate)
    print("\n" + report + "\n")

    report_path = os.path.join("outputs", "reports", f"eval_report_{run_label}.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(report + "\n")
    print(f"[Report] Đã lưu báo cáo đánh giá sang: {report_path}")

    # XUẤT C HEADER SAU KHI ĐÃ ĐÁNH GIÁ VÀ KIỂM TRA CỔNG AN TOÀN STM32
    hdr_dir = config['export']['header_output_dir']
    base_hdr = config['export'].get('header_filename', 'rl_sb3_stm32.h')
    stem, ext = os.path.splitext(base_hdr)
    scoped_hdr_name = f"{stem}_{run_label}{ext}"
    scoped_hdr_path = os.path.join(hdr_dir, scoped_hdr_name)
    export_sb3_c_header(numpy_policy, scoped_hdr_path)

    if args.tag:
        if args.update_stm32_header:
            print("[Notice] Bỏ qua --update-stm32-header khi đang chạy ở chế độ --tag (chỉ xuất file riêng có nhãn).")
        print(f"[Tag Mode] Đã xuất bản có nhãn tại {best_npz_path} & {scoped_hdr_path}. Giữ nguyên rl_sb3_stm32.h và Project_self.")
    else:
        passed, reasons = evaluator.check_deploy_gate(summary, gate_mode=args.deploy_gate)
        if passed:
            best_dst = os.path.join(ckpt_dir, "best_policy.npz")
            numpy_policy.save_weights(best_dst)
            canonical_hdr_path = os.path.join(hdr_dir, base_hdr)
            export_sb3_c_header(numpy_policy, canonical_hdr_path)
            print(f"[Deploy Gate PASSED] Đã cập nhật {best_dst} và {canonical_hdr_path}.")
            stm32_inc = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Project_self", "Core", "Inc", base_hdr))
            if os.path.isdir(os.path.dirname(stm32_inc)):
                if args.update_stm32_header:
                    export_sb3_c_header(numpy_policy, stm32_inc)
                    print(f"[Sync] Đã đồng bộ trực tiếp file C Header sang STM32 Project: {stm32_inc}")
                else:
                    print(f"[Sync Skipped] Model đã đạt gate '{args.deploy_gate}', thêm --update-stm32-header nếu muốn chép trực tiếp sang {stm32_inc}")
        else:
            print(
                f"[Gate Blocked] Model ứng viên CHƯA ĐẠT tiêu chí xuất STM32 (gate='{args.deploy_gate}'):\n"
                + "\n".join(f"  {r}" for r in reasons)
                + f"\n  -> Chỉ lưu bản ứng viên tại {best_npz_path} & {scoped_hdr_path}; KHÔNG ghi đè best_policy.npz, {base_hdr} hay Project_self."
            )

    print("[Done] Huấn luyện, đánh giá độc lập và xuất mô hình hoàn tất 100%!")

if __name__ == "__main__":
    train()
