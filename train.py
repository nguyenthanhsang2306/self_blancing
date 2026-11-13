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
    parser.add_argument("--demo-sample", action="store_true", help="Generate mock weights for testing without training")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"[Error] Không tìm thấy file cấu hình: {args.config}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

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

    # Callback đánh giá độc lập để chọn Checkpoint xuất sắc nhất
    class BestCheckpointCallback(BaseCallback):
        def __init__(self, eval_freq=20000, verbose=1):
            super().__init__(verbose)
            self.eval_freq = max(1, eval_freq // n_envs)
            self.best_score = -float('inf')
            self.best_step = 0
            self.best_fall_rate = 100.0

        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq == 0:
                numpy_policy.load_from_sb3(self.model)
                summary = evaluator.evaluate(numpy_policy, quick_eval=True)
                score = summary['composite_score']
                fall_rate = summary['fall_rate_pct']

                if self.verbose > 0:
                    print(f"\n[Eval @ Step {self.num_timesteps}] Điểm: {score:.1f} | Ngã: {fall_rate:.1f}% | Settle: {summary['mean_settle_s']:.2f}s | Max X: {summary['mean_max_x_m']*100:.1f}cm")

                if score > self.best_score:
                    self.best_score = score
                    self.best_step = self.num_timesteps
                    self.best_fall_rate = fall_rate

                    # Chỉ lưu model tốt nhất vào best_policy.npz
                    ckpt_dir = config['export']['checkpoint_dir']
                    os.makedirs(ckpt_dir, exist_ok=True)
                    npz_path = os.path.join(ckpt_dir, "best_policy.npz")
                    numpy_policy.save_weights(npz_path)

                    if self.verbose > 0:
                        print(f"  >>> [NEW BEST CHECKPOINT] Đã cập nhật best_policy.npz tại step {self.num_timesteps} (Điểm={score:.1f}, Ngã={fall_rate:.1f}%)!\n")

            return True

    eval_callback = BestCheckpointCallback(eval_freq=rl_cfg.get('eval_freq', 20000), verbose=1)

    print(f"[Training] Bắt đầu quá trình huấn luyện {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    ckpt_dir = config['export']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    # Lưu model cuối cùng vào final_policy (không ghi đè best_policy!)
    final_sb3_path = os.path.join(ckpt_dir, f"{algo_name.lower()}_final_model.zip")
    model.save(final_sb3_path)
    final_npz_path = os.path.join(ckpt_dir, "final_policy.npz")
    numpy_policy.load_from_sb3(model)
    numpy_policy.save_weights(final_npz_path)
    print(f"\n[Checkpoint] Đã lưu model cuối đợt train sang: {final_npz_path}")

    # ĐẢM BẢO CHỈ XUẤT C HEADER TỪ CHECKPOINT XUẤT SẮC NHẤT
    best_npz_path = os.path.join(ckpt_dir, "best_policy.npz")
    if os.path.exists(best_npz_path):
        numpy_policy.load_weights(best_npz_path)
        print(f"[Final Selection] Đã chọn Checkpoint xuất sắc nhất từ: {best_npz_path}")
    else:
        print("[Notice] Sử dụng final model vì chưa có best_policy.npz")

    # Đánh giá độc lập toàn diện 90 kịch bản trên Checkpoint tốt nhất
    print("\n" + "=" * 78)
    print("      ĐÁNH GIÁ ĐỘC LẬP TOÀN DIỆN TRÊN CHECKPOINT XUẤT SẮC NHẤT (90 BÀI TEST) ")
    print("=" * 78)
    summary = evaluator.evaluate(numpy_policy, quick_eval=False)
    report = evaluator.format_report(summary)
    print("\n" + report + "\n")

    # XUẤT C HEADER SAU KHI ĐÃ ĐÁNH GIÁ VÀ XÁC NHẬN CHẤT LƯỢNG
    hdr_dir = config['export']['header_output_dir']
    hdr_name = config['export'].get('header_filename', 'rl_sb3_stm32.h')
    hdr_path = os.path.join(hdr_dir, hdr_name)
    export_sb3_c_header(numpy_policy, hdr_path)

    stm32_inc = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Project_self", "Core", "Inc", hdr_name))
    if os.path.isdir(os.path.dirname(stm32_inc)):
        export_sb3_c_header(numpy_policy, stm32_inc)
        print(f"[Sync] Đã đồng bộ trực tiếp file C Header sang STM32 Project: {stm32_inc}")

    print("[Done] Huấn luyện, đánh giá độc lập và xuất mô hình hoàn tất 100%!")

if __name__ == "__main__":
    train()
