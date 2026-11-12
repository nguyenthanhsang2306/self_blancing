import os
import sys
import argparse
import yaml
import numpy as np

from src.utils import set_seed, export_sb3_c_header
from src.environment import SelfBalancingGymEnv
from src.models import NumpyMLPPolicy

def train():
    parser = argparse.ArgumentParser(description="Train RL Self-Balancing Robot via Stable-Baselines3 (PPO)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to default.yaml")
    parser.add_argument("--timesteps", type=int, default=None, help="Override training timesteps")
    parser.add_argument("--demo-sample", action="store_true", help="Generate mock weights for testing without training (NOT for production)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"[Error] Không tìm thấy file cấu hình: {args.config}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("==============================================================================")
    print("      REINFORCEMENT LEARNING FOR SELF-BALANCING ROBOT (SB3 + STM32)          ")
    print("      Features: 9D Temporal Observation Space & Domain Randomization           ")
    print("      Target: STM32L476RG (ARM Cortex-M4 @ 80MHz with Hardware FPU)           ")
    print("==============================================================================")

    seed = config['system']['seed']
    set_seed(seed)

    # 1. Khởi tạo môi trường mô phỏng vật lý Gymnasium
    env = SelfBalancingGymEnv(config)
    obs_dim = env.observation_space.shape[0]
    print(f"[Env] Observation space dimension: {obs_dim}D ({config['gym_env'].get('obs_mode', '9d')})")
    print(f"[Env] Action space:                {env.action_space}")

    # 2. Khởi tạo thuật toán Stable-Baselines3
    rl_cfg = config['rl_training']
    total_timesteps = args.timesteps if args.timesteps is not None else rl_cfg['total_timesteps']
    algo_name = rl_cfg.get('algorithm', 'PPO').upper()
    net_arch = rl_cfg.get('net_arch', [64, 64])

    numpy_policy = NumpyMLPPolicy(input_dim=obs_dim, hidden_dims=(net_arch[0], net_arch[1]), output_dim=1)

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
        from stable_baselines3.common.env_util import make_vec_env
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
        train_env = make_vec_env(lambda: SelfBalancingGymEnv(config), n_envs=n_envs)
        print(f"[Parallel] Kích hoạt Vectorized Environment với {n_envs} môi trường song song (Tăng tốc gấp 8 lần)!")
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

    print(f"[Training] Bắt đầu quá trình huấn luyện {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # Lưu model SB3
    ckpt_dir = config['export']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    sb3_save_path = os.path.join(ckpt_dir, f"{algo_name.lower()}_balancing_robot.zip")
    model.save(sb3_save_path)
    print(f"[Checkpoint] Đã lưu model SB3 sang: {sb3_save_path}")

    # Trích xuất trọng số sang Numpy
    success = numpy_policy.load_from_sb3(model)
    if not success:
        sys.exit("[CRITICAL ERROR] Trích xuất trọng số từ Stable-Baselines3 thất bại! Hủy bỏ xuất C Header.")

    # 3. Lưu weights dạng .npz
    npz_path = os.path.join(ckpt_dir, "best_policy.npz")
    numpy_policy.save_weights(npz_path)
    print(f"[Checkpoint] Đã lưu weights Numpy sang: {npz_path}")

    # 4. Xuất file C Header cho STM32
    hdr_dir = config['export']['header_output_dir']
    hdr_name = config['export'].get('header_filename', 'rl_sb3_stm32.h')
    hdr_path = os.path.join(hdr_dir, hdr_name)
    export_sb3_c_header(numpy_policy, hdr_path)

    # Đồng bộ sang Project_self nếu có
    stm32_inc = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Project_self", "Core", "Inc", hdr_name))
    if os.path.isdir(os.path.dirname(stm32_inc)):
        export_sb3_c_header(numpy_policy, stm32_inc)
        print(f"[Sync] Đã đồng bộ trực tiếp file C Header sang STM32 Project: {stm32_inc}")

    print("\n[Done] Huấn luyện và xuất mô hình hoàn tất thành công 100%!")

if __name__ == "__main__":
    train()
