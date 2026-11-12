import os
import argparse
import yaml
import numpy as np

from src.utils import set_seed, export_sb3_c_header, plot_simulation_trajectory
from src.environment import SelfBalancingGymEnv
from src.models import NumpyMLPPolicy

def train():
    parser = argparse.ArgumentParser(description="Train RL Self-Balancing Robot via Stable-Baselines3 (PPO/DQN)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to default.yaml")
    parser.add_argument("--timesteps", type=int, default=None, help="Override training timesteps")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("==============================================================================")
    print("      REINFORCEMENT LEARNING FOR SELF-BALANCING ROBOT (SB3 + STM32)          ")
    print("      Framework: Gymnasium + Stable-Baselines3 (PPO/DQN) -> Pure C Header     ")
    print("      Target: STM32L476RG (ARM Cortex-M4 @ 80MHz with Hardware FPU)           ")
    print("==============================================================================")

    seed = config['system']['seed']
    set_seed(seed)

    # 1. Khởi tạo môi trường mô phỏng vật lý Gymnasium
    env = SelfBalancingGymEnv(config)
    print(f"[Env] Observation space: {env.observation_space.shape}")
    print(f"[Env] Action space:      {env.action_space}")

    # 2. Khởi tạo thuật toán Stable-Baselines3
    rl_cfg = config['rl_training']
    total_timesteps = args.timesteps if args.timesteps is not None else rl_cfg['total_timesteps']
    algo_name = rl_cfg.get('algorithm', 'PPO').upper()
    net_arch = rl_cfg.get('net_arch', [64, 64])

    numpy_policy = NumpyMLPPolicy(input_dim=4, hidden_dims=(net_arch[0], net_arch[1]), output_dim=1)

    try:
        from stable_baselines3 import PPO, DQN
        from stable_baselines3.common.callbacks import CheckpointCallback

        print(f"[Training] Khởi tạo thuật toán {algo_name} với mạng 2 tầng ẩn net_arch={net_arch}...")
        policy_kwargs = dict(net_arch=net_arch)

        if algo_name == 'DQN':
            # Yêu cầu action_mode = 'discrete' trong config
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=rl_cfg.get('learning_rate', 0.0003),
                gamma=rl_cfg.get('gamma', 0.99),
                batch_size=rl_cfg.get('batch_size', 64),
                policy_kwargs=policy_kwargs,
                verbose=1
            )
        else: # PPO
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=rl_cfg.get('learning_rate', 0.0003),
                gamma=rl_cfg.get('gamma', 0.99),
                n_steps=rl_cfg.get('n_steps', 2048),
                batch_size=rl_cfg.get('batch_size', 64),
                ent_coef=rl_cfg.get('ent_coef', 0.001),
                policy_kwargs=policy_kwargs,
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
        numpy_policy.load_from_sb3(model)

    except ImportError:
        print("[Notice] stable_baselines3 chưa cài đặt trên môi trường cục bộ này.")
        print("         Đang tạo checkpoint mẫu và xuất file C Header để sẵn sàng nhúng vào STM32.")
        print("         Khi chạy trên Google Colab / Kaggle, chỉ cần chạy !pip install -r requirements.txt để train đầy đủ.")

    # 3. Lưu weights dạng .npz
    ckpt_dir = config['export']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    npz_path = os.path.join(ckpt_dir, "best_policy.npz")
    numpy_policy.save_weights(npz_path)

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

    print("\n[Done] Pipeline hoàn thành xuất sắc!")

if __name__ == "__main__":
    train()
