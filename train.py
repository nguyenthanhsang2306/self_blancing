import os
import sys
import copy
import glob
import argparse
import yaml
import numpy as np

from src.utils import set_seed, export_sb3_c_header
from src.environment import SelfBalancingGymEnv
from src.models import NumpyMLPPolicy, OddSymmetricActorCriticPolicy
from src.evaluator import PolicyEvaluator
from verify_export import (
    verify_policy_parity,
    find_one_branch_saturation_vectors,
    simulate_c_inference,
)

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
    parser.add_argument("--top-k", type=int, default=3, help="Number of top quick-eval checkpoints to keep for Stage-2 22-case screening")
    sym_group = parser.add_mutually_exclusive_group()
    sym_group.add_argument("--symmetric", dest="odd_symmetric", action="store_true", help="Enable odd-symmetric actor 0.5*(pi(obs)-pi(-obs))")
    sym_group.add_argument("--no-symmetric", dest="odd_symmetric", action="store_false", help="Disable odd-symmetric actor")
    parser.set_defaults(odd_symmetric=None)
    parser.add_argument("--deploy-gate", type=str, default="real_robot", choices=["real_robot", "no_kick", "all"], help="Gate criteria before updating canonical best_policy.npz / STM32 header")
    parser.add_argument("--update-stm32-header", action="store_true", help="Explicitly sync canonical STM32 header (still blocked unless deploy gate passes and untagged)")
    parser.add_argument("--demo-sample", action="store_true", help="Generate mock weights for testing without training")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"[Error] Không tìm thấy file cấu hình: {args.config}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
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
    use_odd_sym = bool(rl_cfg.get('odd_symmetric', False)) if args.odd_symmetric is None else bool(args.odd_symmetric)
    print(f"[Policy Mode] odd_symmetric = {use_odd_sym} | r_hold_bonus = {config.get('reward_weights', {}).get('r_hold_bonus', 0.0)}")

    numpy_policy = NumpyMLPPolicy(
        input_dim=obs_dim,
        hidden_dims=(net_arch[0], net_arch[1]),
        output_dim=1,
        odd_symmetric=use_odd_sym,
    )
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

    print(f"[Training] Khởi tạo PPO với kiến trúc actor Tanh: [{obs_dim} -> {net_arch[0]} (Tanh) -> {net_arch[1]} (Tanh) -> 1 (Linear)] (odd_symmetric={use_odd_sym})...")
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(pi=net_arch, vf=net_arch)
    )

    if use_odd_sym:
        if OddSymmetricActorCriticPolicy is None:
            sys.exit("[Error] OddSymmetricActorCriticPolicy không khả dụng.")
        policy_cls = OddSymmetricActorCriticPolicy
    else:
        policy_cls = "MlpPolicy"

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
        policy_cls,
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

    # Kiểm tra nhanh tính đối xứng lẻ & đồng nhất SB3 <-> Numpy (gồm ca 1 nhánh chạm trần) trước khi train
    numpy_policy.load_from_sb3(model, odd_symmetric=use_odd_sym)
    if use_odd_sym:
        zero_obs = np.zeros(obs_dim, dtype=np.float32)
        probe_obs = np.linspace(-0.08, 0.08, obs_dim, dtype=np.float32)
        sb3_zero = float(np.asarray(model.predict(zero_obs, deterministic=True)[0]).item())
        sb3_pos = float(np.asarray(model.predict(probe_obs, deterministic=True)[0]).item())
        sb3_neg = float(np.asarray(model.predict(-probe_obs, deterministic=True)[0]).item())
        np_pos = numpy_policy.forward(probe_obs)
        assert abs(sb3_zero) < 1e-6, f"Odd-symmetry check failed at 0: {sb3_zero}"
        assert abs(sb3_pos + sb3_neg) < 1e-6, f"Odd-symmetry check failed: {sb3_pos} vs {sb3_neg}"
        assert abs(sb3_pos - np_pos) < 1e-5, f"SB3 vs Numpy parity failed: {sb3_pos} vs {np_pos}"

        # Kiểm thử bắt buộc trường hợp 1 nhánh chạm trần [-1, 1] trên bản sao policy của SB3
        probe_sb3_pol = copy.deepcopy(model.policy)
        with torch.no_grad():
            probe_sb3_pol.action_net.weight.mul_(200.0)
            probe_sb3_pol.action_net.bias.fill_(0.45)
        probe_np_pol = NumpyMLPPolicy(input_dim=obs_dim, hidden_dims=(net_arch[0], net_arch[1]), output_dim=1, odd_symmetric=True)
        probe_np_pol.w1 = probe_sb3_pol.mlp_extractor.policy_net[0].weight.detach().cpu().numpy().copy()
        probe_np_pol.b1 = probe_sb3_pol.mlp_extractor.policy_net[0].bias.detach().cpu().numpy().copy()
        probe_np_pol.w2 = probe_sb3_pol.mlp_extractor.policy_net[2].weight.detach().cpu().numpy().copy()
        probe_np_pol.b2 = probe_sb3_pol.mlp_extractor.policy_net[2].bias.detach().cpu().numpy().copy()
        probe_np_pol.w3 = probe_sb3_pol.action_net.weight.detach().cpu().numpy().copy()
        probe_np_pol.b3 = probe_sb3_pol.action_net.bias.detach().cpu().numpy().copy()
        sat_probes = find_one_branch_saturation_vectors(probe_np_pol, max_trials=2000, seed=99)
        assert sat_probes, "Không tìm thấy mẫu 1 nhánh chạm trần trên probe_sb3_pol!"
        svec, rp, rn, bc, nc = sat_probes[0]
        sb3_sat = float(np.asarray(probe_sb3_pol.predict(svec, deterministic=True)[0]).item())
        np_sat = probe_np_pol.forward(svec)
        c_sat, _ = simulate_c_inference(probe_np_pol, *svec.tolist())
        assert abs(sb3_sat - np_sat) <= 1e-5 and abs(np_sat - c_sat) <= 1e-5, (
            f"1-branch saturation mismatch: SB3={sb3_sat}, Numpy={np_sat}, C={c_sat}"
        )
        print(
            f"[Symmetry Verified] SB3 f(0)={sb3_zero:+.2e}, f(x)+f(-x)={sb3_pos+sb3_neg:+.2e} | "
            f"1-Branch Sat (raw={rp:+.2f},{rn:+.2f}): SB3={sb3_sat:+.5f}, Numpy={np_sat:+.5f}, C={c_sat:+.5f} "
            f"(naive_diff_clip={nc:+.5f})"
        )

    ckpt_dir = config['export']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    hdr_dir = config['export']['header_output_dir']
    base_hdr = config['export'].get('header_filename', 'rl_sb3_stm32.h')
    stem, ext = os.path.splitext(base_hdr)
    run_label = args.tag if args.tag else "candidate"
    run_suffix = f"_{run_label}"
    best_npz_path = os.path.join(ckpt_dir, f"best_policy{run_suffix}.npz" if args.tag else "candidate_best_policy.npz")
    runner_up_npz_path = os.path.join(ckpt_dir, f"runner_up_policy{run_suffix}.npz" if args.tag else "candidate_runner_up_policy.npz")
    final_npz_path = os.path.join(ckpt_dir, f"final_policy{run_suffix}.npz" if args.tag else "candidate_final_policy.npz")
    final_sb3_path = os.path.join(ckpt_dir, f"{algo_name.lower()}_final_model{run_suffix}.zip")
    final_hdr_path = os.path.join(hdr_dir, f"{stem}_{run_label}_final{ext}")
    cand_glob = os.path.join(ckpt_dir, f"cand{run_suffix}_step*.npz")

    # Xóa checkpoint và file ứng viên cũ của cùng nhãn chạy để tránh lẫn với đợt train trước
    for stale_path in [best_npz_path, runner_up_npz_path] + glob.glob(cand_glob):
        if os.path.exists(stale_path):
            os.remove(stale_path)
            print(f"[Cleanup] Đã xóa file cũ {stale_path}.")

    def _fmt_eval_line(step_label, summary):
        settle_str = f"{summary['mean_settle_s']:.2f}s" if summary['mean_settle_s'] is not None else "N/A"
        return (
            f"[{step_label}] Ngã: {summary['fall_count']}/{summary['total_runs']} ({summary['fall_rate_pct']:.1f}%) | "
            f"Giữ VT: {summary['hold_count']}/{summary['total_runs']} | "
            f"Final X: {summary['mean_final_x_m']*100:.2f}cm | Settle: {settle_str}"
        )

    # Callback giữ lại Top-K checkpoint đứng đầu theo bài test nhanh 6s (mặc định K=3)
    class TopKQuickCheckpointCallback(BaseCallback):
        def __init__(self, eval_freq=20000, top_k=3, verbose=1):
            super().__init__(verbose)
            self.eval_freq = max(1, eval_freq // n_envs)
            self.top_k = max(1, top_k)
            self.top_entries = []

        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq == 0:
                numpy_policy.load_from_sb3(self.model, odd_symmetric=use_odd_sym)
                summary = evaluator.evaluate(numpy_policy, quick_eval=True)
                rank_key = (summary['ranking_key'], self.num_timesteps)

                if self.verbose > 0:
                    print("\n" + _fmt_eval_line(f"Eval @ Step {self.num_timesteps}", summary))

                qualifies = (len(self.top_entries) < self.top_k) or (rank_key > self.top_entries[-1]['rank_key'])
                if qualifies:
                    cand_path = os.path.join(ckpt_dir, f"cand{run_suffix}_step{self.num_timesteps}.npz")
                    numpy_policy.save_weights(cand_path)
                    # Xóa entry trùng step nếu có rồi chèn lại
                    self.top_entries = [e for e in self.top_entries if e['step'] != self.num_timesteps]
                    self.top_entries.append({
                        'label': f"step_{self.num_timesteps} (quick)",
                        'step': int(self.num_timesteps),
                        'path': cand_path,
                        'rank_key': rank_key,
                        'quick_summary': summary,
                    })
                    self.top_entries.sort(key=lambda e: e['rank_key'], reverse=True)

                    while len(self.top_entries) > self.top_k:
                        evicted = self.top_entries.pop()
                        if evicted['path'] not in {e['path'] for e in self.top_entries} and os.path.exists(evicted['path']):
                            os.remove(evicted['path'])

                    if self.verbose > 0:
                        steps_str = ", ".join(f"{e['step']} (ngã {e['quick_summary']['fall_count']})" for e in self.top_entries)
                        print(f"  >>> [TOP-{self.top_k} QUICK] Danh sách giữ lại: [{steps_str}]\n")

            return True

    eval_callback = TopKQuickCheckpointCallback(
        eval_freq=rl_cfg.get('eval_freq', 20000),
        top_k=args.top_k,
        verbose=1,
    )

    print(f"[Training] Bắt đầu quá trình huấn luyện {total_timesteps} timesteps (w_pos={config['reward_weights']['w_pos']}, top_k={args.top_k})...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Lưu checkpoint cuối đợt train vào final_policy & đối chiếu SB3 <-> NumPy <-> C Header (gồm ca 1 nhánh chạm trần)
    final_step = int(model.num_timesteps)
    model.save(final_sb3_path)
    numpy_policy.load_from_sb3(model, odd_symmetric=use_odd_sym)
    numpy_policy.save_weights(final_npz_path)
    export_sb3_c_header(numpy_policy, final_hdr_path)
    print(f"\n[Checkpoint] Đã lưu model cuối đợt train (step {final_step}) sang: {final_npz_path} & {final_hdr_path}")
    verify_policy_parity(numpy_policy, f"trained_final_{run_label}", sb3_model=model, header_path=final_hdr_path)

    final_quick = evaluator.evaluate(numpy_policy, quick_eval=True)
    print(_fmt_eval_line(f"Eval @ Final ({final_step})", final_quick))

    # Gom ứng viên (Top-K quick + checkpoint cuối) để đưa vào sàng lọc 2 bước
    def _load_policy(weights_path):
        pol = NumpyMLPPolicy(input_dim=obs_dim, hidden_dims=(net_arch[0], net_arch[1]), output_dim=1)
        pol.load_weights(weights_path)
        return pol

    candidates = []
    merged_final = False
    for entry in eval_callback.top_entries:
        if entry['step'] == final_step:
            candidates.append({
                'label': f"step_{entry['step']} (quick+final)",
                'step': entry['step'],
                'path': final_npz_path,
                'policy': _load_policy(final_npz_path),
                'quick_summary': final_quick,
            })
            merged_final = True
        elif os.path.exists(entry['path']):
            candidates.append({
                'label': entry['label'],
                'step': entry['step'],
                'path': entry['path'],
                'policy': _load_policy(entry['path']),
                'quick_summary': entry['quick_summary'],
            })

    if not merged_final:
        candidates.append({
            'label': f"step_{final_step} (final)",
            'step': final_step,
            'path': final_npz_path,
            'policy': _load_policy(final_npz_path),
            'quick_summary': final_quick,
        })

    # GIAI ĐOẠN 2 (22 ca không nhiễu) -> GIAI ĐOẠN 3 (Top 2 trên đủ 98 ca)
    print("\n" + "=" * 78)
    print(f"   CHỌN CHECKPOINT 2 BƯỚC ({len(candidates)} ỨNG VIÊN -> 22 CA KHÔNG NHIỄU -> TOP 2 / 98 CA)")
    print("=" * 78)
    screened_records, full_records = evaluator.select_top_candidates(candidates, top_full=2)
    two_stage_report = evaluator.format_two_stage_report(
        screened_records, full_records, deploy_gate_mode=args.deploy_gate
    )
    print("\n" + two_stage_report + "\n")

    winner_rec = full_records[0]
    summary = winner_rec['full_summary']
    numpy_policy = winner_rec['policy']
    numpy_policy.save_weights(best_npz_path)
    print(f"[Final Selection] #1 theo 98 ca: {winner_rec['label']} -> {best_npz_path}")

    if len(full_records) >= 2:
        runner_up_rec = full_records[1]
        runner_up_rec['policy'].save_weights(runner_up_npz_path)
        runner_up_hdr_path = os.path.join(hdr_dir, f"{stem}_{run_label}_runner_up{ext}")
        export_sb3_c_header(runner_up_rec['policy'], runner_up_hdr_path)
        print(f"[Runner-Up] #2 theo 98 ca: {runner_up_rec['label']} -> {runner_up_npz_path} & {runner_up_hdr_path}")

    # Dọn file ứng viên tạm (chỉ giữ lại best, runner_up và final)
    for cand_tmp in glob.glob(cand_glob):
        if os.path.exists(cand_tmp):
            os.remove(cand_tmp)

    # Báo cáo chi tiết của bản #1
    print("\n" + "=" * 78)
    print(f"      BÁO CÁO CHI TIẾT CHECKPOINT #1: {winner_rec['label']} (98 KỊCH BẢN) ")
    print("=" * 78)
    winner_report = evaluator.format_report(summary, deploy_gate_mode=args.deploy_gate)
    print("\n" + winner_report + "\n")

    report_path = os.path.join("outputs", "reports", f"eval_report_{run_label}.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(two_stage_report + "\n\n---\n\n" + winner_report + "\n")
    print(f"[Report] Đã lưu báo cáo chọn lọc 2 bước & đánh giá 98 ca sang: {report_path}")

    # XUẤT C HEADER SAU KHI ĐÃ ĐÁNH GIÁ VÀ KIỂM TRA CỔNG AN TOÀN STM32
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
