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

def _load_policy(ckpt_path, obs_dim, net_arch, allow_random=False, force_symmetric=False):
    policy = NumpyMLPPolicy(input_dim=obs_dim, hidden_dims=(net_arch[0], net_arch[1]), output_dim=1)
    if os.path.exists(ckpt_path):
        policy.load_weights(ckpt_path)
        if force_symmetric:
            policy.odd_symmetric = True
        print(f"[Model] Đã nạp thành công weights từ: {ckpt_path} (input_dim={policy.input_dim}, odd_symmetric={policy.odd_symmetric})")
    elif allow_random:
        if force_symmetric:
            policy.odd_symmetric = True
        print(f"[Warning] Chạy với trọng số ngẫu nhiên (--allow-random, odd_symmetric={policy.odd_symmetric}).")
    else:
        sys.exit(f"[Error] Không tìm thấy checkpoint: {ckpt_path}. Vui lòng huấn luyện hoặc chỉ định --weights.")
    return policy

def run_inference():
    parser = argparse.ArgumentParser(description="Test & Evaluate RL Self-Balancing Robot Controller")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config")
    parser.add_argument("--weights", type=str, default=None, help="Custom weights .npz path")
    parser.add_argument("--compare", nargs="+", default=None, help="Compare multiple .npz checkpoints on the exact same evaluation suite")
    parser.add_argument("--export-winner", action="store_true", help="Save A/B winner to ab_winner_policy.npz; update STM32 header only if deploy gate passes")
    parser.add_argument("--deploy-gate", type=str, default="real_robot", choices=["real_robot", "no_kick", "all"], help="Criteria required before overwriting canonical STM32 header")
    parser.add_argument("--force-export-stm32", action="store_true", help="Force overwriting canonical best_policy.npz and STM32 header even if deploy gate fails")
    parser.add_argument("--output-figure", type=str, default=None, help="Custom output PNG path for trajectory plot")
    parser.add_argument("--report-out", type=str, default=None, help="Optional Markdown file path to save evaluation/comparison report")
    parser.add_argument("--kick", type=float, default=0.0, help="External disturbance force in Newtons (default: 0.0)")
    parser.add_argument("--angle", type=float, default=None, help="Specific initial pitch angle in degrees (e.g. -3.0 or +5.0)")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration in seconds (default: 10.0s)")
    parser.add_argument("--eval-matrix", action="store_true", help="Run full evaluation matrix across angles, kicks, battery levels, and 20-30s drift tests")
    parser.add_argument("--symmetric", action="store_true", help="Force odd-symmetric inference 0.5*(pi(obs)-pi(-obs)) for loaded weights")
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
        print("      SO SÁNH CÁC CHECKPOINT TRÊN CÙNG BỘ TEST ĐỘC LẬP (98 KỊCH BẢN)         ")
        print("      90 ca 10s + 2 ca 0° (±1.5 N) + 6 ca chạy dài 20–30s                     ")
        print("      Xếp hạng: Ít ngã nhất -> Giữ VT (2s cuối) nhiều nhất -> Final X nhỏ     ")
        print("==============================================================================")
        compared = []
        for path in args.compare:
            pol = _load_policy(path, obs_dim, net_arch, allow_random=False, force_symmetric=args.symmetric)
            summ = evaluator.evaluate(pol, quick_eval=False)
            compared.append((path, pol, summ))

        compared.sort(key=lambda item: item[2]['ranking_key'], reverse=True)

        md_sections = []
        md_sections.append("# BÁO CÁO SO SÁNH CHECKPOINT A/B (98 KỊCH BẢN ĐỘC LẬP)")
        md_sections.append(f"- **Deploy Gate kiểm tra STM32:** `{args.deploy_gate}`")
        md_sections.append("\n## 1. Bảng Xếp Hạng Tổng Thể (98 kịch bản)")
        overall_header = [
            "| Hạng | Checkpoint | Ngã | Giữ VT (2s cuối) | Max X (TB / Xấu nhất) | Final X (TB / Xấu nhất) | Settle TB (đạt) | RMS PWM | Cổng STM32 |",
            "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        overall_rows = []
        for rank, (path, _, s) in enumerate(compared, start=1):
            settle_str = f"{s['mean_settle_s']:.2f}s" if s['mean_settle_s'] is not None else "Không đạt"
            gate_ok, _ = evaluator.check_deploy_gate(s, gate_mode=args.deploy_gate)
            gate_txt = "ĐẠT" if gate_ok else "CHẶN"
            overall_rows.append(
                f"| #{rank} | {os.path.basename(path)} | {s['fall_count']}/{s['total_runs']} ({s['fall_rate_pct']:.1f}%) | "
                f"{s['hold_count']}/{s['total_runs']} ({s['hold_rate_pct']:.1f}%) | "
                f"{evaluator._fmt_cm_pair(s, 'mean_max_x_m', 'max_x_worst_m')} | "
                f"{evaluator._fmt_cm_pair(s, 'mean_final_x_m', 'max_final_x_m')} | "
                f"{settle_str} | {s['mean_rms_pwm']:.1f} | {gate_txt} |"
            )
        overall_block = "\n".join(overall_header + overall_rows)
        print("\n" + overall_block)
        md_sections.append(overall_block)

        md_sections.append("\n## 2. So Sánh Theo Nhóm Kịch Bản (Không nhiễu / Có nhiễu / Chạy dài 20–30s)")
        group_header = [
            "| Checkpoint | Nhóm kịch bản | Ngã | Giữ VT (2s cuối) | Max X (TB / Xấu nhất) | Final X (TB / Xấu nhất) | Settle TB |",
            "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
        ]
        group_rows = []
        group_defs = [
            ('no_kick', 'Không nhiễu 10s (0 N)'),
            ('with_kick', 'Có nhiễu 10s (gồm 0° ±1.5 N)'),
            ('long_drift', 'Chạy dài 20–30s'),
        ]
        for path, _, s in compared:
            gs_map = s.get('group_summaries', {})
            for gkey, gtitle in group_defs:
                gs = gs_map.get(gkey)
                if not gs or gs['total_runs'] == 0:
                    continue
                g_settle = f"{gs['mean_settle_s']:.2f}s" if gs['mean_settle_s'] is not None else "Không đạt"
                group_rows.append(
                    f"| {os.path.basename(path)} | {gtitle} | "
                    f"{gs['fall_count']}/{gs['total_runs']} ({gs['fall_rate_pct']:.1f}%) | "
                    f"{gs['hold_count']}/{gs['total_runs']} ({gs['hold_rate_pct']:.1f}%) | "
                    f"{evaluator._fmt_cm_pair(gs, 'mean_max_x_m', 'max_x_worst_m')} | "
                    f"{evaluator._fmt_cm_pair(gs, 'mean_final_x_m', 'max_final_x_m')} | {g_settle} |"
                )
        group_block = "\n".join(group_header + group_rows)
        print("\n--- PHÂN TÍCH THEO NHÓM KỊCH BẢN ---")
        print(group_block)
        md_sections.append(group_block)

        md_sections.append("\n## 3. Đối Chiếu Riêng 2 Ca Cú Đẩy Từ 0° (±1.5 N @ 7.4V, 10s)")
        zero_header = [
            "| Checkpoint | Góc đầu | Nhiễu | Sống sót | Max Theta | Max X | Final X | Giữ VT (2s cuối) | Settling |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        zero_rows = []
        for path, _, s in compared:
            z_runs = [
                r for r in s['detailed_results']
                if (not r.get('is_long')) and abs(r['init_theta']) < 1e-6 and abs(abs(r['kick']) - 1.5) < 1e-6
            ]
            for zr in z_runs:
                status = f"{zr['survive_time']:.1f}s" if not zr['fallen'] else f"NGÃ ({zr['survive_time']:.2f}s)"
                hold_txt = "ĐẠT" if zr['position_held'] else "KHÔNG ĐẠT"
                settle_cell = f"{zr['settling_time']:.2f}s" if zr['settling_time'] is not None else "KHÔNG ĐẠT"
                zero_rows.append(
                    f"| {os.path.basename(path)} | {zr['init_theta']:+4.1f}° | {zr['kick']:+4.1f} N | "
                    f"{status} | {zr['max_deg']:.2f}° | {zr['max_x']*100:.1f} cm | "
                    f"{zr['final_x']*100:.2f} cm | {hold_txt} | {settle_cell} |"
                )
        zero_block = "\n".join(zero_header + zero_rows)
        print("\n--- KẾT QUẢ 2 CA 0° (±1.5 N) ---")
        print(zero_block)
        md_sections.append(zero_block)

        best_path, best_pol, best_summ = compared[0]
        print(f"\n[Winner] Bản xếp hạng #1: {best_path}")
        winner_report = evaluator.format_report(best_summ, deploy_gate_mode=args.deploy_gate)
        print("\n" + winner_report + "\n")

        for rank, (path, _, s) in enumerate(compared, start=1):
            md_sections.append(f"\n---\n## Chi Tiết Hạng #{rank}: `{path}`")
            md_sections.append(evaluator.format_report(s, deploy_gate_mode=args.deploy_gate))

        report_out = args.report_out if args.report_out else os.path.join("outputs", "reports", "ab_comparison_report.md")
        os.makedirs(os.path.dirname(os.path.abspath(report_out)), exist_ok=True)
        with open(report_out, "w", encoding="utf-8") as rf:
            rf.write("\n\n".join(md_sections) + "\n")
        print(f"[Report] Đã lưu đầy đủ bảng điểm A/B sang: {report_out}")

        if args.export_winner:
            ckpt_dir = config['export']['checkpoint_dir']
            hdr_dir = config['export']['header_output_dir']
            hdr_name = config['export'].get('header_filename', 'rl_sb3_stm32.h')
            stem, ext = os.path.splitext(hdr_name)

            ab_winner_npz = os.path.join(ckpt_dir, "ab_winner_policy.npz")
            ab_winner_hdr = os.path.join(hdr_dir, f"{stem}_ab_winner{ext}")
            best_pol.save_weights(ab_winner_npz)
            export_sb3_c_header(best_pol, ab_winner_hdr)
            print(f"[Winner Saved] Đã lưu bản thắng A/B dưới tên riêng: {ab_winner_npz} & {ab_winner_hdr}")

            passed, reasons = evaluator.check_deploy_gate(best_summ, gate_mode=args.deploy_gate)
            if passed or args.force_export_stm32:
                if not passed and args.force_export_stm32:
                    print(f"[Force Export] CẢNH BÁO: Ép ghi đè header dù cổng '{args.deploy_gate}' CHƯA ĐẠT ({'; '.join(reasons)})")
                best_dst = os.path.join(ckpt_dir, "best_policy.npz")
                best_pol.save_weights(best_dst)
                export_sb3_c_header(best_pol, os.path.join(hdr_dir, hdr_name))
                stm32_inc = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Project_self", "Core", "Inc", hdr_name))
                if os.path.isdir(os.path.dirname(stm32_inc)):
                    export_sb3_c_header(best_pol, stm32_inc)
                    print(f"[Sync] Đã đồng bộ winner đạt chuẩn sang STM32 Header: {stm32_inc}")
            else:
                print(
                    f"[Gate Blocked] Bản thắng A/B CHƯA ĐẠT tiêu chí xuất STM32 (gate='{args.deploy_gate}'):\n"
                    + "\n".join(f"  {r}" for r in reasons)
                    + "\n  -> Đã giữ nguyên best_policy.npz, rl_sb3_stm32.h và Project_self/Core/Inc/rl_sb3_stm32.h."
                )
        return

    ckpt_path = args.weights if args.weights is not None else os.path.join(config['export']['checkpoint_dir'], "best_policy.npz")
    policy = _load_policy(ckpt_path, obs_dim, net_arch, allow_random=args.allow_random, force_symmetric=args.symmetric)

    if args.eval_matrix:
        print("==============================================================================")
        print("      CHẠY MA TRẬN ĐÁNH GIÁ ĐỘC LẬP TOÀN DIỆN CHO CONTROLLER (9D PPO RL)     ")
        print("      90 ca 10s + 2 ca 0° (±1.5 N) + 6 ca dài 20–30s (Tổng 98 kịch bản)       ")
        print("==============================================================================")
        summary = evaluator.evaluate(policy, quick_eval=False)
        report = evaluator.format_report(summary, deploy_gate_mode=args.deploy_gate)
        print("\n" + report + "\n")
        if args.report_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
            with open(args.report_out, "w", encoding="utf-8") as rf:
                rf.write(report + "\n")
            print(f"[Report] Đã lưu báo cáo đánh giá sang: {args.report_out}")
        return

    # Chạy đơn kịch bản và vẽ đồ thị
    print("==============================================================================")
    print("      CHẠY MÔ PHỎNG ĐỒNG BỘ CAO ĐỘ (SIM-TO-REAL RESPONSE)                     ")
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
    pwm_before_deadband_hist = []
    effective_pwm_hist = []
    fallen = False

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
        pwm_before_deadband_hist.append(info['u_ppo'])
        effective_pwm_hist.append(info['effective_pwm'])

        if terminated:
            fallen = True
            print(f"[Sim] Robot ngã tại t = {t:.2f}s (Góc = {info['theta_deg']:.1f}°)")
            break

    fig_dir = config['export']['figures_dir']
    fig_path = args.output_figure if args.output_figure else os.path.join(fig_dir, "rl_inference_test.png")
    fig_path_abs = os.path.abspath(fig_path)
    os.makedirs(os.path.dirname(fig_path_abs), exist_ok=True)
    plot_simulation_trajectory(
        t_hist, theta_hist, x_hist, pwm_hist, fig_path_abs,
        pwm_before_deadband=pwm_before_deadband_hist,
        effective_pwm=effective_pwm_hist,
    )

    avg_lat = float(np.mean(latencies))
    max_lat = float(np.max(latencies))
    max_deg = float(np.max(np.abs(theta_hist)))
    max_x = float(np.max(np.abs(x_hist)))
    final_x = float(abs(x_hist[-1]))
    final_deg = float(abs(theta_hist[-1]))
    rms_pwm = float(np.sqrt(np.mean(np.square(pwm_hist))))
    survive_s = t_hist[-1] if fallen else args.duration
    pos_held, settle_t = evaluator.compute_hold_metrics(t_hist, theta_hist, x_hist, fallen, args.duration)
    hold_str = f"ĐẠT (ổn định từ {settle_t:.2f}s)" if pos_held else "KHÔNG ĐẠT"

    print("\n------------------- KẾT QUẢ HIỆU NĂNG -------------------")
    print(f"  Thời gian cân bằng sống sót : {survive_s:.2f} / {args.duration:.2f} s")
    print(f"  Độ lệch góc lớn nhất        : {max_deg:.2f}° (Cuối: {final_deg:.2f}° | Ngưỡng ngã: {env.cfg['gym_env']['fall_angle_deg']}°)")
    print(f"  Độ lệch vị trí lớn nhất     : {max_x*100:.2f} cm (Cuối: {final_x*100:.2f} cm)")
    print(f"  Giữ vị trí 2s cuối (≤2 cm)  : {hold_str}")
    print(f"  RMS PWM                     : {rms_pwm:.1f} / 999")
    print(f"  Độ trễ suy luận trung bình  : {avg_lat:.2f} µs (Cực đại: {max_lat:.2f} µs)")
    print(f"  Thời gian tính chu kỳ 200Hz : 5000 µs -> Tận dụng ~1% CPU của STM32!")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    run_inference()
