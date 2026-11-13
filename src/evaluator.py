import math
import numpy as np
from src.environment import SelfBalancingGymEnv

class PolicyEvaluator:
    """
    Bộ đánh giá độc lập toàn diện cho Controller (9D PPO RL):
      - Ma trận chuẩn 10s (90 ca): Góc [-5°, -3°, -1°, +1°, +3°, +5°] x Xung [0, ±1, ±2 N] x Pin [7.2, 7.4, 8.4 V]
      - Các ca dài 20–30s (6 ca): Bắt lỗi trôi vị trí chậm sau khi góc đã cân bằng.
      - Tiêu chí giữ vị trí ĐẠT: Không ngã và giữ |theta| <= 0.5° và |x| <= 2.0 cm trong 2.0 giây cuối.
      - Xếp hạng checkpoint: Ít ngã nhất -> Nhiều ca giữ vị trí đạt nhất -> Sai số vị trí cuối thấp nhất.
    """

    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config
        self.test_angles = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
        self.test_kicks = [0.0, 1.0, -1.0, 2.0, -2.0]
        self.test_vbats = [7.2, 7.4, 8.4]
        self.duration = 10.0
        self.hold_window_s = 2.0
        self.hold_angle_deg = 0.50
        self.hold_pos_m = 0.020
        # Các ca kiểm tra dài 20-30s để phát hiện trôi vị trí chậm
        self.long_scenarios = [
            (-5.0, 0.0, 7.4, 25.0),
            (+5.0, 0.0, 7.4, 25.0),
            (-3.0, -1.0, 7.4, 20.0),
            (+3.0, +1.0, 7.4, 20.0),
            (-1.0, 0.0, 7.2, 30.0),
            (+1.0, 0.0, 8.4, 30.0),
        ]

    def evaluate_single_run(self, policy, init_theta_deg, kick_n, vbat_v, duration=10.0, eval_seed=1001):
        env = SelfBalancingGymEnv(self.config, seed=eval_seed)
        steps = int(duration / env.Ts)

        obs, _ = env.reset(seed=eval_seed, options={
            'init_theta_deg': init_theta_deg,
            'init_x': 0.0,
            'v_bat': vbat_v,
            'fixed_physics': True
        })

        t_hist = []
        theta_hist = []
        x_hist = []
        pwm_hist = []
        fallen = False

        for step_i in range(steps):
            t = step_i * env.Ts
            dist_force = kick_n if (2.0 <= t <= 2.10) else 0.0

            if hasattr(policy, 'forward'):
                action = policy.forward(obs)
            elif hasattr(policy, 'predict'):
                action, _ = policy.predict(obs, deterministic=True)
            elif callable(policy):
                action = policy(obs)
            else:
                raise ValueError("Policy không hỗ trợ phương thức forward hoặc predict.")

            obs, reward, terminated, truncated, info = env.step(action, disturbance=dist_force)

            t_hist.append(t)
            theta_hist.append(info['theta_deg'])
            x_hist.append(info['x_pos'])
            pwm_hist.append(info['applied_pwm'])

            if terminated:
                fallen = True
                break

        survive_time = t_hist[-1] if fallen else duration
        max_deg = float(np.max(np.abs(theta_hist)))
        max_x = float(np.max(np.abs(x_hist)))
        final_deg = float(abs(theta_hist[-1]))
        final_x = float(abs(x_hist[-1]))
        rms_pwm = float(np.sqrt(np.mean(np.square(pwm_hist))))

        # Chỉ tính là giữ vị trí ĐẠT khi xe không ngã và duy trì đồng thời
        # |theta| <= 0.5° và |x| <= 2.0 cm trong trọn vẹn 2.0 giây cuối.
        hold_steps = max(1, int(round(min(self.hold_window_s, duration) / env.Ts)))
        position_held = False
        settling_time = None
        if not fallen and len(t_hist) >= hold_steps:
            in_bounds = [
                (abs(th) <= self.hold_angle_deg and abs(pos) <= self.hold_pos_m)
                for th, pos in zip(theta_hist, x_hist)
            ]
            if all(in_bounds[-hold_steps:]):
                position_held = True
                last_violation_idx = -1
                for i in range(len(in_bounds) - 1, -1, -1):
                    if not in_bounds[i]:
                        last_violation_idx = i
                        break
                settling_time = 0.0 if last_violation_idx == -1 else float(t_hist[last_violation_idx + 1])

        return {
            'init_theta': init_theta_deg,
            'kick': kick_n,
            'vbat': vbat_v,
            'duration': duration,
            'is_long': duration > self.duration + 1e-6,
            'fallen': fallen,
            'position_held': position_held,
            'survive_time': survive_time,
            'max_deg': max_deg,
            'final_deg': final_deg,
            'max_x': max_x,
            'final_x': final_x,
            'settling_time': settling_time,
            'rms_pwm': rms_pwm
        }

    def evaluate(self, policy, quick_eval=False, include_long=True):
        """
        Đánh giá độc lập với bộ Seed cố định (Independent Evaluation Suite):
          - quick_eval=True: Bộ test nhanh trong khi huấn luyện (18 ca 6s)
          - quick_eval=False: Bộ test đầy đủ (90 ca 10s + 6 ca dài 20–30s để bắt lỗi trôi chậm)
        """
        if quick_eval:
            angles = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
            kicks = [0.0, 1.5, -1.5]
            vbats = [7.4]
            duration = 6.0
        else:
            angles = self.test_angles
            kicks = self.test_kicks
            vbats = self.test_vbats
            duration = self.duration

        results = []
        test_case_id = 0
        for v in vbats:
            for k in kicks:
                for th in angles:
                    test_case_id += 1
                    case_seed = 5000 + test_case_id
                    res = self.evaluate_single_run(policy, th, k, v, duration=duration, eval_seed=case_seed)
                    results.append(res)

        if not quick_eval and include_long:
            for idx, (th, k, v, long_dur) in enumerate(self.long_scenarios, start=1):
                case_seed = 6000 + idx
                res = self.evaluate_single_run(policy, th, k, v, duration=long_dur, eval_seed=case_seed)
                results.append(res)

        total_runs = len(results)
        fall_count = sum(1 for r in results if r['fallen'])
        fall_rate = (fall_count / total_runs) * 100.0
        hold_count = sum(1 for r in results if r['position_held'])
        hold_rate = (hold_count / total_runs) * 100.0

        survived_runs = [r for r in results if not r['fallen']]
        held_runs = [r for r in results if r['position_held']]
        mean_survive_s = float(np.mean([r['survive_time'] for r in results])) if results else 0.0

        if survived_runs:
            mean_max_x = float(np.mean([r['max_x'] for r in survived_runs]))
            max_x_worst = float(np.max([r['max_x'] for r in survived_runs]))
            mean_final_x = float(np.mean([r['final_x'] for r in survived_runs]))
            max_final_x = float(np.max([r['final_x'] for r in survived_runs]))
            mean_rms_pwm = float(np.mean([r['rms_pwm'] for r in survived_runs]))
        else:
            mean_max_x = 1.0
            max_x_worst = 1.0
            mean_final_x = 1.0
            max_final_x = 1.0
            mean_rms_pwm = 999.0

        mean_settle = float(np.mean([r['settling_time'] for r in held_runs])) if held_runs else None

        # Khóa xếp hạng ưu tiên nghiêm ngặt:
        # 1) Ít ngã nhất  2) Nhiều ca giữ vị trí đạt nhất  3) Sai số vị trí cuối thấp nhất
        ranking_key = (
            -fall_count,
            hold_count,
            -round(mean_final_x, 6),
            round(mean_survive_s, 4),
            -round(mean_rms_pwm, 2),
        )

        summary = {
            'total_runs': total_runs,
            'fall_count': fall_count,
            'fall_rate_pct': fall_rate,
            'hold_count': hold_count,
            'hold_rate_pct': hold_rate,
            'mean_max_x_m': mean_max_x,
            'max_x_worst_m': max_x_worst,
            'mean_final_x_m': mean_final_x,
            'max_final_x_m': max_final_x,
            'mean_settle_s': mean_settle,
            'mean_survive_s': mean_survive_s,
            'mean_rms_pwm': mean_rms_pwm,
            'ranking_key': ranking_key,
            'detailed_results': results
        }

        return summary

    def format_report(self, summary):
        r = summary
        std_count = sum(1 for res in r['detailed_results'] if not res.get('is_long', False))
        long_results = [res for res in r['detailed_results'] if res.get('is_long', False)]
        settle_str = f"{r['mean_settle_s']:.2f} s" if r['mean_settle_s'] is not None else "Không đạt"

        lines = []
        lines.append("### BÁO CÁO ĐÁNH GIÁ ĐỘC LẬP CHÙM BÀI TEST TOÀN DIỆN")
        lines.append(
            f"- **Tổng số ca kiểm thử:** {r['total_runs']} kịch bản "
            f"({std_count} ca {self.duration:.0f}s + {len(long_results)} ca dài 20–30s, Seed cố định)"
        )
        lines.append(f"- **Tỷ lệ ngã (Fall Rate):** {r['fall_rate_pct']:.1f}% ({r['fall_count']}/{r['total_runs']} bài ngã)")
        lines.append(
            f"- **Giữ vị trí đạt (|θ| ≤ {self.hold_angle_deg:.1f}°, |x| ≤ {self.hold_pos_m*100:.0f} cm trong {self.hold_window_s:.0f}s cuối):** "
            f"{r['hold_count']}/{r['total_runs']} ca ({r['hold_rate_pct']:.1f}%)"
        )
        lines.append(
            f"- **Sai số vị trí cuối (các ca sống sót):** TB {r['mean_final_x_m']*100:.2f} cm "
            f"(Xấu nhất: {r['max_final_x_m']*100:.2f} cm)"
        )
        lines.append(f"- **Độ lệch vị trí cực đại TB:** {r['mean_max_x_m']*100:.2f} cm (Xấu nhất: {r['max_x_worst_m']*100:.2f} cm)")
        lines.append(f"- **Thời gian ổn định TB (trên ca đạt):** {settle_str}")
        lines.append(f"- **RMS PWM trung bình:** {r['mean_rms_pwm']:.1f} / 999")

        lines.append("\n#### Các kịch bản 10s tại điện áp danh định 7.4V")
        lines.append("| Góc đầu | Nhiễu (N) | Pin | Sống sót | Max Theta | Max X | Final X | Giữ VT (2s cuối) | Settling | RMS PWM |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        sample_results = [
            res for res in r['detailed_results']
            if (not res.get('is_long', False)) and res['vbat'] == 7.4
        ]
        for res in sample_results:
            status = f"{res['survive_time']:.1f}s" if not res['fallen'] else f"NGÃ ({res['survive_time']:.2f}s)"
            hold_txt = "ĐẠT" if res['position_held'] else "KHÔNG ĐẠT"
            settle_cell = f"{res['settling_time']:.2f}s" if res['settling_time'] is not None else "KHÔNG ĐẠT"
            lines.append(
                f"| {res['init_theta']:+4.1f}° | {res['kick']:+4.1f} N | {res['vbat']:.1f}V | "
                f"{status} | {res['max_deg']:.2f}° | {res['max_x']*100:.1f} cm | "
                f"{res['final_x']*100:.2f} cm | {hold_txt} | {settle_cell} | {res['rms_pwm']:.0f} |"
            )

        if long_results:
            lines.append("\n#### Các kịch bản dài 20–30s (Kiểm tra trôi vị trí chậm)")
            lines.append("| Thời lượng | Góc đầu | Nhiễu (N) | Pin | Sống sót | Max Theta | Max X | Final X | Giữ VT (2s cuối) | Settling | RMS PWM |")
            lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
            for res in long_results:
                status = f"{res['survive_time']:.1f}s" if not res['fallen'] else f"NGÃ ({res['survive_time']:.2f}s)"
                hold_txt = "ĐẠT" if res['position_held'] else "KHÔNG ĐẠT"
                settle_cell = f"{res['settling_time']:.2f}s" if res['settling_time'] is not None else "KHÔNG ĐẠT"
                lines.append(
                    f"| {res['duration']:.0f}s | {res['init_theta']:+4.1f}° | {res['kick']:+4.1f} N | {res['vbat']:.1f}V | "
                    f"{status} | {res['max_deg']:.2f}° | {res['max_x']*100:.1f} cm | "
                    f"{res['final_x']*100:.2f} cm | {hold_txt} | {settle_cell} | {res['rms_pwm']:.0f} |"
                )

        return "\n".join(lines)
