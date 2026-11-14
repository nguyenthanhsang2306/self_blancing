import math
import numpy as np
from src.environment import SelfBalancingGymEnv

class PolicyEvaluator:
    """
    Bộ đánh giá độc lập toàn diện cho Controller (9D PPO RL):
      - Ma trận chuẩn 10s (90 ca): Góc [-5°, -3°, -1°, +1°, +3°, +5°] x Xung [0, ±1, ±2 N] x Pin [7.2, 7.4, 8.4 V]
      - 2 ca nhiễu từ góc 0° (10s @ 7.4V): Góc 0.0° x Xung [+1.5, -1.5 N]
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
        # 2 ca bắt đầu từ 0° với xung ±1.5 N để đối chiếu trực tiếp với đồ thị thực nghiệm
        self.zero_deg_kick_scenarios = [
            (0.0, +1.5, 7.4, 10.0),
            (0.0, -1.5, 7.4, 10.0),
        ]
        # Các ca kiểm tra dài 20-30s để phát hiện trôi vị trí chậm
        self.long_scenarios = [
            (-5.0, 0.0, 7.4, 25.0),
            (+5.0, 0.0, 7.4, 25.0),
            (-3.0, -1.0, 7.4, 20.0),
            (+3.0, +1.0, 7.4, 20.0),
            (-1.0, 0.0, 7.2, 30.0),
            (+1.0, 0.0, 8.4, 30.0),
        ]

    def compute_hold_metrics(self, t_hist, theta_hist, x_hist, fallen, duration):
        """
        Chỉ tính là giữ vị trí ĐẠT khi xe không ngã và duy trì đồng thời
        |theta| <= 0.5° và |x| <= 2.0 cm trong trọn vẹn 2.0 giây cuối.
        """
        ts = float(self.config.get('system', {}).get('Ts', 0.005))
        hold_steps = max(1, int(round(min(self.hold_window_s, duration) / ts)))
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
        return position_held, settling_time

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

        position_held, settling_time = self.compute_hold_metrics(t_hist, theta_hist, x_hist, fallen, duration)

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

    def _summarize_subset(self, runs):
        total_runs = len(runs)
        fall_count = sum(1 for r in runs if r['fallen'])
        fall_rate = (fall_count / total_runs) * 100.0 if total_runs else 0.0
        hold_count = sum(1 for r in runs if r['position_held'])
        hold_rate = (hold_count / total_runs) * 100.0 if total_runs else 0.0

        survived_runs = [r for r in runs if not r['fallen']]
        held_runs = [r for r in runs if r['position_held']]
        mean_survive_s = float(np.mean([r['survive_time'] for r in runs])) if runs else 0.0

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

        return {
            'total_runs': total_runs,
            'survived_count': len(survived_runs),
            'has_survivors': bool(survived_runs),
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
        }

    def _build_summary(self, results):
        """Tổng hợp chỉ số tổng thể, theo nhóm và khóa xếp hạng từ danh sách kết quả ca."""
        overall = self._summarize_subset(results)

        no_kick_runs = [r for r in results if (not r['is_long']) and abs(r['kick']) < 1e-6]
        with_kick_runs = [r for r in results if (not r['is_long']) and abs(r['kick']) >= 1e-6]
        long_drift_runs = [r for r in results if r['is_long']]

        group_summaries = {
            'no_kick': self._summarize_subset(no_kick_runs),
            'with_kick': self._summarize_subset(with_kick_runs),
            'long_drift': self._summarize_subset(long_drift_runs),
        }

        # Khóa xếp hạng ưu tiên nghiêm ngặt:
        # 1) Ít ngã nhất  2) Nhiều ca giữ vị trí đạt nhất  3) Sai số vị trí cuối thấp nhất
        ranking_key = (
            -overall['fall_count'],
            overall['hold_count'],
            -round(overall['mean_final_x_m'], 6),
            round(overall['mean_survive_s'], 4),
            -round(overall['mean_rms_pwm'], 2),
        )

        return {
            **overall,
            'group_summaries': group_summaries,
            'ranking_key': ranking_key,
            'detailed_results': results,
        }

    def evaluate(self, policy, quick_eval=False, include_long=True):
        """
        Đánh giá độc lập với bộ Seed cố định (Independent Evaluation Suite):
          - quick_eval=True: Bộ test nhanh trong khi huấn luyện (18 ca 6s)
          - quick_eval=False: Bộ test đầy đủ 98 ca (90 ca chuẩn 10s + 2 ca 0° ±1.5N + 6 ca dài 20–30s)
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

        if not quick_eval:
            for idx, (th, k, v, dur) in enumerate(self.zero_deg_kick_scenarios, start=1):
                case_seed = 5500 + idx
                res = self.evaluate_single_run(policy, th, k, v, duration=dur, eval_seed=case_seed)
                results.append(res)

        if not quick_eval and include_long:
            for idx, (th, k, v, long_dur) in enumerate(self.long_scenarios, start=1):
                case_seed = 6000 + idx
                res = self.evaluate_single_run(policy, th, k, v, duration=long_dur, eval_seed=case_seed)
                results.append(res)

        return self._build_summary(results)

    def evaluate_no_kick_screen(self, policy):
        """
        Sàng lọc giai đoạn 2 (22 ca không nhiễu):
          - 18 ca không nhiễu 10s (6 góc × 3 mức pin) với đúng seed 5000+id
          - 4 ca chạy dài không nhiễu 25–30s với đúng seed 6000+idx
        """
        results = []
        test_case_id = 0
        for v in self.test_vbats:
            for k in self.test_kicks:
                for th in self.test_angles:
                    test_case_id += 1
                    if abs(k) < 1e-6:
                        case_seed = 5000 + test_case_id
                        res = self.evaluate_single_run(
                            policy, th, k, v, duration=self.duration, eval_seed=case_seed
                        )
                        results.append(res)

        for idx, (th, k, v, long_dur) in enumerate(self.long_scenarios, start=1):
            if abs(k) < 1e-6:
                case_seed = 6000 + idx
                res = self.evaluate_single_run(
                    policy, th, k, v, duration=long_dur, eval_seed=case_seed
                )
                results.append(res)

        return self._build_summary(results)

    def select_top_candidates(self, candidates, top_full=2):
        """
        Chọn checkpoint 2 bước:
          1) Chấm tất cả ứng viên trên 22 ca không nhiễu (18 ca 10s + 4 ca dài).
          2) Lấy top `top_full` bản đứng đầu đem chấm đủ 98 ca và xếp hạng lại.
        Mỗi phần tử trong `candidates` là dict chứa tối thiểu 'label' và 'policy'.
        """
        screened = []
        for cand in candidates:
            rec = dict(cand)
            rec['screen_summary'] = self.evaluate_no_kick_screen(cand['policy'])
            screened.append(rec)

        screened.sort(key=lambda r: r['screen_summary']['ranking_key'], reverse=True)

        top_full_records = []
        for rec in screened[:max(1, top_full)]:
            full_rec = dict(rec)
            full_rec['full_summary'] = self.evaluate(rec['policy'], quick_eval=False)
            top_full_records.append(full_rec)

        top_full_records.sort(key=lambda r: r['full_summary']['ranking_key'], reverse=True)
        return screened, top_full_records

    def check_deploy_gate(self, summary, gate_mode='real_robot'):
        """
        Kiểm tra điều kiện an toàn trước khi cho phép ghi đè header STM32 (Project_self).
        gate_mode:
          - 'real_robot': Nhóm không nhiễu 10s (18/18 đạt, 0 ngã) + chạy dài không nhiễu (4/4 đạt, 0 ngã)
                          + 2 ca 0° ±1.5 N (2/2 đạt, 0 ngã).
          - 'no_kick':    Nhóm không nhiễu 10s (18/18 đạt, 0 ngã) + chạy dài không nhiễu (4/4 đạt, 0 ngã).
          - 'all':        Toàn bộ ma trận 0 ngã và giữ vị trí đạt >= 80%.
        """
        reasons = []
        results = summary.get('detailed_results', [])
        if summary.get('total_runs', 0) < 90:
            return False, ["[✗] Thiếu dữ liệu ma trận đánh giá đầy đủ (chỉ có quick_eval)."]

        groups = summary.get('group_summaries', {})
        no_kick_sum = groups.get('no_kick', self._summarize_subset([]))
        long_no_kick = [r for r in results if r.get('is_long') and abs(r['kick']) < 1e-6]
        long_no_kick_sum = self._summarize_subset(long_no_kick)
        zero_kick = [
            r for r in results
            if (not r.get('is_long')) and abs(r['init_theta']) < 1e-6 and abs(abs(r['kick']) - 1.5) < 1e-6
        ]
        zero_kick_sum = self._summarize_subset(zero_kick)

        passed = True
        if gate_mode in ('real_robot', 'no_kick'):
            nk_ok = (no_kick_sum['total_runs'] > 0 and no_kick_sum['fall_count'] == 0 and no_kick_sum['hold_count'] == no_kick_sum['total_runs'])
            reasons.append(
                f"[{'✓' if nk_ok else '✗'}] Không nhiễu 10s: ngã {no_kick_sum['fall_count']}/{no_kick_sum['total_runs']}, "
                f"giữ VT {no_kick_sum['hold_count']}/{no_kick_sum['total_runs']}"
            )
            passed = passed and nk_ok

            lnk_ok = (long_no_kick_sum['total_runs'] > 0 and long_no_kick_sum['fall_count'] == 0 and long_no_kick_sum['hold_count'] == long_no_kick_sum['total_runs'])
            reasons.append(
                f"[{'✓' if lnk_ok else '✗'}] Chạy dài không nhiễu (25–30s): ngã {long_no_kick_sum['fall_count']}/{long_no_kick_sum['total_runs']}, "
                f"giữ VT {long_no_kick_sum['hold_count']}/{long_no_kick_sum['total_runs']}"
            )
            passed = passed and lnk_ok

            if gate_mode == 'real_robot':
                zk_ok = (zero_kick_sum['total_runs'] > 0 and zero_kick_sum['fall_count'] == 0 and zero_kick_sum['hold_count'] == zero_kick_sum['total_runs'])
                zk_final_txt = (
                    f"Final X max {zero_kick_sum['max_final_x_m']*100:.2f} cm"
                    if zero_kick_sum.get('has_survivors') else "Ngã hết"
                )
                reasons.append(
                    f"[{'✓' if zk_ok else '✗'}] Cú đẩy 0° (±1.5 N): ngã {zero_kick_sum['fall_count']}/{zero_kick_sum['total_runs']}, "
                    f"giữ VT {zero_kick_sum['hold_count']}/{zero_kick_sum['total_runs']} ({zk_final_txt})"
                )
                passed = passed and zk_ok
        elif gate_mode == 'all':
            all_ok = (summary['fall_count'] == 0 and summary['hold_rate_pct'] >= 80.0)
            reasons.append(
                f"[{'✓' if all_ok else '✗'}] Toàn bộ 98 ca: ngã {summary['fall_count']}/{summary['total_runs']}, "
                f"giữ VT {summary['hold_count']}/{summary['total_runs']} ({summary['hold_rate_pct']:.1f}%)"
            )
            passed = all_ok
        else:
            raise ValueError(f"Chế độ deploy gate không hợp lệ: {gate_mode}")

        return passed, reasons

    @staticmethod
    def _fmt_cm_pair(subset_sum, metric_mean_key, metric_max_key):
        if not subset_sum.get('has_survivors', True):
            return "Ngã hết"
        return f"{subset_sum[metric_mean_key]*100:.2f} / {subset_sum[metric_max_key]*100:.2f} cm"

    def format_report(self, summary, deploy_gate_mode='real_robot'):
        r = summary
        std_count = sum(1 for res in r['detailed_results'] if not res.get('is_long', False))
        long_results = [res for res in r['detailed_results'] if res.get('is_long', False)]
        settle_str = f"{r['mean_settle_s']:.2f} s" if r['mean_settle_s'] is not None else "Không đạt"

        gate_passed, gate_reasons = self.check_deploy_gate(r, gate_mode=deploy_gate_mode)
        gate_badge = "ĐẠT (Đủ điều kiện xuất STM32)" if gate_passed else "CHƯA ĐẠT (Chặn ghi đè STM32)"

        lines = []
        lines.append("### BÁO CÁO ĐÁNH GIÁ ĐỘC LẬP CHÙM BÀI TEST TOÀN DIỆN")
        lines.append(
            f"- **Tổng số ca kiểm thử:** {r['total_runs']} kịch bản "
            f"({std_count} ca {self.duration:.0f}s [gồm 90 ma trận + 2 ca 0° ±1.5 N] + {len(long_results)} ca dài 20–30s, Seed cố định)"
        )
        lines.append(f"- **Cổng kiểm tra STM32 (`{deploy_gate_mode}`):** **{gate_badge}**")
        for reason in gate_reasons:
            lines.append(f"  - {reason}")
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

        groups = r.get('group_summaries', {})
        if groups:
            lines.append("\n#### Phân Tích Theo Nhóm Kịch Bản (Không nhiễu / Có nhiễu / Chạy dài 20–30s)")
            lines.append("| Nhóm kịch bản | Số ca | Ngã | Giữ VT (2s cuối) | Max X (TB / Xấu nhất) | Final X (TB / Xấu nhất) | Settle TB (đạt) |")
            lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
            group_labels = [
                ('no_kick', 'Không nhiễu 10s (kick = 0 N)'),
                ('with_kick', 'Có nhiễu 10s (gồm 0° ±1.5 N)'),
                ('long_drift', 'Chạy dài 20–30s (bắt trôi chậm)'),
            ]
            for gkey, gtitle in group_labels:
                gs = groups.get(gkey)
                if not gs or gs['total_runs'] == 0:
                    continue
                g_settle = f"{gs['mean_settle_s']:.2f}s" if gs['mean_settle_s'] is not None else "Không đạt"
                lines.append(
                    f"| {gtitle} | {gs['total_runs']} | "
                    f"{gs['fall_count']}/{gs['total_runs']} ({gs['fall_rate_pct']:.1f}%) | "
                    f"{gs['hold_count']}/{gs['total_runs']} ({gs['hold_rate_pct']:.1f}%) | "
                    f"{self._fmt_cm_pair(gs, 'mean_max_x_m', 'max_x_worst_m')} | "
                    f"{self._fmt_cm_pair(gs, 'mean_final_x_m', 'max_final_x_m')} | "
                    f"{g_settle} |"
                )

        lines.append("\n#### Các kịch bản 10s tại điện áp danh định 7.4V (gồm 2 ca 0° ±1.5 N)")
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

    def format_two_stage_report(self, screened_records, full_records, deploy_gate_mode='real_robot'):
        """Định dạng bảng chọn checkpoint 2 bước: Giai đoạn 2 (22 ca không nhiễu) -> Top 2 (98 ca)."""
        lines = []
        lines.append("### CHỌN CHECKPOINT 2 BƯỚC (SÀNG LỌC 22 CA KHÔNG NHIỄU -> TOP 2 TRÊN 98 CA)")
        lines.append("#### Giai đoạn 2: Sàng lọc trên 18 ca không nhiễu 10s + 4 ca dài không nhiễu (22 ca)")
        lines.append("| Hạng 22 ca | Checkpoint | Quick 6s (Ngã / Giữ VT) | 22 ca Ngã | 22 ca Giữ VT | Final X TB (22 ca) | 10s NoKick Ngã | Dài NoKick Ngã |")
        lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for rank, rec in enumerate(screened_records, start=1):
            ss = rec['screen_summary']
            sg = ss.get('group_summaries', {})
            nk = sg.get('no_kick', {})
            lnk = sg.get('long_drift', {})
            qs = rec.get('quick_summary')
            quick_txt = f"{qs['fall_count']}/{qs['total_runs']} | {qs['hold_count']}/{qs['total_runs']}" if qs else "N/A"
            lines.append(
                f"| #{rank} | {rec['label']} | {quick_txt} | "
                f"{ss['fall_count']}/{ss['total_runs']} | {ss['hold_count']}/{ss['total_runs']} | "
                f"{self._fmt_cm_pair(ss, 'mean_final_x_m', 'max_final_x_m')} | "
                f"{nk.get('fall_count', 0)}/{nk.get('total_runs', 0)} | "
                f"{lnk.get('fall_count', 0)}/{lnk.get('total_runs', 0)} |"
            )

        lines.append("\n#### Giai đoạn 3: Chấm đủ 98 ca cho 2 bản đứng đầu")
        lines.append("| Hạng 98 ca | Checkpoint | Ngã (98 ca) | Giữ VT (98 ca) | Final X TB / Max | Không nhiễu 10s | Chạy dài 20–30s | Cổng STM32 |")
        lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for rank, rec in enumerate(full_records, start=1):
            fs = rec['full_summary']
            fg = fs.get('group_summaries', {})
            nk = fg.get('no_kick', {})
            ld = fg.get('long_drift', {})
            gate_ok, _ = self.check_deploy_gate(fs, gate_mode=deploy_gate_mode)
            lines.append(
                f"| #{rank} | {rec['label']} | "
                f"{fs['fall_count']}/{fs['total_runs']} ({fs['fall_rate_pct']:.1f}%) | "
                f"{fs['hold_count']}/{fs['total_runs']} ({fs['hold_rate_pct']:.1f}%) | "
                f"{self._fmt_cm_pair(fs, 'mean_final_x_m', 'max_final_x_m')} | "
                f"Ngã {nk.get('fall_count', 0)}/{nk.get('total_runs', 0)}, Giữ {nk.get('hold_count', 0)}/{nk.get('total_runs', 0)} | "
                f"Ngã {ld.get('fall_count', 0)}/{ld.get('total_runs', 0)}, Giữ {ld.get('hold_count', 0)}/{ld.get('total_runs', 0)} | "
                f"{'ĐẠT' if gate_ok else 'CHƯA ĐẠT'} |"
            )

        return "\n".join(lines)
