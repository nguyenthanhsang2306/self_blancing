import math
import numpy as np
from src.environment import SelfBalancingGymEnv

class PolicyEvaluator:
    """
    Bộ đánh giá độc lập toàn diện cho Controller (9D PPO RL):
      - Ma trận góc khởi đầu: [-5°, -3°, -1°, +1°, +3°, +5°]
      - Xung lực nhiễu hai chiều: [0.0N, +1.0N, -1.0N, +2.0N, -2.0N] (tại t = 2.0s - 2.1s)
      - Các mức điện áp pin: [7.2V, 7.4V, 8.4V]
      - Thời gian kiểm tra: 10.0 giây (2000 bước tại Ts = 5 ms)
      - Sử dụng Seed cố định độc lập (Evaluation Seeds) để đảm bảo tính khách quan và lặp lại 100%.
      - Thu thập chỉ số: Tỷ lệ ngã, Độ lệch vị trí cực đại, Thời gian trở về gốc (Settling time), RMS PWM.
    """

    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config
        self.test_angles = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
        self.test_kicks = [0.0, 1.0, -1.0, 2.0, -2.0]
        self.test_vbats = [7.2, 7.4, 8.4]
        self.duration = 10.0

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
        final_x = float(abs(x_hist[-1]))
        rms_pwm = float(np.sqrt(np.mean(np.square(pwm_hist))))

        # Tính thời gian trở về gốc (Settling time): Thời điểm mà từ đó trở đi |theta| <= 0.5° và |x| <= 0.02 m
        settling_time = duration
        if not fallen:
            settled = False
            for i in range(len(t_hist) - 1, -1, -1):
                if abs(theta_hist[i]) > 0.50 or abs(x_hist[i]) > 0.020:
                    settling_time = t_hist[min(i + 1, len(t_hist) - 1)]
                    settled = True
                    break
            if not settled:
                settling_time = 0.0

        return {
            'init_theta': init_theta_deg,
            'kick': kick_n,
            'vbat': vbat_v,
            'fallen': fallen,
            'survive_time': survive_time,
            'max_deg': max_deg,
            'max_x': max_x,
            'final_x': final_x,
            'settling_time': settling_time,
            'rms_pwm': rms_pwm
        }

    def evaluate(self, policy, quick_eval=False):
        """
        Đánh giá độc lập với bộ Seed cố định (Independent Evaluation Suite):
          - quick_eval=True: Dùng trong lúc huấn luyện (18 test case tiêu biểu để chọn checkpoint nhanh)
          - quick_eval=False: Đánh giá đầy đủ 90 test case (10 giây mỗi bài)
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
                    # Seed cố định riêng cho từng kịch bản đánh giá
                    case_seed = 5000 + test_case_id
                    res = self.evaluate_single_run(policy, th, k, v, duration=duration, eval_seed=case_seed)
                    results.append(res)

        total_runs = len(results)
        fall_count = sum(1 for r in results if r['fallen'])
        fall_rate = (fall_count / total_runs) * 100.0

        survived_runs = [r for r in results if not r['fallen']]
        if survived_runs:
            mean_max_x = float(np.mean([r['max_x'] for r in survived_runs]))
            max_x_worst = float(np.max([r['max_x'] for r in survived_runs]))
            mean_final_x = float(np.mean([r['final_x'] for r in survived_runs]))
            mean_settle = float(np.mean([r['settling_time'] for r in survived_runs]))
            mean_rms_pwm = float(np.mean([r['rms_pwm'] for r in survived_runs]))
        else:
            mean_max_x = 1.0
            max_x_worst = 1.0
            mean_final_x = 1.0
            mean_settle = duration
            mean_rms_pwm = 999.0

        # Điểm đánh giá tổng hợp:
        # 1. Ưu tiên cao nhất: Không ngã (1000 điểm)
        # 2. Phạt nặng hiện tượng con lắc võng dao động liên tục không tắt (mean_settle = 10s)
        # 3. Phạt nếu vị trí cuối cùng không hồi phục về gốc x = 0
        composite_score = (
            (100.0 - fall_rate) * 10.0
            - 80.0 * mean_settle
            - 100.0 * mean_final_x
            - 0.10 * mean_rms_pwm
        )

        summary = {
            'total_runs': total_runs,
            'fall_count': fall_count,
            'fall_rate_pct': fall_rate,
            'mean_max_x_m': mean_max_x,
            'max_x_worst_m': max_x_worst,
            'mean_final_x_m': mean_final_x,
            'mean_settle_s': mean_settle,
            'mean_rms_pwm': mean_rms_pwm,
            'composite_score': composite_score,
            'detailed_results': results
        }

        return summary

    def format_report(self, summary):
        r = summary
        lines = []
        lines.append("### BÁO CÁO ĐÁNH GIÁ ĐỘC LẬP CHÙM BÀI TEST TOÀN DIỆN")
        lines.append(f"- **Tổng số ca kiểm thử:** {r['total_runs']} kịch bản (Thời gian mỗi bài: {self.duration:.1f}s, Seed đánh giá cố định)")
        lines.append(f"- **Tỷ lệ ngã (Fall Rate):** {r['fall_rate_pct']:.1f}% ({r['fall_count']}/{r['total_runs']} bài ngã)")
        lines.append(f"- **Độ lệch vị trí trung bình:** {r['mean_max_x_m']*100:.2f} cm (Xấu nhất: {r['max_x_worst_m']*100:.2f} cm)")
        lines.append(f"- **Thời gian trở về gốc trung bình:** {r['mean_settle_s']:.2f} s")
        lines.append(f"- **RMS PWM trung bình:** {r['mean_rms_pwm']:.1f} / 999")
        lines.append(f"- **Điểm số chất lượng (Composite Score):** {r['composite_score']:.1f}")
        lines.append("\n| Góc khởi đầu | Xung nhiễu (N) | Pin (V) | Sống sót (s) | Max Theta | Max X (cm) | Settling (s) | RMS PWM |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        # In mẫu các kịch bản tiêu biểu ở mức pin danh định 7.4V
        sample_results = [res for res in r['detailed_results'] if res['vbat'] == 7.4]
        for res in sample_results:
            status = f"{res['survive_time']:.1f}s" if not res['fallen'] else f"NGÃ ({res['survive_time']:.2f}s)"
            lines.append(
                f"| {res['init_theta']:+4.1f}° | {res['kick']:+4.1f} N | {res['vbat']:.1f}V | "
                f"{status} | {res['max_deg']:.2f}° | {res['max_x']*100:.1f} cm | "
                f"{res['settling_time']:.2f}s | {res['rms_pwm']:.0f} |"
            )

        return "\n".join(lines)
