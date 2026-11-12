import math
import numpy as np

# Thử import gymnasium, nếu chưa cài thì fallback lớp Env cơ bản
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    class gym:
        class Env:
            pass
    class spaces:
        class Box:
            def __init__(self, low, high, shape=None, dtype=np.float32):
                self.low = np.array(low, dtype=dtype)
                self.high = np.array(high, dtype=dtype)
                self.shape = shape if shape is not None else self.low.shape
                self.dtype = dtype
        class Discrete:
            def __init__(self, n):
                self.n = n

class SelfBalancingGymEnv(gym.Env):
    """
    Môi trường Gymnasium chuẩn cho Robot 2 bánh tự cân bằng:
    Tuân thủ đúng 6 bước trong hướng dẫn RL:
      1. Mô phỏng động lực học phi tuyến (Midpoint RK2, chu kỳ Ts = 5ms / 200 Hz).
      2. Observation Space: Vector 4D [theta (rad), dtheta (rad/s), x (m), v (m/s)].
      3. Action Space:
         - Liên tục (PPO): tau / PWM chuẩn hóa trong [-1.0, 1.0].
         - Rời rạc (DQN): 7 mức lực mượt [-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0].
      4. Reward Shaping: Thưởng tồn tại, phạt góc nghiêng, phạt giật động cơ và phạt trôi vị trí.
    """
    metadata = {"render_modes": ["human"], "render_fps": 200}

    def __init__(self, config=None):
        super().__init__()

        # Cấu hình mặc định nếu không truyền vào
        if config is None:
            config = {
                'robot_physics': {
                    'M': 0.360, 'm': 0.030, 'l': 0.031, 'r': 0.020,
                    'J': 0.00072, 'b': 0.050, 'Km': 0.28, 'V_bat': 7.4,
                    'deadband': 80.0, 'Ts': 0.005, 'max_pwm': 999.0, 'latency_steps': 1
                },
                'gym_env': {
                    'action_mode': 'continuous',
                    'discrete_actions': [-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0],
                    'max_episode_steps': 1000,
                    'fall_angle_deg': 30.0,
                    'position_limit_m': 1.5,
                    'target_position': 0.0
                },
                'reward_weights': {
                    'survival': 1.0,
                    'w_theta': 12.0,
                    'w_dtheta': 0.15,
                    'w_action_diff': 0.10,
                    'w_pos': 2.5,
                    'fall_penalty': 50.0
                }
            }

        self.cfg = config
        p = config['robot_physics']
        self.M = p['M']               # Khối lượng thân xe (kg)
        self.m = p['m']               # Khối lượng bánh xe (kg)
        self.l = p['l']               # Khoảng cách từ trục bánh đến CoG (m)
        self.r = p['r']               # Bán kính bánh xe (m)
        self.J = p['J']               # Quán tính thân xe (kg*m^2)
        self.b = p['b']               # Hệ số ma sát cản (N*s/m)
        self.Km = p['Km']             # Hệ số mô-men động cơ GA12-N20
        self.V_bat = p['V_bat']       # Điện áp pin (V)
        self.deadband = p['deadband'] # Vùng chết động cơ
        self.Ts = p['Ts']             # 0.005s (200 Hz)
        self.max_pwm = p['max_pwm']   # 999
        self.g = 9.81
        self.M_tot = self.M + 1.5 * self.m

        env_cfg = config.get('gym_env', {})
        self.action_mode = env_cfg.get('action_mode', 'continuous')
        self.discrete_levels = np.array(env_cfg.get('discrete_actions', [-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0]), dtype=np.float32)
        self.max_steps = env_cfg.get('max_episode_steps', 1000)
        self.fall_angle = math.radians(env_cfg.get('fall_angle_deg', 30.0))
        self.pos_limit = env_cfg.get('position_limit_m', 1.5)
        self.target_x = env_cfg.get('target_position', 0.0)

        rw = config.get('reward_weights', {})
        self.w_survival = rw.get('survival', 1.0)
        self.w_theta = rw.get('w_theta', 12.0)
        self.w_dtheta = rw.get('w_dtheta', 0.15)
        self.w_action_diff = rw.get('w_action_diff', 0.10)
        self.w_pos = rw.get('w_pos', 2.5)
        self.fall_penalty = rw.get('fall_penalty', 50.0)

        # 2. Định nghĩa Observation Space: [theta, dtheta, x, v]
        high_obs = np.array([self.fall_angle * 1.2, 15.0, self.pos_limit * 1.2, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)

        # 3. Định nghĩa Action Space
        if self.action_mode == 'discrete':
            self.action_space = spaces.Discrete(len(self.discrete_levels))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Trạng thái nội tại
        self.state = np.zeros(4, dtype=np.float32)
        self.prev_action = 0.0
        self.step_count = 0
        self.latency_buffer = [0.0] * max(1, p.get('latency_steps', 1))

    def reset(self, seed=None, options=None):
        """Khởi tạo lại trạng thái ban đầu của Robot với độ nhiễu nhỏ."""
        if seed is not None:
            np.random.seed(seed)

        # Nghiêng ngẫu nhiên nhẹ ban đầu trong khoảng [-4°, +4°]
        init_theta = np.random.uniform(-math.radians(4.0), math.radians(4.0))
        init_x = np.random.uniform(-0.02, 0.02)
        self.state = np.array([init_theta, 0.0, init_x, 0.0], dtype=np.float32)

        self.prev_action = 0.0
        self.step_count = 0
        self.latency_buffer = [0.0] * len(self.latency_buffer)

        return self.state.copy(), {}

    def _apply_deadband(self, u_pwm):
        """Mô phỏng vùng chết và ma sát tĩnh của động cơ GA12-N20."""
        mag = abs(u_pwm)
        if mag < self.deadband:
            return 0.0
        effective_pwm = math.copysign((mag - self.deadband) * (self.max_pwm / (self.max_pwm - self.deadband)), u_pwm)
        return float(np.clip(effective_pwm, -self.max_pwm, self.max_pwm))

    def _dynamics(self, s, u_pwm, disturbance=0.0):
        """Phương trình động lực học vi phân phi tuyến của xe tự cân bằng."""
        theta, dtheta, x, v = s

        effective_pwm = self._apply_deadband(u_pwm)
        F_motor = (self.Km * (self.V_bat / self.max_pwm)) * effective_pwm

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        den = self.M_tot * (self.J + self.M * (self.l**2)) - (self.M * self.l * cos_t)**2
        if abs(den) < 1e-7:
            den = 1e-7

        f_coulomb = 0.02 * math.tanh(v * 20.0)
        F_net = F_motor - self.b * v - f_coulomb + self.M * self.l * (dtheta**2) * sin_t
        tau = self.M * self.g * self.l * sin_t + disturbance * self.l * cos_t

        a = ((self.J + self.M * (self.l**2)) * F_net - (self.M * self.l * cos_t) * tau) / den
        ddtheta = (self.M_tot * tau - (self.M * self.l * cos_t) * F_net) / den

        return np.array([dtheta, ddtheta, v, a], dtype=np.float32)

    def step(self, action):
        """
        Thực hiện một bước điều khiển (Ts = 5 ms):
          - Quy đổi hành động về PWM [-999, +999].
          - Tích phân Runge-Kutta 2 (Midpoint).
          - Tính hàm thưởng kết hợp chống giật rung.
        """
        self.step_count += 1

        # Chuyển đổi hành động sang giá trị float trong [-1.0, 1.0]
        if self.action_mode == 'discrete':
            act_idx = int(action)
            action_val = float(self.discrete_levels[act_idx])
        else:
            if isinstance(action, (np.ndarray, list)):
                action_val = float(action[0])
            else:
                action_val = float(action)
            action_val = float(np.clip(action_val, -1.0, 1.0))

        # Mô phỏng trễ phần cứng (1 chu kỳ = 5 ms)
        self.latency_buffer.append(action_val * self.max_pwm)
        applied_pwm = self.latency_buffer.pop(0)

        # Tích phân Midpoint RK2
        k1 = self._dynamics(self.state, applied_pwm)
        state_mid = self.state + 0.5 * self.Ts * k1
        k2 = self._dynamics(state_mid, applied_pwm)
        self.state = self.state + self.Ts * k2

        theta, dtheta, x, v = self.state

        # Kiểm tra điều kiện ngã hoặc vượt giới hạn (Terminal condition)
        is_fallen = abs(theta) > self.fall_angle
        is_out_of_bounds = abs(x) > self.pos_limit
        terminated = bool(is_fallen or is_out_of_bounds)
        truncated = bool(self.step_count >= self.max_steps)

        # 4. THIẾT KẾ HÀM THƯỞNG (Reward Shaping)
        # R = +1 (tồn tại) - w1*theta^2 - w2*dtheta^2 - w3*(a_t - a_{t-1})^2 - w4*(x - x_target)^2
        action_diff = action_val - self.prev_action
        reward = (self.w_survival
                  - self.w_theta * (theta ** 2)
                  - self.w_dtheta * (dtheta ** 2)
                  - self.w_action_diff * (action_diff ** 2)
                  - self.w_pos * ((x - self.target_x) ** 2))

        # Phạt nặng khi bị ngã xe
        if terminated:
            reward -= self.fall_penalty

        self.prev_action = action_val

        info = {
            "theta_deg": math.degrees(theta),
            "x_pos": x,
            "applied_pwm": applied_pwm,
            "step": self.step_count
        }

        return self.state.copy(), float(reward), terminated, truncated, info
