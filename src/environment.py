import math
import numpy as np

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
    Môi trường Gymnasium chuẩn hóa (Dead-Stiff & Anti-Limit-Cycle):
      1. Động lực học phi tuyến Runge-Kutta 2 (Midpoint) tại 200 Hz (Ts = 5ms).
      2. Observation Space: Vector 9D [s_t, s_{t-1}, a_{t-1}].
      3. Hàm thưởng triệt tiêu dao động hình sin:
         Khâu cản vận tốc góc (w_dtheta = 2.0) và khâu cản vận tốc tịnh tiến (w_vel = 1.0)
         cực mạnh để triệt tiêu hoàn toàn hiện tượng con lắc võng (overshoot/limit cycle).
    """
    metadata = {"render_modes": ["human"], "render_fps": 200}

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = {}

        self.cfg = config
        p = config.get('robot_physics', {})
        self.M_nom = p.get('M', 0.360)
        self.m = p.get('m', 0.030)
        self.l = p.get('l', 0.031)
        self.r = p.get('r', 0.020)
        self.J_nom = p.get('J', 0.00072)
        self.b_nom = p.get('b', 0.050)
        self.Km = p.get('Km', 0.28)
        self.V_bat_nom = p.get('V_bat', 7.4)
        self.deadband = p.get('deadband', 80.0)
        self.Ts = p.get('Ts', 0.005)
        self.max_pwm = p.get('max_pwm', 999.0)
        self.g = 9.81

        env_cfg = config.get('gym_env', {})
        self.obs_mode = env_cfg.get('obs_mode', '9d')
        self.action_mode = env_cfg.get('action_mode', 'continuous')
        self.discrete_levels = np.array(env_cfg.get('discrete_actions', [-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0]), dtype=np.float32)
        self.max_steps = env_cfg.get('max_episode_steps', 1000)
        self.fall_angle = math.radians(env_cfg.get('fall_angle_deg', 15.0))
        self.pos_limit = env_cfg.get('position_limit_m', 1.0)
        self.target_x = env_cfg.get('target_position', 0.0)

        # Cấu hình Domain Randomization
        dr = config.get('domain_randomization', {})
        self.dr_enabled = dr.get('enabled', True)
        self.mass_range = dr.get('mass_range', [0.34, 0.38])
        self.slope_range_deg = dr.get('slope_range_deg', [0.0, 0.0])
        self.friction_range = dr.get('friction_range', [0.045, 0.065])
        self.vbat_range = dr.get('battery_voltage_range', [7.2, 8.4])
        self.imu_noise_std = math.radians(dr.get('imu_noise_std_deg', 0.02))
        self.gyro_noise_std = math.radians(dr.get('gyro_noise_std_dps', 0.20))

        # Cấu hình hàm thưởng triệt tiêu dao động
        rw = config.get('reward_weights', {})
        self.r_alive = rw.get('r_alive', 1.0)
        self.w_theta = rw.get('w_theta', 35.0)
        self.w_dtheta = rw.get('w_dtheta', 2.00)
        self.w_pos = rw.get('w_pos', 0.08)
        self.w_vel = rw.get('w_vel', 1.00)
        self.w_action = rw.get('w_action', 0.005)
        self.w_action_diff = rw.get('w_action_diff', 0.04)
        self.fall_penalty = rw.get('fall_penalty', 100.0)

        # Định nghĩa Observation Space
        if self.obs_mode == '9d':
            high_9d = np.array([
                self.fall_angle * 1.5, 15.0, self.pos_limit * 1.5, 10.0,
                self.fall_angle * 1.5, 15.0, self.pos_limit * 1.5, 10.0,
                1.0
            ], dtype=np.float32)
            self.observation_space = spaces.Box(low=-high_9d, high=high_9d, dtype=np.float32)
        else:
            high_4d = np.array([self.fall_angle * 1.5, 15.0, self.pos_limit * 1.5, 10.0], dtype=np.float32)
            self.observation_space = spaces.Box(low=-high_4d, high=high_4d, dtype=np.float32)

        # Định nghĩa Action Space
        if self.action_mode == 'discrete':
            self.action_space = spaces.Discrete(len(self.discrete_levels))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.M = self.M_nom
        self.J = self.J_nom
        self.b = self.b_nom
        self.V_bat = self.V_bat_nom
        self.slope_rad = 0.0
        self.M_tot = self.M + 1.5 * self.m

        self.state = np.zeros(4, dtype=np.float32)
        self.prev_state = np.zeros(4, dtype=np.float32)
        self.prev_action = 0.0
        self.step_count = 0
        self.latency_buffer = [0.0] * max(1, p.get('latency_steps', 1))

    def _get_obs(self):
        noise_th = np.random.normal(0.0, self.imu_noise_std) if self.dr_enabled else 0.0
        noise_dth = np.random.normal(0.0, self.gyro_noise_std) if self.dr_enabled else 0.0

        meas_s = self.state.copy()
        meas_s[0] += noise_th
        meas_s[1] += noise_dth

        if self.obs_mode == '9d':
            return np.array([
                meas_s[0], meas_s[1], meas_s[2], meas_s[3],
                self.prev_state[0], self.prev_state[1], self.prev_state[2], self.prev_state[3],
                self.prev_action
            ], dtype=np.float32)
        else:
            return meas_s.astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        if self.dr_enabled:
            self.M = np.random.uniform(self.mass_range[0], self.mass_range[1])
            self.J = self.J_nom * (self.M / self.M_nom)
            self.b = np.random.uniform(self.friction_range[0], self.friction_range[1])
            self.V_bat = np.random.uniform(self.vbat_range[0], self.vbat_range[1])
            self.slope_rad = math.radians(np.random.uniform(self.slope_range_deg[0], self.slope_range_deg[1]))
        else:
            self.M = self.M_nom
            self.J = self.J_nom
            self.b = self.b_nom
            self.V_bat = self.V_bat_nom
            self.slope_rad = 0.0

        self.M_tot = self.M + 1.5 * self.m

        # Khởi tạo góc ban đầu nhẹ [-1.0°, +1.0°]
        init_theta = np.random.uniform(-math.radians(1.0), math.radians(1.0))
        init_x = np.random.uniform(-0.005, 0.005)
        self.state = np.array([init_theta, 0.0, init_x, 0.0], dtype=np.float32)
        self.prev_state = self.state.copy()
        self.prev_action = 0.0
        self.step_count = 0
        self.latency_buffer = [0.0] * len(self.latency_buffer)

        return self._get_obs(), {}

    def _apply_deadband(self, u_pwm):
        mag = abs(u_pwm)
        if mag < self.deadband:
            return 0.0
        effective_pwm = math.copysign((mag - self.deadband) * (self.max_pwm / (self.max_pwm - self.deadband)), u_pwm)
        return float(np.clip(effective_pwm, -self.max_pwm, self.max_pwm))

    def _dynamics(self, s, u_pwm, disturbance=0.0):
        theta, dtheta, x, v = s

        effective_pwm = self._apply_deadband(u_pwm)
        F_motor = (self.Km * (self.V_bat / self.max_pwm)) * effective_pwm

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        den = self.M_tot * (self.J + self.M * (self.l**2)) - (self.M * self.l * cos_t)**2
        if abs(den) < 1e-7:
            den = 1e-7

        g_slope_torque = self.M * self.g * self.l * math.sin(theta - self.slope_rad)
        f_slope_gravity = - self.M_tot * self.g * math.sin(self.slope_rad)

        f_coulomb = 0.02 * math.tanh(v * 20.0)
        F_net = F_motor + f_slope_gravity - self.b * v - f_coulomb + self.M * self.l * (dtheta**2) * sin_t
        tau = g_slope_torque + disturbance * self.l * cos_t

        a = ((self.J + self.M * (self.l**2)) * F_net - (self.M * self.l * cos_t) * tau) / den
        ddtheta = (self.M_tot * tau - (self.M * self.l * cos_t) * F_net) / den

        return np.array([dtheta, ddtheta, v, a], dtype=np.float32)

    def step(self, action, disturbance=0.0):
        self.step_count += 1

        if self.action_mode == 'discrete':
            act_idx = int(action)
            action_val = float(self.discrete_levels[act_idx])
        else:
            if isinstance(action, (np.ndarray, list)):
                action_val = float(action[0])
            else:
                action_val = float(action)
            action_val = float(np.clip(action_val, -1.0, 1.0))

        # Trễ phần cứng 5 ms
        self.latency_buffer.append(action_val * self.max_pwm)
        applied_pwm = self.latency_buffer.pop(0)

        # Midpoint RK2
        k1 = self._dynamics(self.state, applied_pwm, disturbance)
        state_mid = self.state + 0.5 * self.Ts * k1
        k2 = self._dynamics(state_mid, applied_pwm, disturbance)

        self.prev_state = self.state.copy()
        self.state = self.state + self.Ts * k2

        theta, dtheta, x, v = self.state

        # Ngắt ván nếu góc nghiêng quá 15 độ
        is_fallen = abs(theta) > self.fall_angle
        is_out_of_bounds = abs(x) > self.pos_limit
        terminated = bool(is_fallen or is_out_of_bounds)
        truncated = bool(self.step_count >= self.max_steps)

        # HÀM THƯỞNG TRIỆT TIÊU HOÀN TOÀN DAO ĐỘNG CON LẮC VÕNG
        action_diff = action_val - self.prev_action
        dx = x - self.target_x

        # Chỉ thưởng khi góc đứng thẳng an toàn (|theta| < 6 độ)
        alive_reward = self.r_alive if abs(theta) < math.radians(6.0) else 0.0

        reward = (alive_reward
                  - self.w_theta * (theta ** 2)
                  - self.w_dtheta * (dtheta ** 2)     # Dập tắt lắc lư hình sin
                  - self.w_pos * (dx ** 2)           # Trọng số nhẹ, không gây quá trớn
                  - self.w_vel * (v ** 2)            # Triệt tiêu vận tốc vung lắc
                  - self.w_action * (action_val ** 2)
                  - self.w_action_diff * (action_diff ** 2))

        if terminated:
            reward -= self.fall_penalty

        self.prev_action = action_val

        info = {
            "theta_deg": math.degrees(theta),
            "x_pos": x,
            "applied_pwm": applied_pwm,
            "step": self.step_count,
            "slope_deg": math.degrees(self.slope_rad),
            "mass_kg": self.M
        }

        return self._get_obs(), float(reward), terminated, truncated, info
