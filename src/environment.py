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
    Môi trường Gymnasium chuẩn hóa Đồng bộ 100% với Firmware STM32L476RG:
      1. Động lực học phi tuyến Runge-Kutta 2 (Midpoint) tại 200 Hz (Ts = 5 ms).
      2. Observation Space: Vector 9D [s_t, s_{t-1}, a_{t-1}].
      3. Vòng Encoder cập nhật mỗi 20 ms (50 Hz, POS_LOOP_DIVIDER = 4):
         - Lượng tử hóa vị trí theo xung Encoder (1400 CPR, r = 0.022 m).
         - Bộ lọc vận tốc đạo hàm bậc 1 với hệ số lọc N = 10 (alpha = 0.1667).
      4. Đường truyền Actuation đồng bộ với motor.c:
         - Hệ số khuếch đại lực kéo PPO: DEFAULT_PPO_BOOST = 1.30.
         - Bộ lọc LPF hành vi 1st-order: u_ppo = 0.60 * u_raw + 0.40 * u_prev.
         - Lượng tử hóa số nguyên: lroundf(u_ppo).
         - Bù vùng chết mềm (Soft Deadband Ramp) y hệt motor.c output():
           mag < 15 -> db = deadband * mag / 15; else db = deadband.
         - Mô hình động cơ vật lý phản ứng với vùng chết thực tế 80 PWM.
      5. Bộ sinh số ngẫu nhiên độc lập (self.rng) cho từng worker tránh trùng lặp.
      6. Phân bố góc khởi đầu mở rộng toàn diện (±1°, ±3°, ±5°) và xung nhiễu 2 chiều.
    """
    metadata = {"render_modes": ["human"], "render_fps": 200}

    def __init__(self, config=None, env_rank=0, seed=None):
        super().__init__()

        if config is None:
            config = {}

        self.cfg = config
        self.env_rank = env_rank
        # Bộ sinh số ngẫu nhiên riêng cho môi trường này
        init_seed = seed if seed is not None else (42 + env_rank * 1009)
        self.rng = np.random.RandomState(init_seed)

        p = config.get('robot_physics', {})
        self.M_nom = p.get('M', 0.360)
        self.m = p.get('m', 0.030)
        self.l = p.get('l', 0.031)
        self.r = p.get('r', 0.022)                       # Bán kính bánh xe r = 0.022 m (D = 44 mm)
        self.J_nom = p.get('J', 0.00072)
        self.b_nom = p.get('b', 0.050)
        self.Km = p.get('Km', 0.28)
        self.V_bat_nom = p.get('V_bat', 7.4)
        self.deadband = p.get('deadband', 80.0)           # Vùng chết động cơ vật lý: 80 PWM
        self.Ts = p.get('Ts', 0.005)                      # Chu kỳ 200 Hz (5 ms)
        self.max_pwm = p.get('max_pwm', 999.0)
        self.ppo_boost = p.get('ppo_boost', 1.30)         # Hệ số khuếch đại PPO đồng bộ board_config.h
        self.g = 9.81

        # Cấu hình Encoder & Bộ lọc vận tốc đồng bộ board_config.h
        self.enc_cpr = p.get('enc_cpr', 1400.0)           # 1400 ticks / rev
        self.tick_m = (2.0 * math.pi * self.r) / self.enc_cpr  # mét / xung
        self.pos_loop_divider = p.get('pos_loop_divider', 4)   # 4 ticks (20 ms = 50 Hz)
        self.vel_filter_n = p.get('vel_filter_n', 10.0)        # N = 10
        self.dt_p = self.Ts * self.pos_loop_divider            # 0.020 s
        self.alpha_vel = self.dt_p / (self.dt_p + 1.0 / self.vel_filter_n)  # 0.020 / 0.120 = 0.16667

        env_cfg = config.get('gym_env', {})
        self.obs_mode = env_cfg.get('obs_mode', '9d')
        self.action_mode = env_cfg.get('action_mode', 'continuous')
        self.discrete_levels = np.array(env_cfg.get('discrete_actions', [-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0]), dtype=np.float32)
        self.max_steps = env_cfg.get('max_episode_steps', 1000)
        self.fall_angle = math.radians(env_cfg.get('fall_angle_deg', 15.0))
        self.pos_limit = env_cfg.get('position_limit_m', 1.0)
        self.target_x = env_cfg.get('target_position', 0.0)

        # Domain Randomization
        dr = config.get('domain_randomization', {})
        self.dr_enabled = dr.get('enabled', True)
        self.mass_range = dr.get('mass_range', [0.34, 0.38])
        self.slope_range_deg = dr.get('slope_range_deg', [0.0, 0.0])
        self.friction_range = dr.get('friction_range', [0.045, 0.065])
        self.vbat_range = dr.get('battery_voltage_range', [7.2, 8.4])
        self.imu_noise_std = math.radians(dr.get('imu_noise_std_deg', 0.02))
        self.gyro_noise_std = math.radians(dr.get('gyro_noise_std_dps', 0.20))

        # Hàm thưởng
        rw = config.get('reward_weights', {})
        self.r_alive = rw.get('r_alive', 1.0)
        self.w_theta = rw.get('w_theta', 35.0)
        self.w_dtheta = rw.get('w_dtheta', 3.50)
        self.w_pos = rw.get('w_pos', 4.00)
        self.w_vel = rw.get('w_vel', 2.50)
        self.w_action = rw.get('w_action', 0.015)
        self.w_action_diff = rw.get('w_action_diff', 1.20)
        self.fall_penalty = rw.get('fall_penalty', 100.0)

        # Observation Space
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

        # Action Space
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
        self.enc_x = 0.0
        self.enc_x_prev = 0.0
        self.enc_v = 0.0
        self.pos_counter = 0

        self.u_prev = 0.0
        self.prev_action = 0.0
        self.step_count = 0
        self.total_env_steps = 0
        self.latency_buffer = [0.0] * max(1, p.get('latency_steps', 1))
        self.prev_obs = np.zeros(4, dtype=np.float32)

    def _get_meas_state(self):
        """Đo lường góc nghiêng (MPU6050) và vị trí/vận tốc (Encoder 20ms)"""
        noise_th = self.rng.normal(0.0, self.imu_noise_std) if self.dr_enabled else 0.0
        noise_dth = self.rng.normal(0.0, self.gyro_noise_std) if self.dr_enabled else 0.0

        meas_theta = self.state[0] + noise_th
        meas_dtheta = self.state[1] + noise_dth
        meas_x = self.enc_x
        meas_v = self.enc_v

        return np.array([meas_theta, meas_dtheta, meas_x, meas_v], dtype=np.float32)

    def _get_obs(self):
        meas_s = self._get_meas_state()

        if self.obs_mode == '9d':
            obs = np.array([
                meas_s[0], meas_s[1], meas_s[2], meas_s[3],
                self.prev_obs[0], self.prev_obs[1], self.prev_obs[2], self.prev_obs[3],
                self.prev_action
            ], dtype=np.float32)
            self.prev_obs = meas_s.copy()
            return obs
        else:
            return meas_s.astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        if options is None:
            options = {}

        if self.dr_enabled and 'fixed_physics' not in options:
            self.M = self.rng.uniform(self.mass_range[0], self.mass_range[1])
            self.J = self.J_nom * (self.M / self.M_nom)
            self.b = self.rng.uniform(self.friction_range[0], self.friction_range[1])
            self.V_bat = self.rng.uniform(self.vbat_range[0], self.vbat_range[1])
            self.slope_rad = math.radians(self.rng.uniform(self.slope_range_deg[0], self.slope_range_deg[1]))
        else:
            self.M = self.M_nom
            self.J = self.J_nom
            self.b = self.b_nom
            self.V_bat = options.get('v_bat', self.V_bat_nom)
            self.slope_rad = 0.0

        self.M_tot = self.M + 1.5 * self.m

        # Khởi tạo góc ban đầu đa tầng (Curriculum Multi-distribution):
        # Đảm bảo mô hình được luyện đều ở góc nhỏ ±1°, góc trung bình ±3° và góc lớn ±5°
        if 'init_theta_deg' in options:
            init_theta = math.radians(float(options['init_theta_deg']))
        else:
            # Phân bố góc ngẫu nhiên cho từng môi trường:
            # 40% khởi đầu góc lớn (±3° đến ±5.2°)
            # 35% khởi đầu góc vừa (±1° đến ±3°)
            # 25% khởi đầu góc nhỏ (±0.2° đến ±1°)
            dice = self.rng.rand()
            if dice < 0.40:
                max_deg = 5.2
            elif dice < 0.75:
                max_deg = 3.0
            else:
                max_deg = 1.0
            init_theta = self.rng.uniform(-math.radians(max_deg), math.radians(max_deg))

        init_x = float(options.get('init_x', self.rng.uniform(-0.005, 0.005)))
        self.state = np.array([init_theta, 0.0, init_x, 0.0], dtype=np.float32)

        # Khởi tạo trạng thái Encoder & LPF
        self.enc_x = round(init_x / self.tick_m) * self.tick_m
        self.enc_x_prev = self.enc_x
        self.enc_v = 0.0
        self.pos_counter = 0

        self.u_prev = 0.0
        self.prev_action = 0.0
        self.step_count = 0
        self.latency_buffer = [0.0] * len(self.latency_buffer)

        # Lên lịch tạo xung lực nhiễu hai chiều ngẫu nhiên trong lúc train (40% số ván)
        self.episode_kick_step = -1
        self.episode_kick_force = 0.0
        self.episode_kick_len = 0
        if self.dr_enabled and 'fixed_physics' not in options and self.rng.rand() < 0.40:
            # Chọn bước bắt đầu va đập ngẫu nhiên từ t = 1.0s đến 3.5s
            self.episode_kick_step = self.rng.randint(200, 700)
            # Lực va đập hai chiều: từ ±0.8N đến ±2.2N
            kick_dir = self.rng.choice([-1.0, 1.0])
            self.episode_kick_force = float(kick_dir * self.rng.uniform(0.8, 2.2))
            self.episode_kick_len = self.rng.randint(10, 25)  # Kéo dài 50 ms - 125 ms

        meas_s = self._get_meas_state()
        self.prev_obs = meas_s.copy()

        return self._get_obs(), {}

    def _apply_deadband_and_motor(self, firmware_pwm):
        """Mô hình phản ứng vật lý của động cơ với vùng chết tĩnh thực tế"""
        firm_mag = abs(firmware_pwm)
        if firm_mag < self.deadband:
            return 0.0
        effective = math.copysign(
            (firm_mag - self.deadband) * (self.max_pwm / (self.max_pwm - self.deadband)),
            firmware_pwm
        )
        return float(np.clip(effective, -self.max_pwm, self.max_pwm))

    def _dynamics(self, s, effective_pwm, disturbance=0.0):
        theta, dtheta, x, v = s

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
        self.total_env_steps += 1

        if self.action_mode == 'discrete':
            act_idx = int(action)
            action_val = float(self.discrete_levels[act_idx])
        else:
            if isinstance(action, (np.ndarray, list)):
                action_val = float(action[0])
            else:
                action_val = float(action)
            action_val = float(np.clip(action_val, -1.0, 1.0))

        # === 1. TÍNH TOÁN PWM ĐỒNG BỘ VỚI FIRMWARE STM32 ===
        # Bước 1.1: Hệ số khuếch đại lực kéo PPO boost (1.30)
        u_raw = action_val * self.max_pwm * self.ppo_boost

        # Bước 1.2: Bộ lọc thông thấp LPF 1st-order khử rung nơ-ron (0.60 raw + 0.40 prev)
        u_ppo = 0.60 * u_raw + 0.40 * self.u_prev
        u_ppo = float(np.clip(u_ppo, -self.max_pwm, self.max_pwm))
        self.u_prev = u_ppo

        # Bước 1.3: Lượng tử hóa số nguyên lroundf(u_ppo)
        pwm_int = int(np.round(u_ppo))

        # Bước 1.4: Bù vùng chết mềm (Soft Deadband Ramp) y hệt hàm output() trong motor.c
        mag = abs(pwm_int)
        if mag == 0:
            firmware_pwm = 0
        else:
            db = (int(self.deadband) * mag) // 15 if mag < 15 else int(self.deadband)
            firmware_pwm = int(math.copysign(min(int(self.max_pwm), mag + db), pwm_int))

        # Bước 1.5: Trễ truyền động phần cứng 5 ms
        self.latency_buffer.append(firmware_pwm)
        applied_firmware_pwm = self.latency_buffer.pop(0)

        # Bước 1.6: Phản ứng động cơ vật lý với vùng chết thực tế 80
        effective_pwm = self._apply_deadband_and_motor(applied_firmware_pwm)

        # === 2. MÔ PHỎNG ĐỘNG LỰC HỌC TẠI 200 HZ (5 MS) ===
        applied_dist = disturbance
        if applied_dist == 0.0 and self.episode_kick_step > 0:
            if self.episode_kick_step <= self.step_count < (self.episode_kick_step + self.episode_kick_len):
                applied_dist = self.episode_kick_force

        k1 = self._dynamics(self.state, effective_pwm, applied_dist)
        state_mid = self.state + 0.5 * self.Ts * k1
        k2 = self._dynamics(state_mid, effective_pwm, applied_dist)
        self.state = self.state + self.Ts * k2

        theta, dtheta, x, v = self.state

        # === 3. VÒNG ENCODER CẬP NHẬT MỖI 20 MS (POS_LOOP_DIVIDER = 4) ===
        self.pos_counter += 1
        if self.pos_counter >= self.pos_loop_divider:
            self.pos_counter = 0
            # Lượng tử hóa theo độ phân giải xung thực tế (1400 CPR)
            self.enc_x = round(x / self.tick_m) * self.tick_m
            raw_v = (self.enc_x - self.enc_x_prev) / self.dt_p
            self.enc_x_prev = self.enc_x
            # Bộ lọc đạo hàm vận tốc bậc 1 (N = 10)
            self.enc_v += self.alpha_vel * (raw_v - self.enc_v)

        # Ngắt ván nếu góc nghiêng quá 15 độ (chuẩn FALL_ANGLE_DEG của firmware)
        is_fallen = abs(theta) > self.fall_angle
        is_out_of_bounds = abs(x) > self.pos_limit
        terminated = bool(is_fallen or is_out_of_bounds)
        truncated = bool(self.step_count >= self.max_steps)

        # === 4. HÀM THƯỞNG ===
        action_diff = action_val - self.prev_action
        dx = x - self.target_x

        alive_reward = self.r_alive if abs(theta) < math.radians(6.0) else 0.0

        reward = (alive_reward
                  - self.w_theta * (theta ** 2)
                  - self.w_dtheta * (dtheta ** 2)     # Dập tắt lắc lư hình sin
                  - self.w_pos * (dx ** 2)           # Khóa vị trí đứng yên x = 0
                  - self.w_vel * (v ** 2)            # Triệt tiêu vận tốc vung lắc
                  - self.w_action * (action_val ** 2)
                  - self.w_action_diff * (action_diff ** 2))

        if terminated:
            reward -= self.fall_penalty

        self.prev_action = action_val

        info = {
            "theta_deg": math.degrees(theta),
            "x_pos": x,
            "v_vel": v,
            "enc_x": self.enc_x,
            "enc_v": self.enc_v,
            "u_ppo": u_ppo,
            "firmware_pwm": firmware_pwm,
            "applied_pwm": applied_firmware_pwm,
            "effective_pwm": effective_pwm,
            "step": self.step_count,
            "slope_deg": math.degrees(self.slope_rad),
            "mass_kg": self.M,
            "v_bat": self.V_bat
        }

        return self._get_obs(), float(reward), terminated, truncated, info
