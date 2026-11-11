import numpy as np
import math

class SelfBalancingRobotEnv:
    """
    High-fidelity non-linear simulation environment for two-wheeled balancing robot.
    Includes:
      - 4D state space: [theta (rad), dtheta (rad/s), x (m), v (m/s)]
      - Runge-Kutta 2 (Midpoint) numerical integrator
      - Sensor-to-motor actuation latency buffer (5 ms delay)
      - Non-linear deadband ramp modeling static friction & stiction
      - History observation buffer for temporal awareness
      - Domain Randomization (Sim-to-Real transfer robustness)
    """
    def __init__(self, config):
        self.cfg = config
        p = config['robot_physics']
        self.M_nom = p['M']
        self.m_nom = p['m']
        self.l_nom = p['l']
        self.r = p['r']
        self.J_nom = p['J']
        self.b_nom = p['b']
        self.Km = p['Km']
        self.V_bat_nom = p['V_bat']
        self.deadband = p['deadband']
        self.Ts = p['Ts']
        self.max_pwm = p['max_pwm']
        self.g = 9.81

        rl_hp = config.get('rl_hyperparameters', {})
        self.history_len = rl_hp.get('history_len', 3)
        self.obs_dim = rl_hp.get('input_dim', 10)

        self.reset()

    def reset(self, domain_randomize=False):
        """Reset state and history buffer with optional parameter randomization."""
        dr = self.cfg['environment']['domain_randomization']
        if domain_randomize and dr['enabled']:
            m_jitter = 1.0 + np.random.uniform(-dr['mass_variation'], dr['mass_variation'])
            f_jitter = 1.0 + np.random.uniform(-dr['friction_variation'], dr['friction_variation'])
            v_sag = 1.0 - np.random.uniform(0.0, dr['battery_sag_pct'])
            self.M = self.M_nom * m_jitter
            self.b = self.b_nom * f_jitter
            self.V_bat = self.V_bat_nom * v_sag
            self.J = self.J_nom * m_jitter
        else:
            self.M = self.M_nom
            self.b = self.b_nom
            self.V_bat = self.V_bat_nom
            self.J = self.J_nom

        self.M_tot = self.M + 1.5 * self.m_nom

        # Initial conditions: default tilt in rad
        init_tilt = math.radians(self.cfg['environment']['initial_tilt_deg'])
        # Add slight initial randomness
        if domain_randomize:
            init_tilt += math.radians(np.random.uniform(-1.5, 1.5))

        self.state = np.array([init_tilt, 0.0, 0.0, 0.0], dtype=np.float64)
        self.time = 0.0

        # Delay buffer for hardware latency (1 step = 5ms)
        self.actuation_delay_buffer = [0.0] * max(1, self.cfg['robot_physics']['latency_steps'])
        self.last_applied_pwm = 0.0

        # History buffers for temporal observation [s_t, u_{t-1}, delta_s_t, u_{t-2}]
        self.prev_state = self.state.copy()
        self.u_history = [0.0, 0.0]

        return self.get_observation()

    def _apply_deadband(self, u_cmd):
        """Simulate real hardware motor stiction and PWM deadband."""
        mag = abs(u_cmd)
        if mag < self.deadband:
            return 0.0
        # Smooth linear ramp past deadband
        effective_pwm = math.copysign((mag - self.deadband) * (self.max_pwm / (self.max_pwm - self.deadband)), u_cmd)
        return float(np.clip(effective_pwm, -self.max_pwm, self.max_pwm))

    def _dynamics(self, s, u_pwm, disturbance_force=0.0):
        """
        Compute non-linear equations of motion:
        s = [theta, dtheta, x, v]
        Returns: [dtheta, ddtheta, v, a]
        """
        theta, dtheta, x, v = s

        # Motor tractive force on ground (parameters.m & GA12-N20 specs)
        effective_pwm = self._apply_deadband(u_pwm)
        F_motor = (self.Km * (self.V_bat / self.max_pwm)) * effective_pwm

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        den = self.M_tot * (self.J + self.M * (self.l_nom**2)) - (self.M * self.l_nom * cos_t)**2
        if abs(den) < 1e-7:
            den = 1e-7

        # Coulomb & viscous friction + centripetal forces
        f_c = 0.02 * math.tanh(v * 20.0)
        F_n = F_motor - self.b * v - f_c + self.M * self.l_nom * (dtheta**2) * sin_t
        tau = self.M * self.g * self.l_nom * sin_t + disturbance_force * self.l_nom * cos_t

        # Non-linear coupled accelerations
        a = ((self.J + self.M * (self.l_nom**2)) * F_n - (self.M * self.l_nom * cos_t) * tau) / den
        ddtheta = (self.M_tot * tau - (self.M * self.l_nom * cos_t) * F_n) / den

        return np.array([dtheta, ddtheta, v, a], dtype=np.float64)

    def step(self, u_cmd, disturbance_force=0.0):
        """
        Execute one control cycle (Ts = 0.005s) using Runge-Kutta 2 (Midpoint).
        """
        # Hardware latency delay: motor sees delayed command
        self.actuation_delay_buffer.append(u_cmd)
        applied_pwm = self.actuation_delay_buffer.pop(0)
        self.last_applied_pwm = applied_pwm

        # Midpoint RK2 integration
        k1 = self._dynamics(self.state, applied_pwm, disturbance_force)
        state_mid = self.state + 0.5 * self.Ts * k1
        k2 = self._dynamics(state_mid, applied_pwm, disturbance_force)

        self.prev_state = self.state.copy()
        self.state += self.Ts * k2
        self.time += self.Ts

        # Update action history
        self.u_history.append(u_cmd)
        if len(self.u_history) > 2:
            self.u_history.pop(0)

        # Observation
        obs = self.get_observation()

        # Reward formulation for balance and zero position drift
        theta_deg = math.degrees(self.state[0])
        dtheta_deg = math.degrees(self.state[1])
        x_m = self.state[2]
        v_m = self.state[3]

        # Penalties: quadratic tilt, angular velocity, position drift, and motor effort
        r_angle = -1.0 * (theta_deg ** 2) - 0.05 * (dtheta_deg ** 2)
        r_pos = -15.0 * (x_m ** 2) - 2.0 * (v_m ** 2)
        r_effort = -0.0001 * (u_cmd ** 2)
        reward = r_angle + r_pos + r_effort

        # Failure condition
        done = abs(theta_deg) > 40.0

        return obs, reward, done, {
            "theta_deg": theta_deg,
            "dtheta_deg": dtheta_deg,
            "x_m": x_m,
            "v_m": v_m,
            "applied_pwm": applied_pwm
        }

    def get_observation(self):
        """
        Generate 10D History-Aware observation vector:
        [theta_deg, dtheta_deg, x_m, v_m, u_{t-1}, delta_theta, delta_dtheta, delta_x, delta_v, u_{t-2}]
        """
        curr = np.array([
            math.degrees(self.state[0]),
            math.degrees(self.state[1]),
            self.state[2],
            self.state[3]
        ], dtype=np.float64)

        prev = np.array([
            math.degrees(self.prev_state[0]),
            math.degrees(self.prev_state[1]),
            self.prev_state[2],
            self.prev_state[3]
        ], dtype=np.float64)

        delta = curr - prev

        obs = np.array([
            curr[0] / 30.0,         # Normalized theta (+/- 30 deg -> +/- 1.0)
            curr[1] / 150.0,        # Normalized dtheta (+/- 150 deg/s)
            curr[2] / 0.5,          # Normalized x (+/- 0.5 m)
            curr[3] / 1.5,          # Normalized v (+/- 1.5 m/s)
            self.u_history[-1] / 999.0, # u_{t-1}
            delta[0] / 5.0,         # delta theta
            delta[1] / 20.0,        # delta dtheta
            delta[2] / 0.05,        # delta x
            delta[3] / 0.1,         # delta v
            self.u_history[-2] / 999.0  # u_{t-2}
        ], dtype=np.float32)

        return obs

