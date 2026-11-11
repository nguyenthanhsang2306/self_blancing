import numpy as np
import math

class CascadePIDController:
    """
    Standard Dual-Loop Cascade PID Controller.
    Outer loop: Position & Velocity -> Target Tilt Angle
    Inner loop: Tilt Angle & Angular Velocity -> Motor PWM
    """
    def __init__(self, config):
        pid = config['baseline_pid']
        self.kp_angle = pid['kp_angle']
        self.kd_angle = pid['kd_angle']
        self.kp_pos = pid['kp_pos']
        self.kd_pos = pid['kd_pos']
        self.max_pwm = config['robot_physics']['max_pwm']

    def reset(self, *args, **kwargs):
        pass

    def compute(self, theta_deg, dtheta_deg, x_m, v_m):
        """
        Compute control action:
        theta_ref = - (kp_pos * x + kd_pos * v)
        u = kp_angle * (theta - theta_ref) + kd_angle * dtheta
        """
        # Outer position loop calculates virtual target tilt (limited to +/- 10 deg)
        theta_ref = -(self.kp_pos * x_m + self.kd_pos * v_m)
        theta_ref = float(np.clip(theta_ref, -10.0, 10.0))

        # Inner balance loop
        err_angle = theta_deg - theta_ref
        u = self.kp_angle * err_angle + self.kd_angle * dtheta_deg
        return float(np.clip(u, -self.max_pwm, self.max_pwm))


class ADRC_ESO_Controller:
    """
    Active Disturbance Rejection Control (ADRC) with 3rd-order Linear Extended State Observer (LESO).
    Estimates and rejects total lumped disturbance (external push, friction, parameter shifts) in real time.
    """
    def __init__(self, config):
        adrc = config['adrc_controller']
        self.b0 = adrc['b0']
        self.omega_o = adrc['omega_o']
        self.omega_c = adrc['omega_c']
        self.pos_kp = adrc['pos_kp']
        self.pos_kd = adrc['pos_kd']
        self.Ts = config['robot_physics']['Ts']
        self.max_pwm = config['robot_physics']['max_pwm']

        # Observer gains derived from pole placement: (s + omega_o)^3
        self.beta1 = 3.0 * self.omega_o
        self.beta2 = 3.0 * (self.omega_o ** 2)
        self.beta3 = self.omega_o ** 3

        # State feedback gains: (s + omega_c)^2
        self.kp = self.omega_c ** 2
        self.kd = 2.0 * self.omega_c

        self.reset()

    def reset(self, init_theta_deg=None):
        if init_theta_deg is not None:
            self.z1 = float(init_theta_deg)
            self.initialized = True
        else:
            self.z1 = 0.0
            self.initialized = False
        self.z2 = 0.0
        self.z3 = 0.0
        self.last_u = 0.0

    def step(self, theta_meas_deg, x_m, v_m):
        """
        Update LESO and calculate active disturbance rejection control command.
        """
        if not self.initialized:
            self.z1 = float(theta_meas_deg)
            self.initialized = True

        # 1. Observer error
        e = self.z1 - theta_meas_deg

        # 2. Update observer states using Euler discretization
        dz1 = self.z2 - self.beta1 * e
        dz2 = self.z3 - self.beta2 * e - self.b0 * self.last_u
        dz3 = -self.beta3 * e

        self.z1 += self.Ts * dz1
        self.z2 += self.Ts * dz2
        self.z3 += self.Ts * dz3

        # 3. Outer position loop: calculates target angle to bring robot back to x=0
        theta_ref = -(self.pos_kp * x_m + self.pos_kd * v_m)
        theta_ref = float(np.clip(theta_ref, -7.0, 7.0))

        # 4. State Feedback Control Law: u0 = kp*(z1 - theta_ref) + kd*z2
        u0 = self.kp * (self.z1 - theta_ref) + self.kd * self.z2

        # 5. Disturbance Rejection Compensation: u = (u0 + z3) / b0
        u_raw = (u0 + self.z3) / self.b0
        u_cmd = float(np.clip(u_raw, -self.max_pwm, self.max_pwm))
        self.last_u = u_cmd

        return u_cmd, self.z3


class ResidualPolicyNet:
    """
    Ultra-compact, low-latency 2-hidden-layer MLP for Residual Reinforcement Learning.
    Input: 10D temporal history vector
    Hidden: 32 -> 32 (Tanh activations)
    Output: 1D residual torque delta_u in [-action_scale, +action_scale]
    """
    def __init__(self, config):
        hp = config['rl_hyperparameters']
        self.input_dim = hp['input_dim']
        self.hidden1_dim = hp['hidden_dims'][0]
        self.hidden2_dim = hp['hidden_dims'][1]
        self.output_dim = hp['output_dim']
        self.action_scale = hp['action_scale']

        self.w1 = np.zeros((self.hidden1_dim, self.input_dim), dtype=np.float32)
        self.b1 = np.zeros(self.hidden1_dim, dtype=np.float32)

        self.w2 = np.zeros((self.hidden2_dim, self.hidden1_dim), dtype=np.float32)
        self.b2 = np.zeros(self.hidden2_dim, dtype=np.float32)

        self.w3 = np.zeros((self.output_dim, self.hidden2_dim), dtype=np.float32)
        self.b3 = np.zeros(self.output_dim, dtype=np.float32)

        self.init_weights()

    def init_weights(self):
        """Xavier / He initialization with small output scale for gentle initial residuals."""
        std1 = math.sqrt(2.0 / (self.input_dim + self.hidden1_dim))
        self.w1 = (np.random.randn(self.hidden1_dim, self.input_dim) * std1).astype(np.float32)
        self.b1 = np.zeros(self.hidden1_dim, dtype=np.float32)

        std2 = math.sqrt(2.0 / (self.hidden1_dim + self.hidden2_dim))
        self.w2 = (np.random.randn(self.hidden2_dim, self.hidden1_dim) * std2).astype(np.float32)
        self.b2 = np.zeros(self.hidden2_dim, dtype=np.float32)

        # Output weights initialized small so initial residual starts near zero
        self.w3 = (np.random.randn(self.output_dim, self.hidden2_dim) * 0.05).astype(np.float32)
        self.b3 = np.zeros(self.output_dim, dtype=np.float32)

    def forward(self, obs):
        """
        Single-vector inference. Returns scalar residual action.
        """
        # Layer 1
        h1 = np.tanh(self.w1 @ obs + self.b1)
        # Layer 2
        h2 = np.tanh(self.w2 @ h1 + self.b2)
        # Output layer with bounded action scale
        out = (self.w3 @ h2 + self.b3).item()
        action = math.tanh(out) * self.action_scale
        return action

    def get_params_flat(self):
        """Flatten all network parameters into 1D vector."""
        return np.concatenate([
            self.w1.flatten(), self.b1.flatten(),
            self.w2.flatten(), self.b2.flatten(),
            self.w3.flatten(), self.b3.flatten()
        ])

    def set_params_flat(self, flat):
        """Reconstruct weight tensors from flat parameter vector."""
        idx = 0
        w1_len = self.hidden1_dim * self.input_dim
        self.w1 = flat[idx : idx + w1_len].reshape(self.hidden1_dim, self.input_dim)
        idx += w1_len

        b1_len = self.hidden1_dim
        self.b1 = flat[idx : idx + b1_len]
        idx += b1_len

        w2_len = self.hidden2_dim * self.hidden1_dim
        self.w2 = flat[idx : idx + w2_len].reshape(self.hidden2_dim, self.hidden1_dim)
        idx += w2_len

        b2_len = self.hidden2_dim
        self.b2 = flat[idx : idx + b2_len]
        idx += b2_len

        w3_len = self.output_dim * self.hidden2_dim
        self.w3 = flat[idx : idx + w3_len].reshape(self.output_dim, self.hidden2_dim)
        idx += w3_len

        b3_len = self.output_dim
        self.b3 = flat[idx : idx + b3_len]

