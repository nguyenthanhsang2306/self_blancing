import numpy as np
import math
import copy

def evaluate_episode(env, base_controller, policy, config, domain_rand=False):
    """
    Run one complete simulation episode under adversarial conditions.
    Control command: u_total = u_base + u_residual
    """
    obs = env.reset(domain_randomize=domain_rand)
    base_controller.reset(math.degrees(env.state[0]))

    sim_duration = config['environment']['sim_duration']
    kick_f = config['environment']['kick_force']
    kick_t0 = config['environment']['kick_start_time']
    kick_dt = config['environment']['kick_duration']
    max_pwm = config['robot_physics']['max_pwm']

    steps = int(sim_duration / env.Ts)
    total_reward = 0.0

    for step in range(steps):
        t = step * env.Ts

        # External lateral punch disturbance
        dist = kick_f if (kick_t0 <= t <= kick_t0 + kick_dt) else 0.0

        theta_deg = math.degrees(env.state[0])
        dtheta_deg = math.degrees(env.state[1])
        x_m = env.state[2]
        v_m = env.state[3]

        # Base controller command
        if hasattr(base_controller, 'step'):
            u_base, _ = base_controller.step(theta_deg, x_m, v_m)
        else:
            u_base = base_controller.compute(theta_deg, dtheta_deg, x_m, v_m)

        # Residual RL adjustment
        delta_u = policy.forward(obs) if policy is not None else 0.0

        # Total clamped action
        u_total = float(np.clip(u_base + delta_u, -max_pwm, max_pwm))

        obs, r, done, _ = env.step(u_total, disturbance_force=dist)
        total_reward += r

        if done:
            # Heavy penalty for falling over
            total_reward -= 500.0 * (steps - step)
            break

    return total_reward


def train_residual_policy(env, base_controller, policy, config):
    """
    Evolutionary Policy Search for Residual RL Policy.
    High efficiency, multi-scenario evaluation with domain randomization.
    """
    hp = config['rl_hyperparameters']
    pop_size = hp['population_size']
    num_gens = hp['num_generations']
    lr = hp['learning_rate']
    sigma = hp['sigma']

    curr_params = policy.get_params_flat().copy()
    n_params = len(curr_params)

    best_reward = -float('inf')
    best_params = curr_params.copy()
    history = {'generation': [], 'mean_reward': [], 'best_reward': []}

    print(f"[Training] Commencing NES Optimization for {num_gens} generations (Population: {pop_size})...")
    print(f"[Training] Model Parameter Count: {n_params} float32 weights.")

    test_policy = copy.deepcopy(policy)

    for gen in range(1, num_gens + 1):
        # Antithetic sampling for variance reduction
        half_pop = pop_size // 2
        noise_half = np.random.randn(half_pop, n_params).astype(np.float32)
        noise = np.vstack([noise_half, -noise_half])

        rewards = np.zeros(pop_size, dtype=np.float32)

        for i in range(pop_size):
            cand_params = curr_params + sigma * noise[i]
            test_policy.set_params_flat(cand_params)

            # Evaluate on nominal and randomized domain
            r1 = evaluate_episode(env, base_controller, test_policy, config, domain_rand=False)
            r2 = evaluate_episode(env, base_controller, test_policy, config, domain_rand=True)
            rewards[i] = 0.5 * (r1 + r2)

        # Fitness normalization
        std_r = np.std(rewards)
        if std_r < 1e-6:
            std_r = 1e-6
        norm_rewards = (rewards - np.mean(rewards)) / std_r

        # Natural gradient update
        grad = np.dot(noise.T, norm_rewards) / pop_size
        curr_params += lr * grad

        gen_best = float(np.max(rewards))
        gen_mean = float(np.mean(rewards))

        if gen_best > best_reward:
            best_reward = gen_best
            best_params = curr_params.copy()

        history['generation'].append(gen)
        history['mean_reward'].append(gen_mean)
        history['best_reward'].append(best_reward)

        if gen % 10 == 0 or gen == 1 or gen == num_gens:
            print(f"  Gen {gen:3d}/{num_gens} | Mean R: {gen_mean:8.1f} | Best R: {best_reward:8.1f} | Sigma: {sigma:.3f}")

    # Load best weights into policy
    policy.set_params_flat(best_params)
    print(f"[Training Complete] Global Best Fitness Achieved: {best_reward:.2f}\n")
    return history


def run_benchmark(env, controllers_dict, config):
    """
    Standardized Comparative Benchmark.
    Tests PID vs ADRC vs ADRC+RL under extreme stress:
      - Initial tilt: 10 degrees
      - Shock punch at t = 1.0s: -1.2 N (80ms)
      - Sudden payload addition (+30%) at t = 2.0s
    """
    sim_dur = config['environment']['sim_duration']
    steps = int(sim_dur / env.Ts)
    kick_f = config['environment']['kick_force']
    kick_t0 = config['environment']['kick_start_time']
    kick_dt = config['environment']['kick_duration']
    max_pwm = config['robot_physics']['max_pwm']

    results = {}

    for name, c_dict in controllers_dict.items():
        base = c_dict['base']
        policy = c_dict.get('policy', None)

        env.reset(domain_randomize=False)
        base.reset(math.degrees(env.state[0]))

        t_log = []
        theta_log = []
        dtheta_log = []
        x_log = []
        v_log = []
        pwm_log = []
        dist_log = []
        z3_log = []

        obs = env.get_observation()

        for step in range(steps):
            t = step * env.Ts

            # Shock punch at t=1.0s
            dist = kick_f if (kick_t0 <= t <= kick_t0 + kick_dt) else 0.0

            # Sudden +30% mass payload at t=2.0s
            if t >= 2.0:
                env.M = env.M_nom * 1.30
                env.M_tot = env.M + 1.5 * env.m_nom

            theta_deg = math.degrees(env.state[0])
            dtheta_deg = math.degrees(env.state[1])
            x_m = env.state[2]
            v_m = env.state[3]

            z3_val = 0.0
            if hasattr(base, 'step'):
                u_base, z3_val = base.step(theta_deg, x_m, v_m)
            else:
                u_base = base.compute(theta_deg, dtheta_deg, x_m, v_m)

            delta_u = policy.forward(obs) if policy is not None else 0.0
            u_total = float(np.clip(u_base + delta_u, -max_pwm, max_pwm))

            obs, _, done, info = env.step(u_total, disturbance_force=dist)

            t_log.append(t)
            theta_log.append(theta_deg)
            dtheta_log.append(dtheta_deg)
            x_log.append(x_m)
            v_log.append(v_m)
            pwm_log.append(u_total)
            dist_log.append(dist)
            z3_log.append(z3_val)

        # Performance Metrics
        arr_t = np.array(t_log)
        arr_th = np.array(theta_log)
        arr_x = np.array(x_log)
        arr_pwm = np.array(pwm_log)

        # Max tilt after kick (t >= 1.0)
        kick_idx = arr_t >= 1.0
        max_kick_tilt = float(np.max(np.abs(arr_th[kick_idx])))
        max_pos_drift = float(np.max(np.abs(arr_x)))
        steady_tilt = float(np.mean(np.abs(arr_th[arr_t >= 2.5])))
        steady_x = float(np.mean(np.abs(arr_x[arr_t >= 2.5])))

        results[name] = {
            'time': arr_t,
            'theta': arr_th,
            'dtheta': np.array(dtheta_log),
            'x': arr_x,
            'v': np.array(v_log),
            'pwm': arr_pwm,
            'dist': np.array(dist_log),
            'z3': np.array(z3_log),
            'metrics': {
                'max_kick_tilt_deg': max_kick_tilt,
                'max_pos_drift_m': max_pos_drift,
                'steady_tilt_deg': steady_tilt,
                'steady_pos_drift_m': steady_x,
                'peak_pwm': float(np.max(np.abs(arr_pwm)))
            }
        }

    return results

