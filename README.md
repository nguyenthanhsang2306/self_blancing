# 🤖 Reinforcement Learning (PPO/DQN) for Two-Wheeled Self-Balancing Robot

Hệ thống mô phỏng và huấn luyện Học tăng cường (Reinforcement Learning) cho Robot 2 bánh tự cân bằng, xuất trực tiếp sang file C Header (`rl_sb3_stm32.h`) cho vi điều khiển STM32L476RG.

Tuân thủ đúng quy chuẩn 6 bước:

---

### 1. Mô phỏng mô hình vật lý (Simulation Environment)
- **Môi trường:** Custom Gymnasium Env (`SelfBalancingGymEnv`).
- **Thông số cơ học:**
  - Khối lượng thân xe: $M = 0.360\text{ kg}$
  - Khối lượng bánh xe: $m = 0.030\text{ kg}$
  - Bán kính bánh xe: $r = 0.020\text{ m}$ ($D = 40\text{ mm}$)
  - Khoảng cách trọng tâm đến trục: $l = 0.031\text{ m}$
  - Mô-men quán tính thân xe: $J = 0.00072\text{ kg}\cdot\text{m}^2$
- **Tích phân vi phân phi tuyến:** Runge-Kutta 2 (Midpoint), chu kỳ điều khiển $T_s = 5\text{ ms}$ ($200\text{ Hz}$ trên ngắt TIM6).
- **Mô phỏng phần cứng thực:** Tích hợp trễ điều khiển $5\text{ ms}$ và hàm dốc vùng chết động cơ GA12-N20 (Deadband).

---

### 2. Định nghĩa Không gian Trạng thái (Observation Space)
Vector trạng thái 4 chiều $s_t$:
$$s_t = \begin{bmatrix} \theta \\ \dot{\theta} \\ x \\ v \end{bmatrix} \in \mathbb{R}^4$$
- $\theta$: Góc nghiêng thân xe (rad) đo từ MPU6050 (Bộ lọc Kalman).
- $\dot{\theta}$: Vận tốc góc nghiêng (rad/s) từ con quay hồi chuyển Gyroscope.
- $x$: Tọa độ vị trí di chuyển (m) đo từ Encoder 2 bánh xe.
- $v$: Vận tốc di chuyển của xe (m/s).

---

### 3. Xác định Không gian Hành động (Action Space)
- **Phương án liên tục (PPO):** Hành động $a_t \in [-1.0, 1.0]$, sau đó nhân với $PWM_{max} = 999$ để điều khiển động cơ TB6612FNG.
- **Phương án rời rạc (DQN):** 7 mức lực mượt mà $\{-1.0, -0.3, -0.1, 0.0, +0.1, +0.3, +1.0\} \times \tau_{max}$ nhằm triệt tiêu rung lắc cơ học so với bộ điều khiển nhị phân 2 mức.

---

### 4. Thiết kế Hàm thưởng chống rung lắc (Reward Shaping)
Hàm thưởng kết hợp đa mục tiêu:
$$R_t = R_{survival} - w_1 \cdot \theta^2 - w_2 \cdot \dot{\theta}^2 - w_3 \cdot (a_t - a_{t-1})^2 - w_4 \cdot (x - x_{target})^2$$

- **$R_{survival} = +1.0$:** Thưởng tồn tại mỗi bước không bị ngã.
- **$-w_1 \cdot \theta^2$ ($w_1 = 12.0$):** Buộc xe giữ thân xe thẳng đứng góc $\theta \approx 0$.
- **$-w_2 \cdot \dot{\theta}^2$ ($w_2 = 0.15$):** Dập tắt dao động lắc lư của thân xe.
- **$-w_3 \cdot (a_t - a_{t-1})^2$ ($w_3 = 0.10$):** **Phạt rung giật (Action Chatter Penalty)**, triệt tiêu hiện tượng motor giật đảo chiều liên tục 200 Hz.
- **$-w_4 \cdot (x - x_{target})^2$ ($w_4 = 2.5$):** Kéo xe về điểm đứng yên $x = 0$ hoặc bám theo vị trí đặt $x_{target}$.
- **Điều kiện dừng (Terminal):** Ngắt ván và phạt nặng $-50.0$ nếu $|\theta| > 30^\circ$ hoặc $|x| > 1.5\text{ m}$.

---

### 5. Cấu hình và Huấn luyện với Stable-Baselines3
- **Thư viện:** Stable-Baselines3 (PPO / DQN) kết hợp PyTorch.
- **Kiến trúc mạng:** MLP gồm 2 tầng ẩn: `net_arch=[64, 64]`.
- **Huấn luyện:** 200,000 timesteps với thuật toán PPO.
- **Chạy trên Colab:** Mở file `train_colab.ipynb` và nhấn Run All (1 cú nhấp chuột).

---

### 6. Đánh giá và Xuất mô hình nhúng STM32 (Deployment)
- Tự động trích xuất ma trận trọng số sang C Header: `outputs/firmware_headers/rl_sb3_stm32.h`.
- Hàm suy luận $O(1)$: `RL_SB3_Inference(theta, dtheta, x, v)`.
- **Thời gian thực thi:** $\approx 35\ \mu\text{s}$ trên ARM Cortex-M4 (80 MHz có phần cứng FPU).
- Chiếm chưa tới $1\%$ thời gian của chu kỳ $5000\ \mu\text{s}$ (200 Hz)!
