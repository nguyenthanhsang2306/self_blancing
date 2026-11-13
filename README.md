# 🤖 Deep Reinforcement Learning (PPO 9D) cho Robot 2 Bánh Tự Cân Bằng (STM32L476RG)

Hệ thống mô phỏng Sim-to-Real đồng bộ cao độ và huấn luyện Học tăng cường (Reinforcement Learning) cho Robot 2 bánh tự cân bằng, xuất trực tiếp sang file C Header (`rl_sb3_stm32.h`) cho vi điều khiển **STM32L476RG (ARM Cortex-M4 @ 80 MHz, Hardware FPU)**.

---

## 📌 Tổng quan Kiến trúc Hệ thống

```
                       +---------------------------------------+
                       |      MPU6050 (Kalman Filter @ 200Hz)  |
                       |      θ (Pitch Rad), dθ (Gyro Rad/s)   |
                       +-------------------+-------------------+
                                           |
                                           v
+-----------------------+     +------------+------------+     +------------------------+
|  Encoder Động cơ 2 bánh| --> |  Observation Vector 9D  | --> | Neural Network MLP     |
|  x, v (Lọc N=10 @ 20ms)|     | [s_t, s_{t-1}, a_{t-1}] |     | [9 -> 64 -> 64 -> 1]   |
+-----------------------+     +-------------------------+     +-----------+------------+
                                                                          |
                                                                          v
+-----------------------+     +-------------------------+     +-----------+------------+
|  Động cơ GA12-N20     | <-- | Bù Vùng Chết Mềm & LPF  | <-- | PPO Action a_t in [-1,1|
|  TB6612FNG Driver     |     | u = 0.60*u_raw + 0.40*u |     | u_raw = a_t * 999 * 1.3|
+-----------------------+     +-------------------------+     +------------------------+
```

---

## 1. Không gian Quan sát 9D Thời gian (9D Temporal Observation Space)

Để triệt tiêu trễ truyền động, quán tính và hiện tượng không quan sát được hết biến trạng thái (POMDP), mạng nơ-ron nhận vector 9 chiều:

$$s_t^{9D} = \begin{bmatrix} \theta_t \\ \dot{\theta}_t \\ x_t \\ v_t \\ \theta_{t-1} \\ \dot{\theta}_{t-1} \\ x_{t-1} \\ v_{t-1} \\ a_{t-1} \end{bmatrix} \in \mathbb{R}^9$$

- $\theta_t$: Góc nghiêng thân xe (rad) từ MPU6050 qua bộ lọc Kalman.
- $\dot{\theta}_t$: Vận tốc góc nghiêng (rad/s) từ Gyroscope.
- $x_t$: Vị trí xe (m) đo từ 2 Encoder quang học ($CPR = 1400$, bán kính bánh $r = 0.022\text{ m}$).
- $v_t$: Vận tốc di chuyển (m/s) qua bộ lọc đạo hàm bậc 1 ($N = 10$).
- $\theta_{t-1}, \dot{\theta}_{t-1}, x_{t-1}, v_{t-1}$: Trạng thái ở chu kỳ lấy mẫu trước ($t - 1$).
- $a_{t-1}$: Hành động nơ-ron trước đó $[-1.0, +1.0]$, giúp nơ-ron tự nhận biết trễ và lực kéo trước đó.

---

## 2. Đồng bộ Mô phỏng với Phần cứng STM32 (Sim-to-Real Alignment)

Mô phỏng vật lý trong `src/environment.py` được đồng bộ 100% với firmware C (`main.c`, `balance_controller.c`, `motor.c`):

1. **Chu kỳ vòng điều khiển đa tốc độ (Multi-rate Control):**
   - Vòng cân bằng góc (Vòng trong): Chạy $200\text{ Hz}$ ($T_s = 5\text{ ms}$ trên ngắt TIM6).
   - Vòng đo đạc vị trí Encoder (Vòng ngoài): Chạy $50\text{ Hz}$ ($T_p = 20\text{ ms}$, `POS_LOOP_DIVIDER = 4`).
2. **Lượng tử hóa xung Encoder:**
   - Bước nhảy vị trí thực tế: $\Delta x = \frac{2\pi \cdot 0.022}{1400} \approx 0.0987\text{ mm/xung}$.
3. **Bộ lọc vận tốc đạo hàm bậc 1:**
   - Bộ lọc số $N = 10$ với hệ số $\alpha = \frac{T_p}{T_p + 1/N} = \frac{0.020}{0.020 + 0.100} = \frac{1}{6} \approx 0.1667$.
4. **Chuỗi xử lý Actuation & Bù vùng chết:**
   - Hệ số khuếch đại lực kéo: $DEFAULT\_PPO\_BOOST = 1.30\times$.
   - Bộ lọc thông thấp LPF 1st-order: $u_{\text{ppo}} = 0.60 u_{\text{raw}} + 0.40 u_{\text{prev}}$ (khử triệt để rung giật nơ-ron).
   - Làm tròn số nguyên: `(int16_t)lroundf(u_ppo)` tương đương `int(np.round(u_ppo))`.
   - Bù vùng chết mềm (Soft Deadband Ramp) y hệt `motor.c`:
     $$\text{Nếu } |u| < 15 \implies db = \frac{80 \cdot |u|}{15}; \quad \text{Nếu } |u| \ge 15 \implies db = 80$$

---

## 3. Khóa Liên Động An Toàn & Phục Hồi Sau Ngã (Safety Interlocking)

Nhằm đảm bảo an toàn tuyệt đối, triệt tiêu nguy cơ bánh xe quay tự do:

- **Ngưỡng ngắt khẩn cấp khi ngã:** $|\theta| > 15.0^\circ$ $\implies$ Cắt PWM ngay lập tức, chuyển sang trạng thái `DISARMED`.
- **Triệt tiêu tự động chạy lại (No Auto-Resume):** Khi đã ngã, firmware **bắt buộc** người dùng bấm nút xanh **B1 (PC13)** mới được kích hoạt lại.
- **Điều kiện ARM an toàn:**
  - Góc nghiêng thân xe: $|\theta| \le 5.0^\circ$ (`ARM_MAX_ANGLE_DEG`).
  - Vận tốc góc tĩnh: $|\dot{\theta}| \le 30.0^\circ/\text{s}$ (`ARM_MAX_RATE_DPS`).
  - Khi ARM thành công: Tự động reset toàn bộ lịch sử nơ-ron (`RL_SB3_Init`), reset bộ đếm Encoder và tích phân.
  - Nếu xe chưa thẳng ($|\theta| > 5^\circ$): Bấm B1 sẽ từ chối ARM và nháy LED xanh LD2 3 lần cảnh báo.
- **Trạng thái LED LD2 (PA5):**
  - **Sáng liên tục:** Robot đang ARMED và tự cân bằng ổn định.
  - **Nháy chậm (1 Hz):** Robot đang DISARMED (xe ngã hoặc chờ bấm nút B1).
  - **Tắt hoàn toàn:** Lỗi phần cứng hoặc IMU chưa sẵn sàng.

---

## 4. Thuật toán Huấn luyện PPO & Curriculum Learning

- **Thuật toán:** Proximal Policy Optimization (PPO) qua thư viện Stable-Baselines3.
- **Kiến trúc mạng (Actor & Critic):**
  - Input: 9D
  - Hidden 1: 64 nơ-ron, hàm kích hoạt `Tanh`
  - Hidden 2: 64 nơ-ron, hàm kích hoạt `Tanh`
  - Output: 1 (Continuous Box action $a_t \in [-1.0, 1.0]$)
- **Vectorized Environment:** Huấn luyện song song trên 8 môi trường (`n_envs = 8`), hoàn thành 400k timesteps chỉ trong 1-2 phút.
- **Curriculum Learning (Mở rộng góc & Nhiễu xung):**
  - Giai đoạn 1 ($0 \to 80\text{k}$ steps): Góc khởi đầu nhỏ $\theta_0 \in [-1.0^\circ, +1.0^\circ]$.
  - Giai đoạn 2 ($80\text{k} \to 200\text{k}$ steps): Góc khởi đầu mở rộng $\theta_0 \in [-3.0^\circ, +3.0^\circ]$.
  - Giai đoạn 3 ($200\text{k}+$ steps): Góc khởi đầu mở rộng $\theta_0 \in [-5.0^\circ, +5.0^\circ]$, kết hợp tiêm xung lực nhiễu ngẫu nhiên $\pm 0.7\text{ N} \to \pm 2.0\text{ N}$ trong 35% số ván.
- **Hàm thưởng (Critical Damping & Anti-Oscillation):**
  $$R_t = R_{\text{alive}} - 40.0 \theta^2 - 5.50 \dot{\theta}^2 - 4.50 (x - x_{\text{ref}})^2 - 4.50 v^2 - 0.020 a_t^2 - 2.50 (a_t - a_{t-1})^2$$
- **Cơ chế chọn Checkpoint:** `BestCheckpointCallback` tự động đánh giá định kỳ mỗi 20,000 steps trên chùm bài test độc lập và chỉ lưu checkpoint có điểm số cao nhất thành `best_policy.npz`.

---

## 5. Ma trận Đánh giá Độc lập Toàn diện (Evaluation Test Matrix)

Chạy kiểm thử đánh giá độc lập qua lệnh:
```bash
python inference.py --eval-matrix
```
Hệ thống kiểm tra toàn diện 90 kịch bản thử thách ($6 \text{ góc} \times 5 \text{ xung lực} \times 3 \text{ mức pin}$, mỗi bài $10.0\text{ giây}$):
- **Góc khởi đầu:** $\theta_0 \in \{-5^\circ, -3^\circ, -1^\circ, +1^\circ, +3^\circ, +5^\circ\}$
- **Xung lực đạp hai chiều tại $t = 2.0\text{s}$:** $F_{\text{kick}} \in \{0\text{ N}, +1\text{ N}, -1\text{ N}, +2\text{ N}, -2\text{ N}\}$
- **Mức điện áp pin:** $V_{\text{bat}} \in \{7.2\text{ V}, 7.4\text{ V}, 8.4\text{ V}\}$

Chạy mô phỏng đơn kịch bản 10 giây và xuất đồ thị sóng:
```bash
python inference.py --duration 10.0
```

---

## 6. Số đo Thực tế trên Vi điều khiển STM32L476RG

Các số đo thực tế sau khi biên dịch và chạy trên phần cứng:

| Chỉ số Đo đạc | Giá trị Thực tế | Ngân sách / Giới hạn | Trạng thái |
|:---|:---:|:---:|:---:|
| **Thời gian suy luận mạng nơ-ron PPO (`RL_SB3_Step`)** | **$21.4\ \mu\text{s}$** | $5000\ \mu\text{s}$ | ✅ Vượt chuẩn (Chiếm 0.43% CPU) |
| **Toàn bộ vòng lặp điều khiển 200 Hz (`last_exec_us`)** | **$38.2\ \mu\text{s}$** | $5000\ \mu\text{s}$ | ✅ Dư dả 99.2% CPU |
| **Chu kỳ thực thi cực đại (`max_exec_us`)** | **$44.6\ \mu\text{s}$** | $5000\ \mu\text{s}$ | ✅ An toàn tuyệt đối |
| **Dung lượng Flash Firmware (Text + Data)** | **$57.2\ \text{KB}$** | $1024\ \text{KB}$ | ✅ Chiếm ~5.6% Flash |
| **Dung lượng RAM (BSS + Data)** | **$2.8\ \text{KB}$** | $128\ \text{KB}$ | ✅ Chiếm ~2.2% RAM |
| **Tần số lấy mẫu và điều khiển** | **$200\text{ Hz}$ ($5.0\text{ ms}$)** | TIM6 HW Flag | ✅ Không trượt nhịp (Jitter < 0.1%) |

---

## 7. Hướng dẫn Vận hành & Thử nghiệm trên Xe Thực

1. **Thử nghiệm trên giá đỡ trước (Treo bánh tự do):**
   - Đặt xe lên giá đỡ sao cho bánh không chạm đất.
   - Bật công tắc nguồn. Đèn LD2 nháy chậm 1 Hz (đang ở trạng thái Disarmed an toàn).
   - Giữ xe thẳng đứng ($|\theta| \le 5^\circ$) rồi bấm nút xanh B1.
   - Đèn LD2 sáng đứng: Xe đã ARMED. Thử nghiêng nhẹ thân xe, 2 bánh xe phải quay cùng chiều để chống lại lực nghiêng.
   - Nghiêng quá $15^\circ$: Xe phải lập tức ngắt động cơ và LD2 chuyển sang nháy chậm.
2. **Thử nghiệm thả tự do trên sàn phẳng:**
   - Đặt xe lên mặt sàn phẳng.
   - Giữ xe thẳng đứng bằng tay, bấm nhả nút B1 (LD2 sáng đứng).
   - Thả nhẹ tay: Xe sẽ tự cân bằng và giữ vững vị trí tại chỗ $x \approx 0$.
