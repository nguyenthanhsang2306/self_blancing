# Báo Cáo Thực Nghiệm & Hướng Dẫn Điều Khiển Robot Tự Cân Bằng (STM32L476RG)

[![Hardware Target](https://img.shields.io/badge/Hardware-STM32L476RG%20(ARM%20Cortex--M4%20%40%2080MHz)-blue.svg)]()
[![Control Loop](https://img.shields.io/badge/Control%20Frequency-200%20Hz%20(5%20ms)-brightgreen.svg)]()
[![Inference Latency](https://img.shields.io/badge/Inference%20Latency-6.23%20%CE%BCs-orange.svg)]()
[![Framework](https://img.shields.io/badge/Framework-Hybrid%20ADRC%20%2B%20Residual%20RL-purple.svg)]()

---

## 1. Hướng Dẫn Chạy Toàn Bộ Thí Nghiệm Theo Bộ Khung Chuẩn

Mọi thí nghiệm đều được điều phối qua điểm chạy duy nhất [`main.py`](file:///c:/Users/Nguyen%20Thanh%20Sang/OneDrive/Tài%20liệu/nhập%20môn%20lập%20trình%20nhúng/xe%20cân%20bằng/MATLAB_RL_Workspace/main.py) và cấu hình qua các file YAML chuẩn trong thư mục [`configs/`](file:///c:/Users/Nguyen%20Thanh%20Sang/OneDrive/Tài%20liệu/nhập%20môn%20lập%20trình%20nhúng/xe%20cân%20bằng/MATLAB_RL_Workspace/configs):

### Thí nghiệm 1: Huấn luyện Bản Chuẩn (Hybrid ADRC + Adversarial Residual RL)
Huấn luyện mạng nội suy vi sai thích nghi (Residual MLP 10D $\to$ 32 $\to$ 32 $\to$ 1) kết hợp bộ điều khiển ADRC qua 3 seed (42, 43, 44) kèm Sanity Check động học, Domain Randomization (chống thay đổi tải trọng/điện áp pin) và tự động xuất mã nguồn C cho STM32:
```bash
python main.py --config configs/default.yaml
```

### Thí nghiệm 2: Đánh giá Baseline (Cascade PID cổ điển)
Chạy bộ điều khiển PID 2 vòng lặp (vòng góc nghiêng trong + vòng vị trí ngoài) làm chuẩn đối sánh hiệu năng dập tắt nhiễu:
```bash
python main.py --config configs/exp_pid.yaml
```

### Thí nghiệm 3: Đánh giá ADRC thuần túy (3rd-Order LESO)
Khảo sát khả năng triệt tiêu ngoại lực tức thời của Bộ quan sát trạng thái mở rộng (Linear Extended State Observer) khi không có mạng học tăng cường:
```bash
python main.py --config configs/exp_adrc.yaml
```

### Thí nghiệm 4: Khảo sát Ablation Băng thông Bộ quan sát ($\omega_o$)
Khảo sát 3 mức băng thông $\omega_o \in [15, 48, 120]\text{ rad/s}$ để tìm điểm cân bằng giữa tốc độ dập tắt xung lực và khả năng lọc nhiễu cảm biến MPU6050:
```bash
python main.py --config configs/ablation_bandwidth.yaml
```

### Thí nghiệm 5: Chạy suy luận (Inference) và Đo độ trễ vi điều khiển thực tế
Kiểm tra khả năng phản xạ khi xe bị đấm một lực cực mạnh ($F = -1.5\text{ N}$) tại thời điểm $t = 1.0\text{ s}$ và đo thời gian thực thi tính toán:
```bash
python inference.py --checkpoint outputs/checkpoints/best_model.npy --kick_force -1.5
```

---

## 2. Báo Cáo Thực Nghiệm

### 2.1. Thông Số Vật Lý & Động Học Hệ Thống
* **Phương trình vi phân phi tuyến Lagrange** của xe con lắc ngược hai bánh:
  $$(J + M l^2)\ddot{\theta} + M l \cos(\theta) a - M g l \sin(\theta) = 0$$
  $$M_{tot} a + M l \cos(\theta)\ddot{\theta} - M l \sin(\theta)\dot{\theta}^2 = F_{motor} + F_{push} - b v - f_{coulomb}$$
* **Thông số thực nghiệm trích xuất từ phần cứng**:
  * Khối lượng thân xe: $M = 0.360\text{ kg}$, Khối lượng 2 bánh: $m = 0.030\text{ kg} \implies M_{tot} = 0.405\text{ kg}$.
  * Chiều cao trọng tâm: $l = 0.031\text{ m}$, Bán kính bánh xe: $r = 0.020\text{ m}$ (Đường kính $D = 40\text{ mm}$).
  * Momen quán tính thân xe: $J = 0.00072\text{ kg}\cdot\text{m}^2$.
  * Hệ số ma sát nhớt: $b = 0.050\text{ N}\cdot\text{s/m}$.
  * Động cơ: 2x GA12-N20, $K_m = 0.28\text{ N/V}$, Pin 2S $V_{bat} = 7.4\text{V}$.
  * Vùng chết động cơ: $\text{Deadband} = 80 / 999$ (Xung PWM nhỏ hơn 80 động cơ không quay do ma sát tĩnh).
  * Chu kỳ điều khiển ngắt phần cứng: $T_s = 0.005\text{ s}$ ($200\text{ Hz}$ trên TIM6 ngắt STM32).
  * Độ trễ cảm biến - cơ cấu chấp hành: $\tau = 5\text{ ms}$ (1 chu kỳ lấy mẫu).

---

### 2.2. Bảng So Sánh Các Phương Pháp Điều Khiển
Thử nghiệm trên kịch bản khắc nghiệt: Góc nghiêng ban đầu $10^\circ$, chịu **cú đấm ngoại lực $F = -1.2\text{ N}$ trong $80\text{ ms}$ tại $t = 1.0\text{ s}$**, và tải trọng tăng đột ngột $+30\%$ tại $t = 2.0\text{ s}$.

| Phiên Bản Điều Khiển | Max Kick Tilt (Độ) | Max Pos Drift (cm) | Sai Số Xác Lập (Độ) | Cơ Chế Triệt Tiêu Nhiễu | Thời Gian Tính Toán |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bản 1: Cascade PID** | $13.49^\circ$ | $42.66\text{ cm}$ | $8.816^\circ$ | Phản hồi sai lệch trễ ($P + I + D$) | $\approx 0.8\text{ }\mu\text{s}$ |
| **Bản 2: ADRC (3rd LESO)** | $6.86^\circ$ | $16.30\text{ cm}$ | $1.352^\circ$ | Bù chủ động ước lượng ngoại lực $z_3$ | $\approx 2.1\text{ }\mu\text{s}$ |
| **Bản 3: Hybrid ADRC + RL** | **$4.12^\circ$** | **$5.80\text{ cm}$** | **$0.000^\circ$** | **ESO bù động học + RL bù phi tuyến trễ & vùng chết** | **$6.23\text{ }\mu\text{s}$** |

---

### 2.3. Bảng Khảo Sát Ablation Băng Thông Bộ Quan Sát ($\omega_o$)

| Băng Thông $\omega_o$ | Max Kick Tilt | Max Pos Drift | Hiện Tượng Quan Sát Được | Đánh Giá Kỹ Thuật |
| :---: | :---: | :---: | :--- | :--- |
| **Quá nhỏ ($\omega_o = 15$)** | $1940.02^\circ$ | $1042.60\text{ cm}$ | Bộ quan sát phản ứng quá chậm, không ước lượng kịp $z_3$ | **Thất bại hoàn toàn (Xe ngã đập sàn)** |
| **Tối ưu ($\omega_o = 48$)** | **$6.86^\circ$** | **$16.30\text{ cm}$** | Ước lượng $z_3$ bám sát ngoại lực trong 10ms, triệt tiêu tức thời | **Hội tụ siêu cứng, ổn định trên phần cứng** |
| **Quá lớn ($\omega_o = 120$)** | $2.34^\circ$ | $15.14\text{ cm}$ | Cực kỳ cứng trong mô phỏng toán học nhưng khuếch đại nhiễu MPU6050 | **Gây nóng động cơ & rung lắc vi sai trên thực tế** |

---

## 3. Trả Lời Các Câu Hỏi Kiểm Tra / Phản Biện Chuyên Sâu

### Câu 1: Hệ số khuếch đại danh định $b_0$ được tính toán như thế nào và mang ý nghĩa gì?
* **Công thức toán học**:
  Xung quanh điểm cân bằng $\theta \approx 0$, tuyến tính hóa phương trình chuyển động của con lắc ngược ta được:
  $$\ddot{\theta} = f(\theta, \dot{\theta}, t) - b_0 u$$
  với mẫu số định thức quán tính:
  $$den_0 = M_{tot}(J + M l^2) - (M l)^2 = 0.4050 \times (0.00072 + 0.36 \times 0.031^2) - (0.36 \times 0.031)^2 = 3.0717 \times 10^{-4}\text{ kg}^2\cdot\text{m}^2$$
  Hệ số khuếch đại điều khiển danh định:
  $$b_0 = \frac{M l K_m (V_{bat} / 1000)}{den_0} \times \frac{180}{\pi} = \frac{0.36 \times 0.031 \times 0.28 \times (7.4 / 1000)}{3.0717 \times 10^{-4}} \times 57.2958 \approx 4.3132\text{ deg/(s}^2\cdot\text{PWM)}$$
* **Ý nghĩa vật lý**: Khi cấp $1\text{ LSB}$ xung PWM cho cầu H, bánh xe sẽ sinh ra lực kéo gia tốc làm con lắc lùi ngược lại một lượng gia tốc góc bằng đúng $4.3132^\circ/\text{s}^2$. Đây là tham số xương sống giúp ADRC triệt tiêu ngoại lực $u = (u_0 + z_3) / b_0$.

---

### Câu 2: Vì sao Cascade PID truyền thống lại thất bại trong việc giữ xe đứng yên (vị trí bị trôi dạt $>42\text{ cm}$)?
1. **Vấn đề vùng chết cơ khí (Deadband $80/999$)**: Động cơ GA12-N20 có ma sát tĩnh (stiction). Khi góc lệch nhỏ, tín hiệu PID sinh ra nhỏ hơn 80, bánh xe hoàn toàn đứng yên không chuyển động $\to$ Tích phân I-term bắt đầu tích lũy $\to$ Đến khi vượt qua 80 thì năng lượng tích lũy quá lớn gây vọt lố (Overshoot) và dao động lắc lư không dứt quanh vị trí 0.
2. **Trễ pha lấy mẫu 5ms**: Vòng lặp PID tính toán dựa trên sai số quá khứ $e(t) = \theta_{meas} - \theta_{ref}$. Khi có lực đấm cực mạnh, PID phải đợi xe nghiêng đi một góc đáng kể thì ngõ ra $P \times e$ mới đủ lớn $\to$ Quá trễ, xe bị giật lùi hàng chục centimet mới gượng lại được.

---

### Câu 3: Bộ quan sát trạng thái mở rộng (LESO) làm thế nào ước lượng được ngoại lực mà không cần cảm biến lực?
LESO xem **toàn bộ những gì chưa biết** (ma sát sàn, lực tay người đập vào xe, điện áp pin sụt giảm, sai số mô hình) gộp chung thành một biến trạng thái hư cấu mở rộng $z_3 = f_{total}$:
$$\dot{z}_1 = z_2 - \beta_1(z_1 - \theta_{meas})$$
$$\dot{z}_2 = z_3 - \beta_2(z_1 - \theta_{meas}) - b_0 u_{last}$$
$$\dot{z}_3 = -\beta_3(z_1 - \theta_{meas})$$
* Khi có lực tay đẩy vào xe, con lắc chuyển động lệch khỏi quỹ đạo mà gia tốc $-b_0 u_{last}$ dự đoán.
* Phần sai lệch $e = z_1 - \theta_{meas}$ ngay lập tức được khuếch đại qua hệ số $\beta_3 = \omega_o^3 = 48^3 = 110,592$.
* Chỉ trong vòng **$2 \sim 3$ chu kỳ lấy mẫu ($10\text{ ms} - 15\text{ ms}$)**, biến $z_3$ đã hội tụ chính xác về độ lớn của ngoại lực và đưa thẳng vào bộ điều khiển để triệt tiêu trực tiếp trước khi xe kịp ngã!

---

### Câu 4: Vì sao mạng nơ-ron Residual RL cần vector quan sát 10 chiều (10D History Vector)?
Môi trường nhúng thực tế là một hệ thống **POMDP (Partially Observable Markov Decision Process)** vì có độ trễ phần cứng $5\text{ ms}$ giữa thời điểm đọc IMU và thời điểm băm xung PWM:
$$o_t = [\underbrace{\theta, \dot{\theta}, x, v}_{\text{Trạng thái 4D tức thời}}, \underbrace{u_{t-1}}_{\text{Lệnh PWM trước}}, \underbrace{\Delta\theta, \Delta\dot{\theta}, \Delta x, \Delta v}_{\text{Gia tốc vi sai bậc 2}}, \underbrace{u_{t-2}}_{\text{Trễ bậc 2}}]$$
Nhờ 6 chiều lịch sử thời gian này, mạng nơ-ron học được hàm truyền ngược của độ trễ pha $z^{-1}$ và đặc tính phi tuyến của vùng chết motor, từ đó bơm ra xung lực dự đoán trước $\Delta u \in [-450, +450]$ giúp xe đứng thẳng tuyệt đối (sai số xác lập $= 0.00^\circ$).

---

### Câu 5: Làm thế nào để nạp và thực thi trên STM32L476RG với ngắt 200 Hz mà không quá tải CPU?
* File [`adrc_controller_stm32.h`](file:///c:/Users/Nguyen%20Thanh%20Sang/OneDrive/Tài%20liệu/nhập%20môn%20lập%20trình%20nhúng/xe%20cân%20bằng/MATLAB_RL_Workspace/outputs/firmware_headers/adrc_controller_stm32.h) và [`rl_controller_stm32.h`](file:///c:/Users/Nguyen%20Thanh%20Sang/OneDrive/Tài%20liệu/nhập%20môn%20lập%20trình%20nhúng/xe%20cân%20bằng/MATLAB_RL_Workspace/outputs/firmware_headers/rl_controller_stm32.h) được thiết kế thuần ANSI C, sử dụng hoàn toàn kiểu dữ liệu `float` 32-bit tương thích phần cứng FPU của ARM Cortex-M4.
* **Thời gian tính toán đo đạc thực tế**:
  * ADRC Step: $2.1\text{ }\mu\text{s}$ (khoảng 168 chu kỳ xung nhịp @ 80MHz).
  * Residual RL Forward Pass: $6.23\text{ }\mu\text{s}$ (khoảng 498 chu kỳ xung nhịp).
  * Tổng thời gian: $\approx 8.33\text{ }\mu\text{s}$.
* **Tải CPU**: Trong một chu kỳ ngắt $T_s = 5000\text{ }\mu\text{s}$ (200 Hz), thuật toán chỉ chiếm $\frac{8.33}{5000} \approx \mathbf{0.16\%}$ năng lực xử lý của vi điều khiển STM32, dành trọn $>99.8\%$ thời gian cho xử lý Bluetooth, đọc cảm biến và hiển thị OLED.
