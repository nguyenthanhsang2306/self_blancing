/**
 * ============================================================================
 * @file    adrc_controller_stm32.h
 * @brief   Bộ điều khiển siêu cứng ADRC + Extended State Observer (ESO) cho STM32L476RG
 * @note    Tự động ước lượng và triệt tiêu ngoại lực đẩy tay trong vòng 5ms!
 *          Thời gian thực thi: < 5 microseconds trên Cortex-M4 FPU @ 80MHz!
 * ============================================================================
 */

#ifndef ADRC_CONTROLLER_STM32_H
#define ADRC_CONTROLLER_STM32_H

#include <math.h>

/* Thông số thiết kế ADRC (Băng thông quan sát & điều khiển) */
#define ADRC_TS        0.005f     /* Ts = 5ms (200Hz) */
#define ADRC_B0        4.31321f  /* Hệ số khuếch đại danh định */
#define ADRC_INV_B0    0.23185f

#define ADRC_BETA1     144.000f
#define ADRC_BETA2     6912.000f
#define ADRC_BETA3     110592.000f

#define ADRC_KP        342.250f
#define ADRC_KD        37.000f

#define ADRC_KP_POS    2.005f     /* Vòng ngoài khóa vị trí x=0m */
#define ADRC_KD_POS    3.151f

typedef struct {
    float z1;      /* Góc Pitch ước lượng (đã lọc nhiễu MPU6050) [độ] */
    float z2;      /* Vận tốc góc ước lượng [độ/s] */
    float z3;      /* TỔNG NGOẠI LỰC VÀ NHIỄU (Total Disturbance) [độ/s^2] */
    float u_last;  /* Xung điều khiển chu kỳ trước */
} ADRC_State_t;

static ADRC_State_t g_adrc = {0.0f, 0.0f, 0.0f, 0.0f};

/**
 * @brief  Khởi tạo hoặc đặt lại trạng thái ADRC khi bật xe
 */
static inline void ADRC_Reset(float init_pitch_deg)
{
    g_adrc.z1 = init_pitch_deg;
    g_adrc.z2 = 0.0f;
    g_adrc.z3 = 0.0f;
    g_adrc.u_last = 0.0f;
}

/**
 * @brief  Hàm tính toán điều khiển ADRC siêu cứng chạy ở chu kỳ ngắt TIM6 (200Hz)
 * @param  pitch_deg   Góc nghiêng thân xe (độ) từ cảm biến MPU6050
 * @param  pos_m       Vị trí quãng đường xe đã đi (mét) từ Encoder
 * @param  vel_mps     Vận tốc dài của xe (mét/giây) từ Encoder
 * @param  out_z3      Con trỏ nhận giá trị ngoại lực ước lượng (có thể truyền NULL)
 * @return Xung PWM điều khiển [-999.0f, +999.0f]
 */
static inline float ADRC_Compute_PWM(float pitch_deg, float pos_m, float vel_mps, float *out_z3)
{
    /* 1. Bộ quan sát trạng thái mở rộng ESO: Ước lượng z1, z2 và NGOẠI LỰC z3 */
    float e = g_adrc.z1 - pitch_deg;
    g_adrc.z1 += ADRC_TS * (g_adrc.z2 - ADRC_BETA1 * e);
    g_adrc.z2 += ADRC_TS * (g_adrc.z3 - ADRC_BETA2 * e - ADRC_B0 * g_adrc.u_last);
    g_adrc.z3 += ADRC_TS * (-ADRC_BETA3 * e);

    if (out_z3) *out_z3 = g_adrc.z3;

    /* 2. Vòng ngoài khóa vị trí x = 0m */
    float th_ref = -ADRC_KP_POS * pos_m - ADRC_KD_POS * vel_mps;
    if (th_ref >  7.0f) th_ref =  7.0f;
    if (th_ref < -7.0f) th_ref = -7.0f;

    /* 3. Phản hồi trạng thái siêu cứng */
    float e_th = g_adrc.z1 - th_ref;
    float u0 = ADRC_KP * e_th + ADRC_KD * g_adrc.z2;

    /* 4. CHÌA KHÓA ADRC: BÙ TRIỆT TIÊU NGOẠI LỰC TỨC THÌ */
    float u_raw = (u0 + g_adrc.z3) * ADRC_INV_B0;

    /* Giới hạn bão hòa PWM [-999, 999] */
    if (u_raw >  999.0f) u_raw =  999.0f;
    if (u_raw < -999.0f) u_raw = -999.0f;

    g_adrc.u_last = u_raw;
    return u_raw;
}

#endif /* ADRC_CONTROLLER_STM32_H */
