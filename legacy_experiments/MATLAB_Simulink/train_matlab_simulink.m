%% =========================================================================
%  SCRIPT MÔ PHỎNG VÀ KIỂM TRA MÔ HÌNH RL TRÊN MATLAB/SIMULINK
%  Dành cho Robot Hai Bánh Tự Cân Bằng STM32L476RG (Khớp thông số thực tế)
% ==========================================================================
clear; clc; close all;

%% 1. Nạp thông số vật lý thực tế từ "di chuyển/parameters.m"
run(fullfile('..', 'di chuyển', 'robot_model.m'));

fprintf('====================================================\n');
fprintf(' ĐÃ ĐỒNG BỘ THÔNG SỐ VẬT LÝ XE VÀO SIMULINK:\n');
fprintf(' - Khối lượng thân M = %.3f kg, Bánh xe m = %.3f kg\n', M, m);
fprintf(' - Chiều cao trọng tâm l = %.3f m, Bán kính r = %.3f m\n', l, r);
fprintf(' - Pin 2S: V_bat = %.1f V, Km = %.4f N/PWM\n', V_bat, Km);
fprintf(' - Chu kỳ ngắt Ts = %.4f s (200 Hz)\n', Ts);
fprintf('====================================================\n');

%% 2. Mở mô hình Simulink
mdl = 'STM32_Bot_RL_PD';
if exist(mdl, 'file') == 4
    open_system(mdl);
    fprintf('Đã mở mô hình Simulink: %s.slx\n', mdl);
else
    fprintf('Không tìm thấy file %s.slx trong thư mục hiện tại.\n', mdl);
end

