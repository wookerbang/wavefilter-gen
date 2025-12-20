import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def generate_bandpass_s21(center_freq, bandwidth, order=3, ripple=0.1, f_axis=None):
    """
    生成带通滤波器的 S21 复数响应 (模拟 Chebyshev I 型或 Butterworth)
    """
    # 计算截止频率 (rad/s)
    w_center = 2 * np.pi * center_freq
    w_bw = 2 * np.pi * bandwidth
    w_low = w_center - w_bw / 2
    w_high = w_center + w_bw / 2
    
    # 这里使用 Butterworth 作为基底，因为它最平坦，容易看清陷波的效果
    # 如果想要纹波，可以改用 signal.cheby1
    b, a = signal.butter(order, [w_low, w_high], btype='bandpass', analog=True)
    
    # 计算频响
    w, h = signal.freqs(b, a, worN=2 * np.pi * f_axis)
    return h

def generate_notch_s21(notch_freq, q_factor=10, depth_db=30, f_axis=None):
    """
    生成陷波器 (Notch) 的 S21 响应
    这是一个标准的二阶陷波传递函数模型： (s^2 + w0^2) / (s^2 + w0/Q*s + w0^2)
    """
    w0 = 2 * np.pi * notch_freq
    
    # 构建陷波器的分子分母系数 (Analog Domain)
    # H(s) = (s^2 + w0^2) / (s^2 + (w0/Q)*s + w0^2)
    # 注意：这个理想模型的陷波深度是无限的，为了模拟真实工程中的有限深度（如有损耗），
    # 我们可以给分子也加一点阻尼，或者简单地接受它很深。
    # 这里使用标准 IIR Notch 设计思路的模拟版
    
    b = [1, 0, w0**2]
    a = [1, w0 / q_factor, w0**2]
    
    w, h = signal.freqs(b, a, worN=2 * np.pi * f_axis)
    return h

def plot_s21(f_axis, s21_data, label, color, linestyle='-'):
    """绘制 S21 幅频曲线 (dB)"""
    mag_db = 20 * np.log10(np.abs(s21_data) + 1e-9) # 加 1e-9 防止 log(0)
    plt.plot(f_axis / 1e9, mag_db, label=label, color=color, linewidth=2, linestyle=linestyle)

# ================= 主程序 =================

# 1. 设置频率轴 (1GHz 到 4GHz)
f_start = 1e9
f_stop = 4e9
num_points = 1001
freqs = np.linspace(f_start, f_stop, num_points)

# 2. 参数设置
fc = 2.45e9      # 中心频率 2.45 GHz (WiFi 频段)
bw = 200e6       # 带宽 200 MHz
notch_f = 2.42e9 # 陷波频率 2.42 GHz (故意设在通带内部，制造“缺陷”)
notch_q = 50     # 陷波的 Q 值（值越大，陷波越窄）

# 3. 生成基础带通响应
s21_bp = generate_bandpass_s21(fc, bw, order=4, f_axis=freqs)

# 4. 生成陷波响应
s21_notch = generate_notch_s21(notch_f, q_factor=notch_q, f_axis=freqs)

# 5. 合成：带通 + 陷波 (级联 = 复数相乘)
# 这模拟了在带通滤波器后面串联一个 LC 陷波器的效果
s21_combined = s21_bp * s21_notch

# ================= 绘图 =================
plt.figure(figsize=(10, 6))

# 画带通
plot_s21(freqs, s21_bp, label='Base Bandpass (Order=4)', color='dodgerblue', linestyle='--')

# 画带通+陷波
plot_s21(freqs, s21_combined, label=f'BP + Notch @ {notch_f/1e9:.2f}GHz', color='crimson')

# 设置图表样式 (仿论文风格)
plt.title('Waveform Zoo Generation: Bandpass vs. Bandpass with Notch', fontsize=14)
plt.xlabel('Frequency (GHz)', fontsize=12)
plt.ylabel('Magnitude (dB)', fontsize=12)
plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.ylim(-60, 5) # 限制 Y 轴范围，聚焦通带
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()

# 保存
save_path = 'bp_notch_s21_demo.png'
plt.savefig(save_path, dpi=300)
print(f"图像已保存至: {save_path}")
plt.show()