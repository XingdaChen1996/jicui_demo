# fight for the bright future
# contend: 
# author: xingdachen
# time: 
# email: chenxingda@iat-center.com

# import numpy as np
#
# from scipy.fftpack import fft, ifft
#
# import matplotlib.pyplot as plt
#
# x = np.linspace(0, 1, 1400)
# x = np.linspace(0, 1, 1401)
#
# # 设置需要采样的信号，频率分量有180，390和600
#
# y = 7 * np.sin(2 * np.pi * 180 * x) + 2.8 * np.sin(2 * np.pi * 390 * x) + 5.1 * np.sin(2 * np.pi * 600 * x)
# y = 7 * np.sin(2 * np.pi * 180 * x) + 7 * np.cos(2 * np.pi * 180 * x)
# y = 7 * np.exp(2 * np.pi * 180 * x * 1j)
#
#
# yy = fft(y)  # 快速傅里叶变换
#
# yreal = yy.real  # 获取实数部分
#
# yimag = yy.imag  # 获取虚数部分
#
# yf = abs(fft(y))  # 取绝对值
#
# yf1 = abs(fft(y)) / len(x)  # 归一化处理
#
# yf2 = yf1[range(int(len(x) / 2))]  # 由于对称性，只取一半区间
#
# xf = np.arange(len(y))  # 频率
#
# xf1 = xf
#
# xf2 = xf[range(int(len(x) / 2))]  # 取一半区间
#
# plt.subplot(221)
#
# plt.plot(x[0:50], y[0:50])
#
# plt.title('Original wave')
#
# plt.subplot(222)
#
# plt.plot(xf, yf, 'r')
#
# plt.title('FFT of Mixed wave(two sides frequency range)', fontsize=7, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表
#
# plt.subplot(223)
#
# plt.plot(xf1, yf1, 'g')
#
# plt.title('FFT of Mixed wave(normalization)', fontsize=9, color='r')
#
# plt.subplot(224)
#
# plt.plot(xf2, yf2, 'b')
#
# plt.title('FFT of Mixed wave)', fontsize=10, color='#F08080')
#
# plt.show()


import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
N = 1400
# 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x = np.linspace(0, 1, N)  # 时间是[0,1], 采样点的个数 N=1400, fs=1400, 频域分辨率 fbin=fs/N=1
zero_array = np.zeros((int(N / 2)))
x1 = x[0:int(N / 2)]
x2 = x[int(N / 2):]

y1 = 7 * np.sin(2 * np.pi * 200 * x1)
y2 = 5 * np.sin(2 * np.pi * 500 * x2)
y = (np.concatenate((y1, zero_array)) + np.concatenate((zero_array, y2)))   # 第一个y
y_2 = (np.concatenate((y2, zero_array)) + np.concatenate((zero_array, y1)))   # 第二个y

fft_y = fft(y)  # 快速傅里叶变换
yy = fft(y_2)

N = len(x)
x = np.arange(N)  # 频率个数
half_x = x[range(int(N / 2))]  # 取一半区间

abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
angle_y = np.angle(fft_y)  # 取复数的角度
normalization_y = abs_y / N  # 归一化处理（双边频谱）
normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）

plt.subplot(231)
plt.plot(x, y)
plt.title('原始波形')

plt.subplot(232)
plt.plot(x, fft_y, 'black')
plt.title('双边振幅谱(未求振幅绝对值)', fontsize=9, color='black')

plt.subplot(233)
plt.plot(x, abs_y, 'r')
plt.title('双边振幅谱(未归一化)', fontsize=9, color='red')

plt.subplot(234)
plt.plot(x, angle_y, 'violet')
plt.title('双边相位谱(未归一化)', fontsize=9, color='violet')

plt.subplot(235)
plt.plot(x, normalization_y, 'g')
plt.title('双边振幅谱(归一化)', fontsize=9, color='green')

plt.subplot(236)
plt.plot(half_x, normalization_half_y, 'blue')
plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')

# plt.show()
plt.savefig("图")

print(normalization_y.max())
print(np.where(normalization_y == normalization_y.max()))

print(normalization_y[200] * 2)
print(normalization_y[500] * 2)
print(normalization_y[100] * 2)
print(y[100])
