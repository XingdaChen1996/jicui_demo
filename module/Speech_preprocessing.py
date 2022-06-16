# fight for the bright future
# contend: 
# author: xingdachen
# time: 
# email: chenxingda@iat-center.com

import math
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft


class speech_pre_time(object):

    def __init__(self, a_hyper=0.05, b_hyper=0.95, split_quite_num=100, c_hyper=0.05, d_hyper=0.95, text_sum_pro_choose=0.04):
        self.a_hyper = a_hyper
        self.b_hyper = b_hyper
        self.c_hyper = c_hyper
        self.d_hyper = d_hyper
        self.split_quite_num = split_quite_num
        self.text_sum_pro_choose = text_sum_pro_choose

    class node_time_domain(object):              # 这个类相当于静态方法,类名和类对象都可以调用
        def __init__(self, x, FS_samplerate):
            self.FS = FS_samplerate
            self.x = x

    class node_fre_domain(object):
        def __init__(self, x, FS_samplerate):
            N = len(x)
            fft_y = fft(x)  # 快速傅里叶正变换，如果实函数，这个就是一个复数
            self.FS = FS_samplerate
            self.num = N
            self.fre_bin = FS_samplerate/N
            self.angle_y = np.angle(fft_y)  # 取复数的角度  Angle
            self.normalization_y = np.abs(fft_y) / N  # 归一化处理（双边频谱） Amplituted,   注意这里除的是N，但是如果按照采样定理应该除以N/2
            self.normalization_half_y = self.normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
            self.normalization_x = self.fre_bin * np.arange(0, self.num, 1)
            self.normalization_half_x = self.normalization_x[range(int(N / 2))]
            self.sort_normalization_x, self.sort_normalization_y = self.sort_y_fre()

        def sort_y_fre(self):
            index = np.argsort(self.normalization_half_y)
            index = np.flip(index)   # 取按照从大到小的序列进行排序，这是大到小的索引
            x_sort = self.normalization_half_x[index]
            y_sort = self.normalization_half_y[index]
            return x_sort, y_sort

    class node_uniform_fre_compression(object):
        def __init__(self, fre_domain_obj, fre_len=5):    # fre_len=5 以fu-fr=5hz
            self.fre_len = fre_len
            self.step = int(np.ceil(fre_len / fre_domain_obj.fre_bin))  # step 求和的步长
            self.fre_compression_amplitude, self.fre_origin_amplitude = split_sum(fre_domain_obj.normalization_half_y, self.step)
            self.fre_center = self.compute_fre_center(fre_domain_obj)

        def compute_fre_center(self, fre_domain_obj):
            N = self.step
            start = (N-1)/2
            step = N
            end = self.fre_origin_amplitude.shape[0]
            num = self.fre_compression_amplitude.shape
            fre_center = fre_domain_obj.fre_bin * np.arange(start, end, step)
            if fre_center.shape != num:
                print("Wrong")

            return fre_center

    def wave(self, X_quite_need_index, x):
        out_index = [val + self.split_quite_num / 2 for val in X_quite_need_index]  # 索引每一段的中间值
        out_index = out_index[0:-1]
        wave_out = []
        for i in range(len(X_quite_need_index) - 1):
            index_1 = X_quite_need_index[i] - 1
            index_2 = X_quite_need_index[i + 1] - 1
            X_list_small = sorted(x[index_1:index_2])
            len_X_list_small = len(X_list_small)
            index_head_X_list_small = round(len_X_list_small * self.a_hyper) - 1
            index_tail_X_list_small = round(len_X_list_small * self.b_hyper) - 1
            X_list_small_wave = X_list_small[index_tail_X_list_small] - X_list_small[index_head_X_list_small]
            wave_out.append(X_list_small_wave)

        output_x_index_wave = [out_index, wave_out]
        output_x_index_wave = list(list(i) for i in zip(*output_x_index_wave))
        return output_x_index_wave

    def subtract_wave(self, output_x_index_wave, text_sum_pro):
        miss_index = 5
        output_index = self.find_ratio_max_index(text_sum_pro, self.text_sum_pro_choose)
        output_x_wave_quite = output_x_index_wave[1][output_index[0]:output_index[1]+1]
        output_index_min = self.find_ratio_max_index_just_one(output_x_wave_quite, 1 - self.c_hyper)
        output_x_wave_quite_min_judge = output_x_wave_quite[output_index_min[0]]
        output_index_max = self.find_ratio_max_index_just_one(output_x_wave_quite, 1 - self.d_hyper)
        output_x_wave_quite_max_judge = output_x_wave_quite[output_index_max[0]]
        val_judge = output_x_index_wave[1][0:5]

        for i in range(miss_index, len(output_x_index_wave[1]), 1):
            temp = np.mean(output_x_index_wave[1][i-miss_index:i])
            val_judge.append(temp)

        SubtractWave = []

        for i in range(len(val_judge)):
            if val_judge[i] <= output_x_wave_quite_max_judge:
                temp = 0
            else:
                temp = val_judge[i] - output_x_wave_quite_max_judge
            SubtractWave.append(temp)

        output_list_index_SubtractWave = [output_x_index_wave[0], SubtractWave]

        return output_list_index_SubtractWave

    def __find_ratio_max_index(self, text_sum_pro):
        bool_judge = 0
        text_sum_pro_sort = sorted(text_sum_pro)
        ind_temp = round(len(text_sum_pro_sort) * (1 - self.text_sum_pro_choose)) - 1
        val_temp = text_sum_pro_sort[ind_temp]
        # ind_ValMax = text_sum_pro.index(max(text_sum_pro))
        ind_ValMax = np.where(text_sum_pro == max(text_sum_pro))[0][0]

        for i in range(ind_ValMax, 0, -1):
            if text_sum_pro[i] <= val_temp:
                out_index_left = i
                bool_judge = bool_judge + 1
                break

        for i in range(ind_ValMax, len(text_sum_pro) + 1, 1):
            if text_sum_pro[i] <= val_temp:
                out_index_right = i
                bool_judge = bool_judge + 1
                break

        if bool_judge != 2:
            print("Something wrong with you")

        output_index = [out_index_left, out_index_right]

        return output_index

    def __find_ratio_max_index_just_one(self, text_sum_pro, unforeseen_circumstances_threshold_length_ratio):
        text_sum_pro_sort = sorted(text_sum_pro)
        ind_temp = math.ceil(len(text_sum_pro_sort)*(1 - unforeseen_circumstances_threshold_length_ratio)) - 1
        val_temp = text_sum_pro_sort[ind_temp]
        # output_index = text_sum_pro.index(val_temp)
        output_index = np.where(text_sum_pro == val_temp)[0][0]
        output_index = [output_index]
        return output_index

    def fft(self, wave_file_path, N_zero_padding=None):
        x, FS_samplerate = sf.read(wave_file_path)
        if N_zero_padding is not None:
            x = np.array(fill_list(x, N_zero_padding, 0))
        time_domain_obj = self.node_time_domain(x, FS_samplerate)
        fre_domain_obj = self.node_fre_domain(x, FS_samplerate)
        return time_domain_obj, fre_domain_obj


def fill_list(my_list: list, length, fill=None):  # 使用 fill字符/数字 填充，使得最后的长度为 length
    my_list = list(my_list)
    if len(my_list) >= length:
        return my_list
    else:
        return my_list + (length - len(my_list)) * [fill]


def split_sum(X, step):   # 只针对一维
    fill_len = int(np.ceil(X.shape[0]/step)) * step
    my_list = fill_list(X, fill_len, 0)
    my_array = np.array(my_list)
    x = my_array.reshape((-1, step))
    out_com_amplitude = np.sum(x, axis=1)
    out_origin_amplitude = my_array
    return out_com_amplitude, out_origin_amplitude


def plot_save(x):
    plt.plot(range(len(x)), x)
    plt.savefig("check.png")