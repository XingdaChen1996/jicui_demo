# fight for the bright future
# contend: 
# author: xingdachen
# time: 
# email: chenxingda@iat-center.com
import copy
import os
import time
import numpy as np
import xml.dom.minidom
import pickle
from Gaussion_model import *
import sys

import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import pandas as pd

a = np.array([1,2,3, 4])
b = a[0:2]
c = a[0]

aa = [1, 2, 3]
bb = aa[0]

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

# 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x = np.linspace(0, 1, 1400)  # 时间是[0,1], 采样点的个数 N=1400, fs=1400, 频域分辨率 fbin=fs/N=1
# x = np.arange(0, 1, 1/1400)

# 设置需要采样的信号，频率分量有200，400和600
y = 7 * np.sin(2 * np.pi * 200 * x) + 5 * np.sin(2 * np.pi * 400 * x) + 3 * np.sin(2 * np.pi * 600 * x)

fft_y = fft(y)  # 快速傅里叶变换

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
print(normalization_y[400] * 2)
print(normalization_y[600] * 2)

#
#
# def max_min_normalization(data_list):
#     """
#     利用最大最小数将一组数据进行归一化输出
#     x_new = (x - min) / (max - min)
#     :param data_list:
#     :return:
#     """
#     normalized_list = []
#     max_min_interval = max(data_list) - min(data_list)
#     for data in data_list:
#         data = float(data)
#         new_data = (data - min(data_list)) / max_min_interval
#         normalized_list.append(round(new_data, 3))
#
#     return normalized_list
#
#
# def max_min_normalization(data, val_min=1):
#     Max_data = np.max(data)
#     Min_data = np.min(data)
#     Nor_data = (data - Min_data)/(Max_data - Min_data) + val_min
#     return Nor_data


# from sklearn.metrics import fbeta_score
#
# y_true = [0, 1, 2, 0, 1, 2]
# y_pred = [0, 2, 1, 0, 0, 1]
# a = fbeta_score(y_true, y_pred, average='macro', beta=0.5)
#
# b = fbeta_score(y_true, y_pred, average='micro', beta=0.5)
#
# c = fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
#
# d = fbeta_score(y_true, y_pred, average=None, beta=0.5)

# plt.subplot(121)
# plt.plot(range(len(testX[0][200:500])), testX[0][200:500])
# plt.title('good')
#
# plt.subplot(122)
# plt.plot(range(len(testX[-1][200:500])), testX[-1][200:500], 'black')
# plt.title('bad')
# plt.savefig("badandgood.png")
# # hhhhha = []
# # for i in range(len(testX)):
# #     temp = int(np.where(testX[i] == max(testX[i]))[0])
# #     hhhhha.append(temp)
# #
# # list1 = hhhhha
# # filename = "ha.txt"
# # with open(filename, "w") as file_obj:
# #     for i in range(len(list1)):
# #         file_obj.write(str(list1[i])+"\n")


# print(sys.path)
#
# hha = sys.path
# speech_split_obj = Gaussian_Model()
# saving_model_path = __file__.split("source")[0] + r"input\saving_model\model_obj.pkl"
#
#
#
#
# print(1)


# with open(saving_model_path, "wb") as f:
#     pickle.dump(speech_split_obj, f)
#
#
# with open(saving_model_path, "rb") as f:
#     hh = pickle.load(f)

# import time
#
# s_time = time.time()
#
# for i in range(1000):
#
#     print(i)
#
# c_time = time.time()
#
# ss_time = c_time - s_time
#
# print('%.3f秒'%ss_time)
#
# print('执行结束！！！')
#
#
# def fun(cv_predictions, label):
#     true_positives = 0
#     false_positives = 0
#     false_negatives = 0
#     for i in cv_predictions:
#         # print(i)
#         for j in label:
#             if i == 1 and j == 1:
#                 true_positives += 0
#             elif i == 1 and j == 0:
#                 false_positives += 0
#             elif i == 0 and j == 1:
#                 false_negatives += 1
#     return true_positives, false_positives, false_negatives


# a = np.array([True, False])
# print(a)
# b = a + 0
# print(b)


# for i in np.arange(0,5,0.5):
#     print(i)


# a = np.array([[1,2],[3,4]])
#
# hh = sum(a)
#
#
# # 按行相加，并且保持其二维特性
# print(np.sum(a, axis=1, keepdims=True))
#
# # 按行相加，不保持其二维特性
# print(np.sum(a, axis=1))


# s = [r"F:\chen_work_备份\ling_ding\Step_1\\01\\01NG\\tu (2).wav", 'tu (1).wav','tu (10).wav']
# h1 = s[0].rfind("(") + 1
# h2 = s[0].rfind(")")
#
# gg = [ int(s[i][s[i].rfind("(")+1 : s[i].rfind(")")] )  for i in range(len(s))]
#
#
# new = sorted(s, key=lambda s: int(s[s.rfind("(")+1 : s.rfind(")")]))


# new = sorted(s, key = lambda )
#
#
# sort_goods = sorted(goods, key=lambda x: x[1], reverse=True)
# new = sorted(s,key = lambda i:int(re.match(r'(\d+)',i).group()))


# a=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
# b = a
# a = list(list(i) for i in zip(*a))


# i = 0
# numpy.transpose()
# file_dir = r'F:\chen_work_备份\ling_ding\Step_1\01'
# for curDir, dirs, files in os.walk(file_dir):
#     i += 1
#     print("====================")
#     print("现在的目录：" + curDir)
#     print("该目录下包含的子目录：" + str(dirs))
#     print("该目录下包含的文件：" + str(files))
#     print(i)
#     print(i)
#
#
#
# print(output_list_index_SubtractWave[1][0])
# print(output_list_index_SubtractWave[1][-1])
# print(output_list_index_SubtractWave[1][-2])
#
# print(right_queue[0])
# print(right_queue[-1])
# print(right_queue[-2])


# a = [[1, 2], 2, 3]
# b = copy.copy(a)
# c = copy.deepcopy(a)
# d = a
# e = a[:]
#
# print(id(a[0]))
# print(id(b[0]))
# print(id(c[0]))
# print(id(d[0]))
# print(id(e[0]))


# a = np.array([1, 2, 3])
# # b = copy.copy(a)
# b = a[:]
#
# print(id(a[0]))
# print(id(a[1]))
# print(id(a[2]))
#
#
# [0.9950327334180954, 0.9972930616896993, 0.9961616153630257]
#
# [0.9950327334180954, 0.9972930616896993, 0.9961616153630257]


# fight for the bright future
# contend:
# author: xingdachen
# time:
# email: chenxingda@iat-center.com


# import numpy as np
# from scipy import stats
# import time
#
#
# class Gaussian_Model(object):
#     def __init__(self):
#         self.mu = None  # training set mean
#         self.sigma = None  # training set std
#         self.Threshold = None  # validation set std
#         self.F1 = None  # validation set std
#
#     def estimateGaussian(self, trainX):
#         m, n = trainX.shape
#         self.mu = np.sum(trainX, axis=0)/m
#         self.sigma = (np.sum((trainX - self.mu)**2, axis=0)/m)**0.5
#
#     def Gaussian_pro(self, testX):
#
#         SubtractWave_text_index_probability = stats.norm(self.mu, self.sigma).pdf(testX)  # 计算累和(假设特征独立)
#         out_pro = np.ones_like(SubtractWave_text_index_probability[:, 0])
#         for i in range(len(SubtractWave_text_index_probability[0, :])):
#             out_pro *= SubtractWave_text_index_probability[:, i]
#
#         return out_pro
#
#     def Confusion_matrix(self, cv_predictions, label):
#         true_positives = sum((cv_predictions == 1) & (label == 1))
#         false_positives = sum((cv_predictions == 1) & (label == 0))
#         false_negatives = sum((cv_predictions == 0) & (label == 1))
#         precision = true_positives / (true_positives + false_positives)
#         recall = true_positives / (true_positives + false_negatives)
#         F1 = 2 * precision * recall / (precision + recall)
#         precision_recall_F1 = [precision, recall, F1]
#         return precision_recall_F1
#
#
#
#     def SelectThreshold(self, testX, label):
#         testX = np.array(testX)
#         label = np.array(label)
#         pval = self.Gaussian_pro(testX)
#         bestEpsilon = 0
#         bestF1 = 0
#         F1 = 0
#         stepsize = (max(pval) - min(pval)) / 1000
#
#         # def fun(cv_predictions, label):
#         #     true_positives = sum(((cv_predictions * label) == 1))
#         #     false_positives = sum((cv_predictions * (label + 1)) == 1)
#         #     false_negatives = sum(((cv_predictions + 1) * label) == 1)
#         #
#         #     true_positives_1 = sum((cv_predictions == 1) & (label == 1))
#         #     false_positives_1 = sum((cv_predictions == 1) & (label == 0))
#         #     false_negatives_1 = sum((cv_predictions == 0) & (label == 1))
#         #
#         #     # false_negatives_1 = cv_predictions[[(cv_predictions == 0) & (label == 1)]].shape[0]
#         #
#         #     # print(true_positives == true_positives_1)
#         #     # print(false_positives == false_positives_1)
#         #     # print(false_negatives == false_negatives_1)
#         #
#         #     return true_positives, false_positives, false_negatives
#
#         for epsilon in np.arange(min(pval), max(pval)+stepsize, stepsize):
#             cv_predictions = (pval < epsilon) + 0
#             F1 = self.Confusion_matrix(cv_predictions, label)[-1]
#
#             # true_positives, false_positives, false_negatives = fun(cv_predictions, label)
#             # precision = true_positives / (true_positives + false_positives)
#             # recall = true_positives / (true_positives + false_negatives)
#             # F1 = 2 * precision * recall / (precision + recall)
#             # if not F1 == F11:
#             #     print("Wrong")
#             #     print(F1)
#
#             if F1 > bestF1:  # nan 比较运算均是0
#                 bestF1 = F1
#                 bestEpsilon = epsilon
#
#         self.Threshold = bestEpsilon
#         self.F1 = bestF1
#         return bestEpsilon, bestF1
#
#     def Gau_prediect(self, testX, label=None):
#         testX = np.array(testX)
#
#         def fun(cv_predictions, label):
#
#             # true_positives = sum(((cv_predictions * label) == 1) + 0)
#             # false_positives = sum((cv_predictions * (label + 1)) == 1 + 0)
#             # false_negatives = sum(((cv_predictions + 1) * label) == 1 + 0)
#
#             true_positives = cv_predictions[[(cv_predictions == 1) & (label == 1)]].shape[0]
#             false_positives = cv_predictions[[(cv_predictions == 1) & (label == 0)]].shape[0]
#             false_negatives = cv_predictions[[(cv_predictions == 0) & (label == 1)]].shape[0]
#             return true_positives, false_positives, false_negatives
#
#         pro = self.Gaussian_pro(testX)
#         outliers = (pro < self.Threshold) + 0
#
#         if label is None:
#             precision_recall_F1 = None
#             return outliers, precision_recall_F1
#         label = np.array(label)
#
#         F11 = self.Confusion_matrix(outliers, label)
#
#         true_positives, false_positives, false_negatives = fun(outliers, label)
#         precision = true_positives / (true_positives + false_positives)
#         recall = true_positives / (true_positives + false_negatives)
#         F1 = 2 * precision * recall / (precision + recall)
#         precision_recall_F1 = [precision, recall, F1]
#
#         if not precision_recall_F1 == F11:
#             print("Wrong")
#
#         return outliers, precision_recall_F1


# class node_uniform_fre_compression(object):
#     def __init__(self, fre_domain_obj, fre_len=5):  # fre_len=5 以fu-fr=5hz
#         self.fre_len = fre_len
#         self.step = int(np.ceil(fre_len / fre_domain_obj.fre_bin))  # step 求和的步长
#         self.fre = split_sum(fre_domain_obj.normalization_half_y, self.step)
#         # index = np.arange(0, fre_domain_obj.num, self.step)[1:]
#         # if fre_domain_obj.normalization_y.ndim ==1:
#         #     # np.sum(fre_domain_obj.normalization_y, index, axis=0)
#         #     split_sum()
#         #     np.reshape(fre_domain_obj.normalization_y, (-1, self.step))
#         #     hh = np.split(fre_domain_obj.normalization_y, index, axis=0)[:-1]
#         #     fre = np.array(hh)
#         #     np.cumsum()
#         # elif fre_domain_obj.normalization_y.ndim ==2:
#         #     fre = np.array(np.split(fre_domain_obj.normalization_y, index, axis=1))
#         #
#         # self.fre = np.sum(fre)
#         # f_center = np.sum()
#         pass

# if __name__ == '__main__':
#     print("------------text example ------------")
#     # 加载模型
#     speech_pre_time_obj = speech_pre_time()
#     saving_model_path = __file__.split("source")[0] + r"input\wu_exp\TOcheng\saving_model\model_obj.pkl"
#
#     with open(saving_model_path, "rb") as f:
#         obj_model = pickle.load(f)
#
#     ml_read_obj = ML_read()
#     file_dir_0 = __file__.split("source")[0] + r"input\wu_exp\TOcheng\testing_data_set\label0"
#     file_dir_1 = __file__.split("source")[0] + r"input\wu_exp\TOcheng\testing_data_set\label1"
#     output_path_0 = ml_read_obj.read_specified_file(input_dir=file_dir_0, specified_type=".wav", flag=0)
#     label_0 = len(output_path_0) * [0]
#     output_path_1 = ml_read_obj.read_specified_file(input_dir=file_dir_1, specified_type=".wav", flag=0)
#     label_1 = len(output_path_1) * [1]
#
#     output_path = output_path_0 + output_path_1
#     label = label_0 + label_1
#
#     # 语音预处理
#     speech_pre_time_obj = speech_pre_time()
#
#     output_fea = []  # feature
#     for i in range(len(output_path)):
#         time_domain_obj, fre_domain_obj = speech_pre_time_obj.fft(output_path[i], N_zero_padding)
#         fea = [fre_domain_obj.sort_normalization_x[0]]
#         output_fea.append(fea)
#     output_fea = np.array(output_fea)
#
#     outliers, precision_recall_F1, confuse_m = obj_model.Gau_prediect(output_fea[100:, :], label[100:], flag_joint_pro)
#
#     print("OKOKO")
