# fight for the bright future
# contend:
# author: xingdachen
# time:
# email: chenxingda@iat-center.com

from module.Speech_preprocessing import *
from module.ML_file_read import *
import matplotlib.pyplot as plt
from module.Gaussion_model import *
import numpy as np
import pickle
import os
import argparse
import math

flag_joint_pro = 1  # hyper1 是否使用联合概率分布
allocate_OK = [0.6, 0.8, 1]  # hyper(不需要保存) training, val, test data set
allocate_NG = [0, 0.5, 1]  # hyper(不需要保存) training, val, test data set

dic_par = {}
dic_par["flag_joint_pro"] = flag_joint_pro

Base_Dir = os.path.dirname(os.path.abspath(__file__))
file_dir_all = os.path.join(os.path.dirname(Base_Dir), "manually_label")
file_dir_OK = os.path.join(os.path.dirname(Base_Dir), "manually_label", "OK")
file_dir_NG = os.path.join(os.path.dirname(Base_Dir), "manually_label", "NG")

ml_read_obj = ML_read()

try:
    out_path_TrainValTest = ml_read_obj.read_specified_file(input_dir=file_dir_all, specified_type=".wav", flag=1)
    output_path_OK = ml_read_obj.read_specified_file(input_dir=file_dir_OK, specified_type=".wav", flag=0)
    output_path_NG = ml_read_obj.read_specified_file(input_dir=file_dir_NG, specified_type=".wav", flag=0)
except Exception as result:
    print(result)
    raise f"something wrong with {file_dir_all} or {file_dir_OK} or {file_dir_NG}"

output_path_train = output_path_OK[:int(len(output_path_OK)*allocate_OK[0])] + output_path_NG[:int(len(output_path_NG)*allocate_NG[0])]
output_path_val = output_path_OK[int(len(output_path_OK)*allocate_OK[0]):int(len(output_path_OK)*allocate_OK[1])] + output_path_NG[int(len(output_path_NG)*allocate_NG[0]):int(len(output_path_NG)*allocate_NG[1])]
output_path_test = output_path_OK[int(len(output_path_OK)*allocate_OK[1]):] + output_path_NG[int(len(output_path_NG)*allocate_NG[1]):]

saving_model_path = os.path.join(os.path.dirname(Base_Dir), "model_obj.pkl")

# 读取训练集的wav文件路径,并且排序(origin data)
print("------------training pro------------")
ml_read_obj = ML_read()
output_path = output_path_train
# 语音预处理
speech_pre_time_obj = speech_pre_time()
N_zero_padding = None
temp = []
for i in range(len(out_path_TrainValTest)):
    x, FS_samplerate = sf.read(out_path_TrainValTest[i])
    temp.append(len(x))
    # time_domain_obj, fre_domain_obj = speech_pre_time_obj.fft(out_path_TrainValTest[i])
    # temp.append(len(fre_domain_obj.normalization_half_y))
N_zero_padding = max(temp)    # hyper2

dic_par["N_zero_padding"] = N_zero_padding

output_fea = []  # feature
for i in range(len(output_path)):
    time_domain_obj, fre_domain_obj = speech_pre_time_obj.fft(output_path[i], dic_par["N_zero_padding"])
    # plan 1
    # node_uniform_fre_compression_obj = speech_pre_time_obj.node_uniform_fre_compression(fre_domain_obj, fre_len=5)
    # index = np.argmax(node_uniform_fre_compression_obj.fre_compression_amplitude)
    # fre_center_correspond_amp_max = node_uniform_fre_compression_obj.fre_center[index]
    # amp_max = node_uniform_fre_compression_obj.fre_compression_amplitude[index]
    # # fea = [fre_center_correspond_amp_max, amp_max]
    # fea = [fre_center_correspond_amp_max]

    # plan 2
    fea = [fre_domain_obj.sort_normalization_x[0]]

    output_fea.append(fea)
    # output_fea.append(fre) ; plot_save(output_fea[0])
    # node_uniform_fre_compression_obj.fre_compression_amplitude[index+1]
output_fea = np.array(output_fea)

# Gaussian_Model
speech_split_obj = Gaussian_Model()
speech_split_obj.estimateGaussian(output_fea)


# Validation file

# 读取验证集的wav文件(origin data)
print("------------Validation pro------------")

output_path = output_path_val
label = len(output_path_OK[int(len(output_path_OK)*allocate_OK[0]):int(len(output_path_OK)*allocate_OK[1])]) * [0] +  len(output_path_NG[int(len(output_path_NG)*allocate_NG[0]):int(len(output_path_NG)*allocate_NG[1])]) * [1]
# 语音预处理
speech_pre_time_obj = speech_pre_time()

output_fea = []  # feature
for i in range(len(output_path)):
    time_domain_obj, fre_domain_obj = speech_pre_time_obj.fft(output_path[i], dic_par["N_zero_padding"])
    fea = [fre_domain_obj.sort_normalization_x[0]]
    output_fea.append(fea)

output_fea = np.array(output_fea)

# Validation process and saving model to "input\saving_model"
Threshold, F1 = speech_split_obj.SelectThreshold(output_fea, label, dic_par["flag_joint_pro"])
# vv = Gaussian_Model()    vv.estimateGaussian(output_fea[200:,:])
# vv = Gaussian_Model()    hh.estimateGaussian(output_fea[:200,:])

# 保存模型

# saving_model_path = __file__.split("source")[0] + r"input\wu_exp\TOcheng\saving_model\dic_par.pkl"
# with open(saving_model_path, "wb") as f:
#     pickle.dump(dic_par, f)

speech_split_obj.dic_par = dic_par
with open(saving_model_path, "wb") as f:
    pickle.dump(speech_split_obj, f)


print("------------testing pro ------------")
# 加载模型
speech_pre_time_obj = speech_pre_time()

# Base_Dir = os.path.dirname(os.path.abspath(__file__))
# saving_model_path = Base_Dir + "\model_obj.pkl"
with open(saving_model_path, "rb") as f:
    obj_model = pickle.load(f)

output_path = output_path_test
label = len(output_path_OK[int(len(output_path_OK)*allocate_OK[1]):]) * [0] + len(output_path_NG[int(len(output_path_NG)*allocate_NG[1]):]) * [1]

# 语音预处理
speech_pre_time_obj = speech_pre_time()
output_fea = []  # feature
for i in range(len(output_path)):
    time_domain_obj, fre_domain_obj = speech_pre_time_obj.fft(output_path[i], obj_model.dic_par["N_zero_padding"])
    fea = [fre_domain_obj.sort_normalization_x[0]]
    output_fea.append(fea)
output_fea = np.array(output_fea)

outliers, confuse_m = obj_model.Gau_prediect(output_fea, label, obj_model.dic_par["flag_joint_pro"])
print("OKOKOK, Model has been trained, you can try to predict '.wav' in the directory pre_pro")
print("model_obj.pkl have been saved")
print(f"model_obj.pkl model confuse_matrix: {confuse_m.confuse_matrix}")
print(f"model_obj.pkl model acc_precision_recall_F1: {confuse_m.acc_precision_recall_F1}")


