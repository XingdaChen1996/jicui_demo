# fight for the bright future
# contend: 
# author: xingdachen
# time: 
# email: chenxingda@iat-center.com


import module.Speech_preprocessing as SP
from module.ML_file_read import *
from module.Gaussion_model import *
import numpy as np
import argparse
import pickle
import os
import operator


# 标签参数：   如果你有个没有label的数据需要你取完成预测
# 算法参数：   flag_joint_pro = 0 代表：pro_min;   flag_joint_pro = 1/None, 或者你不表示这个参数，代表：pro_multiply
# path参数：  input_dir：输入的文件夹;   flag=0, 读取改文件夹下面第一层的所有的specified_type,
#                                    flag=1, 读取改文件夹下面所有层的所有的specified_type

# 加载模型
def Anomaly_det(wave_path, label_path):
    Base_Dir = os.path.dirname(os.path.abspath(__file__))
    saving_model_path = os.path.join(os.path.dirname(Base_Dir), "model_obj.pkl")

    print(f"--------your model path  :{saving_model_path}------------")
    with open(saving_model_path, "rb") as f:
        obj_model = pickle.load(f)

    if os.path.isdir(wave_path):
        print("--------now this is a directory------------")
        ml_read_obj = ML_read()
        output_path = ml_read_obj.read_specified_file(input_dir=wave_path, specified_type=".wav", flag=0)

        if label_path is not None:  # 有无输入label
            label = ml_read_obj.read_excel_file(label_path)
            label = np.array(label)
            label = label.reshape((label.shape[0]))
            print("label_path has been loaded")
        else:
            label = None
        # label1 = 200 * [0] + 50 * [1] + 50 * [1]
        # print(operator.eq(list(label), label1))

        # 语音预处理
        speech_pre_time_obj = SP.speech_pre_time()
        output_fea = []  # feature
        for i in range(len(output_path)):
            time_domain_obj, fre_domain_obj = speech_pre_time_obj.fft(output_path[i],
                                                                      obj_model.dic_par["N_zero_padding"])
            fea = [fre_domain_obj.sort_normalization_x[0]]
            output_fea.append(fea)
        output_fea = np.array(output_fea)

        #  创建文件夹，将分好类的文件放到这个文件夹里面
        dstpath = os.path.join(wave_path, "Model_pre")   # wave_path 是文件夹或者.wav路径
        if os.path.exists(dstpath):
            shutil.rmtree(dstpath)
        os.makedirs(dstpath)  # 创建路径
        print("----Now we create a directory named Model_pre----")

        if label is not None:
            outliers, confuse_m = obj_model.Gau_prediect(output_fea, label, obj_model.dic_par["flag_joint_pro"])
            output_path = np.array(output_path)  # output_path 这边是文件夹下所有的.wav路径

            Anomaly_det_move_file(outliers, output_path, dstpath)

            outliers = output_path[outliers == 1]
            return outliers, confuse_m.confuse_matrix
        else:
            outliers = obj_model.Gau_prediect(output_fea, label, obj_model.dic_par["flag_joint_pro"])
            output_path = np.array(output_path)

            Anomaly_det_move_file(outliers, output_path, dstpath)

            outliers = output_path[outliers == 1]
            return outliers, None

    print("--------now this is a wave path------------")
    output_path = wave_path

    # 语音预处理
    speech_pre_time_obj = SP.speech_pre_time()
    time_domain_obj, fre_domain_obj = speech_pre_time_obj.fft(output_path, obj_model.dic_par["N_zero_padding"])
    fea = [fre_domain_obj.sort_normalization_x[0]]
    output_fea = np.array(fea).reshape([1, -1])
    outliers = obj_model.Gau_prediect(testX=output_fea, label=None, flag_joint_pro=obj_model.dic_par["flag_joint_pro"])
    return outliers, None


if __name__ == '__main__':

    # str = r"C:\Users\25760\Desktop\matlab2pyhon4.12\matlab2pyhon\input\wu_exp\TOcheng\testing_data_set\test(0).wav"
    # outliers = Anomaly_det(str)

    parser = argparse.ArgumentParser(description="Anomaly detection")
    parser.add_argument('-p', '--wave_path', type=str, metavar="", required=True,
                        help='path of a wave')

    parser.add_argument('-l', '--label_path', type=str, metavar="", required=False, default=None,
                        help='label of a wave,type .excel')

    # 你设置 required=False， 若在 python命令行 不写参数的话，参数没有赋值则为 default 后的参数
    # 你设置 required=True，  若在 python命令行 不写参数的话，直接报错
    # 注意啊： parser.add_argument("-h")会出错， 不允许出现 -h；应为这个库自带-h代表为help命令

    args = parser.parse_args()
    outliers, confuse_matrix = Anomaly_det(args.wave_path, args.label_path)
    if confuse_matrix is None and not os.path.isdir(args.wave_path):
        if outliers[0] == 0:
            result = "OK"
        else:
            result = "NG"
        # args.radius 中的 radius 代表 parser.add_argument 中的 '--radius'
        # args.height 中的 radius 代表 parser.add_argument 中的 '--height'
        print(f"this wave label: {outliers}  {result}")
    elif confuse_matrix is None and os.path.isdir(args.wave_path):
        print(f"predicted outliers label: \n {outliers}")
    else:
        print(f"predicted outliers label: \n {outliers}")
        print(f"confuse_matrix:\n {confuse_matrix}")
