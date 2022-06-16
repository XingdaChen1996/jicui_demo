# fight for the bright future
# contend: 
# author: xingdachen
# time: 
# email: chenxingda@iat-center.com


import os
import shutil
import soundfile as sf
import pandas as pd


# def create_dir_class(path, dir_name="Model_pre"):
#     """
#
#     """
#     dir_path = os.path.join(path, dir_name)
#     dir_OK_path = os.path.join(dir_path, "OK")
#     dir_NG_path = os.path.join(dir_path, "NG")
#
#     if os.path.exists(dir_path):
#         print(f"The {dir_name} already exits")
#     elif os.path.exists(dir_OK_path):
#         print(f"The {dir_OK_path} already exits")
#     elif os.path.exists(dir_NG_path):
#         print(f"The {dir_NG_path} already exits")
#     else:
#         os.makedirs(dir_path)
#         os.makedirs(dir_OK_path)
#         os.makedirs(dir_NG_path)


def mymovefile(srcfile,dstpath):
    """
    srcfile ： 一个文件的path(若不存在就报错)
    dstpath ： copy srcfile文件到 dstpath文件夹下(若不存在就新建)
    """
    if not os.path.isfile(srcfile):
        print(f"Thr path :  {srcfile}  not exist!")
        return

    if not os.path.exists(dstpath):
        os.makedirs(dstpath)  # 创建路径
    shutil.move(srcfile, dstpath)  # 移动文件
    print("move %s -> %s" % (srcfile, dstpath))


def Anomaly_det_move_file(outliers, output_path, dstpath):

    dstpath_NG = os.path.join(dstpath, "NG")
    if os.path.exists(dstpath_NG):
        raise f"{dstpath_NG} already exist"
    os.makedirs(dstpath_NG)
    print("Now we make directory named NG")

    dstpath_OK = os.path.join(dstpath, "OK")
    if os.path.exists(dstpath_OK):
        raise f"{dstpath_OK} already exist"
    os.makedirs(dstpath_OK)
    print("Now we make directory named OK")

    for i in range(len(outliers)):
        if outliers[i] == 1:
            mymovefile(output_path[i], dstpath_NG)
        elif outliers[i] == 0:
            mymovefile(output_path[i], dstpath_OK)


class ML_read(object):

    def add_filename_rank(self, input_dir, specified_type, path_list):
        """
        如果一个文件下的文件名不是以(int)排好的，则添加(int)为末尾
        path_list:  所有的文件的list
        """
        # first step：判断所有的文件是否以(int)结尾
        add_filename_flag = False
        for item in path_list:
            index_left_parentheses = item.rfind("(")
            index_right_parentheses = item.rfind(")")
            if index_left_parentheses == -1 or index_right_parentheses == -1 or index_right_parentheses - index_left_parentheses <= 1:
                add_filename_flag = True
                break
            elif not item[index_left_parentheses+1 : index_right_parentheses].isdigit():
                add_filename_flag = True
                break

        # second step：改名字
        if add_filename_flag:
            xlsbpath = input_dir
            os.chdir(xlsbpath)  # 更改当前路径
            specified_type_len = len(specified_type)
            for i in range(len(path_list)):
                old = path_list[i]   # 旧文件名
                path_list[i] = path_list[i][:-specified_type_len] + f"({i+1})" + specified_type
                new = path_list[i]  # 新文件名
                os.rename(old, new)  # 重命名

        print("--------Your filename has been changed--------")

    def read_specified_file(self, input_dir, specified_type=".wav", flag=0):
        """
        读取一个文件夹下面的的指定文件，返回一个list
        :param input_dir: str
        :param specified_type: str
        :param flag: int 0 or 1
        0: 读取改文件夹下面第一层的所有 specified_type 文件
        1： 读取改文件夹下面所有层的所有 specified_type 文件
        :return: list
        所有指定文件的目录组成的list
        """
        output_path = []
        num_specified_type = len(specified_type)
        for curDir, dirs, files in os.walk(input_dir):
            for str_type in files:
                if str_type[-num_specified_type:] == specified_type:
                    temp = curDir + "\\" + str_type
                    output_path.append(temp.replace("\\", "/"))
            if flag == 0:
                break

        self.add_filename_rank(input_dir, specified_type, output_path)
        output_path.sort(key=lambda s: int(s[s.rfind("(") + 1: s.rfind(")")]))

        return output_path

    def read_excel_file(self, input_path, col_str=None):
        """
        :param input_path: str
        :param col_str: list
        读取指定的列，组成list；默认None，读取所有的列称为list
        :return:list
        """
        output_fea = []
        df = pd.read_excel(input_path)

        if col_str is None:
            col_str = list(df)[1:]

        for val in col_str:
            temp = list(df[val])
            output_fea.append(temp)
        output_fea = list(list(i) for i in zip(*output_fea))
        return output_fea


if __name__ == '__main__':
    ml_read_obj = ML_read()
    file_dir = __file__.split("source")[0] + r"input\training_data_set\01NG"
    output_path = ml_read_obj.read_specified_file(input_dir=file_dir, specified_type=".wav", flag=0)
    x, FS_samplerate = sf.read(output_path[0])
    print(str(output_path))
    file_path = os.path.join(__file__.split("source")[0], r"input\training_data_set\01NG.xlsx")
    # file_path = __file__.split("source")[0] + r"input\training_data_set\01NG.xlsx"
    col_str = ["T2", "T3"]
    file_path = os.path.abspath(file_path)
    fea = ml_read_obj.read_excel_file(file_path, col_str)
    print(fea)





