# fight for the bright future
# contend: 
# author: xingdachen
# time: 
# email: chenxingda@iat-center.com

import numpy as np
from scipy import stats
# from sklearn.metrics import fbeta_score
import matplotlib.pyplot as plt


# def max_min_normalization(data, val_min=0.5):
#     Max_data = np.max(data)
#     Min_data = np.min(data)
#     Nor_data = (data - Min_data)/(Max_data - Min_data) + val_min
#     return Nor_data


class Gaussian_Model(object):
    """
    先 self.estimateGaussian(x)
    其中x：
    横坐标是 sample num(最高维度)
    纵坐标是 fea num
    """
    dic_par = None  # 其他一些参数设置在这里，目的为了保存好model需要的参数

    class node_confuse_matrix(object):
        def __init__(self, cv_predictions, label):
            self.confuse_matrix = np.zeros((2, 2))
            true_positives = sum((cv_predictions == 1) & (label == 1))
            false_positives = sum((cv_predictions == 1) & (label == 0))
            false_negatives = sum((cv_predictions == 0) & (label == 1))
            true_negatives = sum((cv_predictions == 0) & (label == 0))
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            F1 = 2 * precision * recall / (precision + recall)
            acc = sum((cv_predictions == label) + 0) / label.shape[0]
            self.acc_precision_recall_F1 = [acc, precision, recall, F1]
            self.confuse_matrix[0][0] = true_positives
            self.confuse_matrix[0][1] = false_positives
            self.confuse_matrix[1][0] = false_negatives
            self.confuse_matrix[1][1] = true_negatives

    def __init__(self):
        self.mu = None  # training set mean
        self.sigma = None  # training set std
        self.Threshold = None  # validation set std
        self.F1 = None  # validation set std

    def estimateGaussian(self, trainX):
        trainX = np.array(trainX)
        m, n = trainX.shape
        self.mu = np.sum(trainX, axis=0) / m
        self.sigma = (np.sum((trainX - self.mu) ** 2, axis=0) / m) ** 0.5

    def Gaussian_pro(self, testX, flag_joint_pro=1):   #     def Gaussian_pro(self, testX, flag_joint_pro=None):
        if testX.ndim != 2:
            raise ValueError("the shape of testX must be (m,n)")

        SubtractWave_text_index_probability = stats.norm(self.mu, self.sigma).pdf(testX)  # 计算累和(假设特征独立)
        if flag_joint_pro == 0:  # 时间序列的异常检测，不用joint_probability
            out_pro = SubtractWave_text_index_probability.min(axis=1)
            return out_pro

        out_pro = np.ones_like(SubtractWave_text_index_probability[:, 0])
        for i in range(len(SubtractWave_text_index_probability[0, :])):
            nor_pro = SubtractWave_text_index_probability[:, i]
            out_pro *= nor_pro
        return out_pro

    def Confusion_matrix(self, cv_predictions, label):
        true_positives = sum((cv_predictions == 1) & (label == 1))
        false_positives = sum((cv_predictions == 1) & (label == 0))
        false_negatives = sum((cv_predictions == 0) & (label == 1))
        precision = true_positives / (true_positives + false_positives)  # 0/0 无所谓为nan，可以忽略，最后也不会选这个
        recall = true_positives / (true_positives + false_negatives)
        F1 = 2 * precision * recall / (precision + recall)
        acc = sum((cv_predictions == label) + 0) / label.shape[0]
        acc_precision_recall_F1 = [acc, precision, recall, F1]
        return acc_precision_recall_F1

    def SelectThreshold(self, testX, label, flag_joint_pro=1):   #  def SelectThreshold(self, testX, label, flag_joint_pro=None):
        testX = np.array(testX)
        label = np.array(label)
        pval = self.Gaussian_pro(testX, flag_joint_pro)
        bestEpsilon = 0
        bestF1 = 0
        F1 = 0
        stepsize = (max(pval) - min(pval)) / 1000
        # stepsize_all = np.linspace(min(pval), max(pval), 1000)
        for epsilon in np.arange(min(pval), max(pval) + stepsize, stepsize):
            cv_predictions = (pval < epsilon) + 0  # 小于设为1，异常样本，与一般的分类任务不同
            F1 = self.Confusion_matrix(cv_predictions, label)[-1]

            if F1 > bestF1:  # nan 比较运算均是0
                bestF1 = F1
                bestEpsilon = epsilon

        self.Threshold = bestEpsilon
        self.F1 = bestF1
        return bestEpsilon, bestF1

    def Gau_prediect(self, testX, label=None, flag_joint_pro=1):  #  def Gau_prediect(self, testX, label=None, flag_joint_pro=None):
        testX = np.array(testX)
        if testX.ndim != 2:
            raise ValueError("the shape of testX must be (m,n)")
        pro = self.Gaussian_pro(testX, flag_joint_pro)
        outliers = (pro < self.Threshold) + 0

        if label is None:
            return outliers

        label = np.array(label)
        if label.ndim != 1:
            raise ValueError("the shape of label must be (n,)")
        # acc_precision_recall_F1 = self.Confusion_matrix(outliers, label)
        node_confuse_matrix_obj = self.node_confuse_matrix(outliers, label)
        return outliers, node_confuse_matrix_obj

