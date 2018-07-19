#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
"""


Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/7/17 19:42
"""
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim = '\t'):
    """从数据文件中加载原始数据

    Args:
        fileName: 文件名
        delim: 数据分隔符号

    Returns:
        数据矩阵
    """
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    dataSet = [list(map(float, line)) for line in stringArr]
    return mat(dataSet)

def pca(dataMat, topNfeat):
    """

    Args:
        dataMat:
        topNfeat:

    Returns:

    """
    meanVals = mean(dataMat, axis=0)                            # 求解均值
    meanRemoved = dataMat - meanVals                            # 去中心化
    covMat = cov(meanRemoved, rowvar=0)                         # 求解协方差矩阵
    print("covMat:\n", str(covMat))
    eigVals, eigVects = linalg.eig(mat(covMat))                 # 计算特征向量和特征值
    print("eigVals:\n", str(eigVals))
    print("eigVects:\n", str(eigVects))
    eigValInd = argsort(eigVals)                                # 对特征值排序，求得排序ind
    eigValInd = eigValInd[: -(topNfeat + 1): -1]                # 倒序找出topNfeat个值
    print("eigValInd:", str(eigValInd))
    projEigVect = eigVects[:, eigValInd]                        # 找出用于投影降维的特征向量组成基变化矩阵
    print("projEigVect:\n", str(projEigVect))
    projDataMat = meanRemoved * projEigVect                     # 求解原始数据投影降维基变化后所对应的数据
    print("projDataMat:\n", str(projDataMat))
    reconMat = projDataMat * projEigVect.T + meanVals           # 求解降维基变化后点所对应在原始坐标轴（基）的坐标
    print("reconMat:\n", str(reconMat))
    return projDataMat, reconMat

def showData(dataMat, reconDataMat):
    """展示原始数据以及降维基变化后数据

    Args:
        dataMat: 原始数据
        reconDataMat: 降维基变化后的数据
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].T.A[0], dataMat[:, 1].T.A[0], marker='^', s=10)
    ax.scatter(reconDataMat[:, 0].T.A[0], reconDataMat[:, 1].T.A[0], marker='o', s=10, c='red')
    plt.show()

if __name__ == '__main__':
    dataMat = loadDataSet("testSet.txt")
    projMat, reconMat = pca(dataMat, 1)
    showData(dataMat, reconMat)
