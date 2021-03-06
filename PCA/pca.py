#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
"""
PCA降噪算法的实现和应用

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
    """将原始数据进行投影降维基变化

    Args:
        dataMat: 原始数据
        topNfeat: 需要降到的维数

    Returns:
        projDataMat: 去中心化后的原始数据对应的降维的数据
        reconDat: 原始数据降维后所对应的原始坐标
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


def replaceNanWithMean(dataMat):
    """将原始数据中nan值转换成均值

    Args:
        dataMat: 原始数据

    Returns:
        去除掉nan值得数据
    """
    numFeat = shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:, i].A))[0], i])
        dataMat[nonzero(isnan(dataMat[:, i].A))[0], i] = meanVal
    return dataMat

if __name__ == '__main__':
    # 使用testSet测试pca
    # dataMat = loadDataSet("testSet.txt")
    # projMat, reconMat = pca(dataMat, 1)
    # showData(dataMat, reconMat)

    # 使用secom.data测试pca
    dataMat = loadDataSet("secom.data", " ")
    dataMat = replaceNanWithMean(dataMat)
    print("dataMat:\n", str(dataMat))
    varVal = var(dataMat, axis=0)                                               # 求解每个特征的方差
    # print("varVal:\n", str(varVal))
    sortVarInd = argsort(varVal[0]).A[0].tolist()                               # 对方差进行排序
    sortVarInd = sortVarInd[: -21: -1]                                          # 从大到小找出20个方差
    print("sortVarInd:", str(sortVarInd), "type:", type(sortVarInd))
    varSum = sum(varVal[0], axis=1)[0, 0]                                       # 求解总方差
    print("varSum:\n", str(varSum))
    n = shape(dataMat)[1]
    print("n:\n", n)
    sumTmp = 0.0
    for i in range(20):                                                         # 循环累计20个方差
        sumTmp += varVal[0, sortVarInd[i]]
        print("ind:", sortVarInd[i], "var:", str(varVal[0, sortVarInd[i]]))
        perc = sumTmp / varSum
        print("i:", i, "perc:", str(perc))                                      # 求累计后方差占总方差的大小

    # 通过上述求解能够知道，前六个方差的累计已经超过了95%，所以设topNfeat为6
    # 只留6个主成分
    projMat, reconMat = pca(dataMat, 6)
    print("projMat:\n", str(projMat), "\nreconMat:\n", str(reconMat))