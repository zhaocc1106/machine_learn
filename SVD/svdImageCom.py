#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
利用SVD算法对图像进行压缩

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/8/9 22:47
"""
from numpy import *

def loadData(file):
    """从文件中加载数据

    Args:
        file: 文件名

    Returns:
        数据矩阵
    """
    fr = open(file)
    dataSet = []
    for line in fr.readlines():
        data = []
        for i in range(32):
            data.append(int(line[i]))
        dataSet.append(data)
    return mat(dataSet)


def printMat(dataMat, thresh = 0.8):
    """将矩阵打印出来

    Args:
        dataMat: 数据矩阵
        thresh: 阈值

    """
    m, n = shape(dataMat)
    for i in range(m):
        for j in range(n):
            if float(dataMat[i, j]) > thresh:
                print("1", end="")
            else:
                print("0", end="")
        print("\n")


if __name__ == '__main__':
    dataMat = loadData("0_5.txt")
    printMat(dataMat)
    U, Sigma, VT = linalg.svd(dataMat)              # 奇异值分解
    print("Sigma:\n", str(Sigma))
    Sigma2 = Sigma ** 2
    print("Sigma ** 2:\n", str(Sigma2))
    SumSigma2 = sum(Sigma2)                         # 计算奇异值平方和
    print("SumSigma2:\n", SumSigma2)
    MainSum = 0.95 * SumSigma2                      # 计算95%的奇异值平方和
    print("MainSum:\n", MainSum)
    print(sum(Sigma2[: 6]))                         # 求出来6个奇异值就可以满足95%，所以我们只留6个奇异值

    Sig6 = mat(eye(6) * Sigma[: 6])
    print("Sig6:\n", str(Sig6))
    dataMatCom = U[:, : 6] * Sig6 * VT[: 6, :]      # 求保留6个最大奇异值对应的数据矩阵，图像已经从32 * 32个像素点压缩成
                                                    # 32 * 6 * 2 + 6 * 6个像素点
    print("dataMatCom:\n")
    printMat(dataMatCom)