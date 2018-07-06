#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
"""
kNN算法的实现

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/7/6 14:42
"""
from numpy import *
import operator

def createDateSet():
    group = array([[1.0], [1.1], [0.1], [0.0]])
    labels = array(['A', 'A', 'B', 'B'])
    return group, labels


def classify0(inX, dataSet, labels, k):
    """根据输入数据对该数据进行分类

    通过kNN距离计算公式，计算并找到inX距离dataSet中最近的数据群，并把该数据群的标签作为inX的分类标签

    Args:
        inX: 需要分类的数据
        dataSet: 训练数据集
        labels: 训练数据的标签
        k: kNN的k参数

    Returns:
        分类标签
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis = 1)
    distance = sqDistance ** 0.5                                    # 计算出inX距离dataSet数据集中所有数据点的距离
    sortedDistanceIndicies = distance.argsort()                     # 对距离进行排序，距离由小到大
    classCount = {}
    for i in range (k):                                             # 只保留距离最近的k个数据点
        voteIlabel = labels[sortedDistanceIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 计算k个数据点中，每个标签数据的个数
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]                                   # 标签个数最多的标签即为该inX的分类标签


def file2matrix(filename):
    """从文件中读取数据并转为矩阵

    Args:
        filename: 文件名

    Returns:
        数据矩阵
    """
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """数据标准化

    Args:
        dataSet: 需要标准化的数据

    Returns:
        标准化后的数据
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normalDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normalDataSet = dataSet - tile(minVals, (m, 1))
    normalDataSet = normalDataSet / tile(ranges, (m, 1))
    return normalDataSet, ranges, minVals

def img2Vector(filename):
    """从数字图形文件中的数据转成数据矩阵

    Args:
        filename: 数字图形文件名

    Returns:
        数据矩阵
    """
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        # print(str(lineStr))
        for j in range(32):
            returnVect[0, i*32+j] = int(lineStr[j])
    return returnVect

