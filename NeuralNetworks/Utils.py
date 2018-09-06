#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
The utils.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/9/5 19:28
"""
from numpy import *
from os import listdir

def img2Vector(filename):
    """从数字图形文件中的数据转成数据矩阵

    Args:
        filename: 数字图形文件名

    Returns:
        数据矩阵
    """
    fr = open(filename)
    returnVect = []
    for i in range(32):
        lineStr = fr.readline()
        # print(str(lineStr))
        lineVect = []
        for j in range(32):
            lineVect.append(int(lineStr[j]))
        returnVect.append(lineVect)
    return mat(returnVect)


def loadData(dirName):
    """加载训练数据或者测试数据

    Args:
        dirName: 数据对应的目录名，trainingDigits或者testDigits

    Returns:
        inputList: 训练数据输入矩阵列表
        labelList: 训练数据对应的分类标签列表

    """
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    inputList = []
    labelList = []
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        labelList.append([classNumStr])
        inputList.append(img2Vector("%s/%s" % (dirName, fileNameStr)))
    return array(inputList), array(labelList)


if __name__ == "__main__":
    trainInputList, trainLabelList = loadData("trainingDigits")
    print("trainInputList:\n", str(trainInputList))
    print("trainLabelList:\n", str(trainLabelList))