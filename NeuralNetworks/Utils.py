# _*_ coding:utf-8 _*_
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
The utils.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/9/5 19:28
"""
from os import listdir
import gzip
import cPickle

from numpy import *
import ImprovedNN


def img2Matrix(filename):
    """从数字图形文件中的数据转成数据图形矩阵

    Args:
        filename: 数字图形文件名

    Returns:
        数据图形矩阵
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


def img2Flat(filename):
    """从数字图形文件中的数据转成一维数据矩阵

    Args:
        filename: 数字图形文件名

    Returns:
        一维数据矩阵
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        # print(str(lineStr))
        for j in range(32):
            returnVect[0, i * 32 + j] = int(lineStr[j])
    return mat(returnVect).reshape(1024, 1)


def loadMatrixData(dirName, imgMethod=img2Flat, isTrainingData=True):
    """加载训练数据或者测试数据，构成图形矩阵

    Args:
        dirName: 数据对应的目录名，trainingDigits或者testDigits
        imgMethod: 图形转换矩阵方法，img2Matrix转换成2维矩阵，img2Flat转换成1维矩阵
        isTrainingData: 是否是训练数据，训练数据的标签需要向量化

    Returns:
        inputList: 训练数据图形矩阵列表
        labelList: 训练数据对应的分类标签列表
`
    """
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    inputList = []
    labelList = []
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        labelList.append(ImprovedNN.vectorized_result(
            classNumStr) if isTrainingData else classNumStr)
        inputList.append(imgMethod("%s/%s" % (dirName, fileNameStr)))
    return zip(inputList, labelList)


def load_mnist_data():
    """加载mnist数据库

    Returns:
        training_data: 训练数据，输出数据进行了向量化，这样为了方便进行神经元的构建。
        validation_data: 检验数据
        test_data: 测试数据
    """
    f = gzip.open("mnistData/mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    training_inputs = [reshape(x, (784, 1)) for x in training_data[0]]
    training_outputs = [ImprovedNN.vectorized_result(y) for y in
                        training_data[1]]
    training_data = zip(training_inputs, training_outputs)
    validation_inputs = [reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = zip(validation_inputs, validation_data[1])
    test_inputs = [reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_inputs, test_data[1])
    return (training_data, validation_data, test_data)


if __name__ == "__main__":
    # trainInputList, trainLabelList = loadData("trainingDigits")
    # print("trainInputList:\n", str(trainInputList))
    # print("trainLabelList:\n", str(trainLabelList))

    # load_mnist_data测试
    training_data, validation_data, test_data = load_mnist_data()
    print(len(training_data))
