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
import ImprovedBP


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


def loadMatrixData(dirName):
    """加载训练数据或者测试数据，构成图形矩阵

    Args:
        dirName: 数据对应的目录名，trainingDigits或者testDigits

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
        labelList.append([classNumStr])
        inputList.append(img2Matrix("%s/%s" % (dirName, fileNameStr)))
    return array(inputList), array(labelList)


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
    training_outputs = [ImprovedBP.vectorized_result(y) for y in
                        training_data[1]]
    training_data = zip(training_inputs, training_outputs)
    validation_inputs = [reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = zip(validation_inputs, validation_data[1])
    test_inputs = [reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_inputs, test_data[1])
    return (training_data, validation_data, test_data)


def dropout(x, level):
    """dropout的实现

    Args:
        x: 输入数据
        level: dropout的概率值，比如0.4，数据有10个，则表示4个数据被dropout

    Returns:
        dropout过的数据值
    """
    if level < 0. or level >= 1:  # level是概率值，必须在0~1之间
        raise ValueError('Dropout level must be in interval [0, 1[.')
    retain_prob = 1. - level

    # 我们通过binomial函数，生成与x一样的维数向量。binomial函数就像抛硬币一样，我们可以把每个神经元当做抛硬币一样
    # 硬币 正面的概率为p，n表示每个神经元试验的次数
    # 因为我们每个神经元只需要抛一次就可以了所以n=1，size参数是我们有多少个硬币。
    random_tensor = random.binomial(n=1, p=retain_prob, size=x.shape)  #
    # 即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
    print(random_tensor)

    x *= random_tensor
    print(x)
    x /= retain_prob
    print(x)
    return x


if __name__ == "__main__":
    # trainInputList, trainLabelList = loadData("trainingDigits")
    # print("trainInputList:\n", str(trainInputList))
    # print("trainLabelList:\n", str(trainLabelList))

    # 对dropout的测试
    # x = asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float32)
    # dropout(x, 0.4)

    # load_mnist_data测试
    training_data, validation_data, test_data = load_mnist_data()
    print(len(training_data))