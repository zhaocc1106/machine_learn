#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""

The BP(Error back propagation) algorithm for multi-layer feed forward neural networks.
Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/8/28 8:49
"""
from numpy import *

def loadData():
    """返回一个训练集合以及对应的标签数组

    Returns:
        data: 一个训练集合
        labels: 训练集合对应的标签数组
    """
    data = array([[0, 0], [1, 0], [1, 1], [0, 2], [2, 2]])
    labels = array([[0], [1], [1], [1], [1]])
    return data, labels


def sigmoid(x):
    """sigmoid阶跃函数，用于神经元激活函数

    Args:
        x: 输入值

    Returns: sigmoid计算得到的值
    """
    return 1 / (1 + exp(-x))


def printParams(params):
    """输出参数集合

    Args:
        conWeights: 参数集合
    """
    m = len(params)
    for i in range(m):
        print("layer %d-layer %d" %(i, i+1))
        print(params[i])


def calcOutput(dataX, params, activFunc = sigmoid):
    """根据输入数据矩阵X，以及网络中的所有参数值，计算输出数据矩阵。

    Args:
        dataX: 输入矩阵X
        params: 连接权以及阈值的参数矩阵集合
        activFunc: 激活函数

    Returns:
        dataY: 输出矩阵Y
    """
    totalLayer = len(params) + 1                                        # 求总层数
    m, n = shape(dataX)

    layer_output = dataX                                                # 输入层的输出即为输入层的值
    for layer in range(totalLayer):
        print("###############layer %d begin###############" % (layer))

        if layer == 0:
            print("layer_output:\n", str(layer_output))
            print("###############layer %d end###############" % (layer))
            continue

        layerInput = layer_output                                       # 记录layer层的输入矩阵
        print("layer_input:\n", str(layerInput))

        mParams, nParams = shape(params[layer - 1])
        if layer == totalLayer - 1:
            layer_output = mat(-ones((m, mParams)))                     # 输出层不用添加-1哑结点
        else:
            layer_output = mat(-ones((m, mParams + 1)))                 # 生成layer层的输出矩阵，多了一列-1列作为哑结点

        """
        使用M-P神经元模型以及激活函数计算输出矩阵:
        即神经元的输出 y = f(1~n∑wi*xi - θ), 其中f为激活函数，wi为xi对应连接权，θ为该神经元的阈值。
        网络的第n层对应的参数是params[n-1]，第一层是没有参数的。
        """
        layer_output[:, :mParams] = sigmoid(layerInput *
                                            params[layer - 1].T)
        print("layer_output:\n", str(layer_output))

        print("###############layer %d end###############" % (layer))

    return layer_output


def standBP(dataX, dataY, numHidLayer = 1, hidLayerWid = [], activFunc = sigmoid):
    """标准BP算法

    通过训练数据集合X和Y，不断训练多层前馈神经网络的参数集合直至收敛稳定下来。

    Args:
        dataX: 输入数据矩阵X
        dataY: 输出数据矩阵Y
        numHidLayer: 隐藏层数
        hidLayerWid: 各个隐藏层的宽度，即神经元的个数
        activFunc: 激活函数

    Returns:
    """
    dataX = mat(dataX)
    dataY = mat(dataY)

    numTrainX, inputLayerWid = shape(dataX)                             # 获取输入层神经元的个数
    numTrainY, outputLayerWid = shape(dataY)                            # 获取输出层神经元的个数
    if numTrainX != numTrainY:
        print("Wrong numTrain.")
        return

    totalLayer = 2 + numHidLayer                                        # 网络总层数

    # 初始化网络中所有连接权以及阈值
    params = []

    for layer in range(totalLayer):
        if layer == 0:
            continue                                                    # 输入层没有任何参数
        if layer == totalLayer - 1:
            params.append(mat(random.rand(outputLayerWid,
                                          hidLayerWid[layer - 2] + 1))) # 为输出层与最后一个隐藏层生成随机的连接权值, +1是为了将阈值和连接权值组合在一起，最后一列是阈值
            break

        if layer == 1:
            params.append(mat(random.rand(hidLayerWid[layer - 1],
                                          inputLayerWid + 1)))          # 为第一层隐藏层与输入层之间生成随机的连接权值, +1是为了将阈值和连接权值组合在一起，最后一列是阈值
        else:
            params.append(mat(random.rand(hidLayerWid[layer - 1],
                                          hidLayerWid[layer - 2] + 1))) # 为非第一层隐藏层生成连接权值, +1是为了将阈值和连接权值组合在一起，最后一列是阈值


    print("conWeights and threshold:", str(params))
    # printParams(params)
    newDataX = mat(-ones((numTrainX, inputLayerWid + 1)))
    newDataX[:, :inputLayerWid] = dataX                                 # 为输入矩阵添加一个全部有-1组成的列对应阈值列，被称作哑结点
    print("newDataX:\n", str(newDataX))

    calcOutput(newDataX, params, activFunc)


if __name__ == '__main__':
    # dataX = random.rand(4, 5)
    # dataY = random.rand(4, 3)
    dataX, dataY = loadData()
    print("dataX:", str(dataX))
    print("dataY:", str(dataY))
    numHidLayer = 1                                                     # 一个隐藏层
    hidLayerWid = [2]                                                   # 隐层的神经元个数为2
    standBP(dataX, dataY, numHidLayer, hidLayerWid)
