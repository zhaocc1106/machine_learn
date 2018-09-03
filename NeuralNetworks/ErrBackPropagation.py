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


def loadXORData():
    """返回一个异或问题训练集合以及对应的标签数组

    Returns:
        data: 一个异或问题训练集合
        labels: 异或问题训练集合对应的标签数组
    """
    data = array([[0, 0], [1, 0], [0, 1], [1, 1]])
    labels = array([[0], [1], [1], [0]])
    return data, labels


def sigmoid(x):
    """sigmoid阶跃函数，用于神经元激活函数

    Args:
        x: 输入值

    Returns: sigmoid计算得到的值
    """
    return 1.0 / (1 + exp(-x))


def printLayerData(data):
    """展示每一层的相关数据

    Args:
        data: 数据
    """
    m = len(data)
    for i in range(m):
        print("layer %d-layer %d" % (i, i + 1))
        print(data[i])


def calcOutput(dataX, params, activFunc=sigmoid):
    """根据输入数据矩阵X，以及神经网络中的所有参数值，计算输出数据矩阵。

    Args:
        dataX: 输入矩阵X
        params: 连接权以及阈值的参数矩阵集合
        activFunc: 激活函数

    Returns:
        dataY: 输出矩阵Y
    """
    # print("#############################################calcOutput begin#############################################")
    totalLayer = len(params) + 1  # 求总层数
    m, n = shape(dataX)

    output = []  # 用于记录每一层的输出
    layer_output = dataX  # 输入层的输出即为输入层的值
    output.append(layer_output)
    for layer in range(totalLayer):
        # print("###############layer %d begin###############" % (layer))

        if layer == 0:
            # print("layer_output:\n", str(layer_output))
            # print("###############layer %d end###############" % (layer))
            continue

        layerInput = layer_output  # 记录layer层的输入矩阵
        # print("layer_input:\n", str(layerInput))

        mParams, nParams = shape(params[layer - 1])
        """
        使用M-P神经元模型以及激活函数计算输出矩阵:
        即神经元的输出 y = f(1~n∑wi*xi - θ), 其中f为激活函数，wi为xi对应连接权，θ为该神经元的阈值。
        神经网络的第n层对应的参数是params[n-1]，第一层是没有参数的。
        """
        layer_output = activFunc(layerInput *
                                 params[layer - 1].T)
        if layer == totalLayer - 1:
            output.append(layer_output)
            # print("layer_output:\n", str(layer_output))
            # print("###############layer %d end###############" % (layer))
            break
        else:
            layer_output_new = mat(-ones((m, mParams + 1)))  # 生成layer层的输出矩阵，多了一列-1列作为哑结点
            layer_output_new[:, :-1] = layer_output
            layer_output = layer_output_new
            # print("layer_output:\n", str(layer_output))
            output.append(layer_output)
            # print("###############layer %d end###############" % (layer))

    # print("#############################################calcOutput end#############################################")
    return output


def calcGradient(realOutPut, estOutput, params):
    """计算神经网络中所有参数的梯度

    Args:
        realOutPut: 一个测试样例的真实输出矩阵，例如Yk
        estOutput: 计算得到的神经网络中每层输出矩阵
        numHidLayer: 隐层层数

    Returns:
        神经网络所有参数的梯度
    """
    # print("######################calcGradient begin######################")
    deltaParams = []  # 存放神经网络参数的梯度
    gs = []  # 存储神经网络每层g值，g值用于计算梯度
    gLayer = 0.0  # 存储上一层的g值，用于计算本层的g值
    totalLayer = len(params) + 1  # 神经网络总层数
    for layer in reversed(range(totalLayer)):
        # print("###############layer %d begin###############" % (layer))
        if layer == totalLayer - 1:
            """
            g值用于计算参数的梯度。
            注意每个神经元都有g值。
            输出层第j个神经元的g值计算公式如下:
            gJ = estOutputJ * (1- estOutputJ) * (realOutputJ - estOutputJ)
            其中estOutputJ为计算出的输出值，realOutputJ为真实的输出值。
            """
            # print("realOutput:\n", str(realOutPut))
            # print("estOutput[layer]:\n", str(estOutput[layer]))
            gLayer = multiply(estOutput[layer],
                              multiply((1 - estOutput[layer]),
                                       (realOutPut - estOutput[layer])))  # 计算输出层对应的g值
            # print("gLayer:\n", str(gLayer))
            gs.insert(0, gLayer)

            # print("estOutput[layer - 1]\n", str(estOutput[layer - 1]))
            deltaParamLayer = multiply(gLayer.T, estOutput[layer - 1])  # 计算该层所有神经元所有参数的梯度
            # print("deltaParamLayer:\n", str(deltaParamLayer))
            deltaParams.insert(0, deltaParamLayer)  # 存储该层神经元的参数的梯度
            # print("###############layer %d end###############" % (layer))
            continue
        if layer == 0:
            # print("###############layer %d end###############" % (layer))
            break  # 输入层没有g值
        """
        g值用于计算参数的梯度。
        非输出层第h个神经元的g值计算公式如下:
        gH = bH * (1-bH) * (1~l∑ wHJ * g'J)
        其中bH为本层第h个神经元输入值，wHJ为上层神经元J与本层神经元H连接权值，g'J为上层神经元J的g值
        """
        # print("params[layer][:, :-1]:\n", str(params[layer][:, :-1]))
        # print("previous gLayer:\n", str(gLayer))
        # print("estOutput[layer][:, :-1]\n", str(estOutput[layer][:, :-1]))
        temp = gLayer * params[layer][:, :-1]  # 参数最后一列是阈值列，不用于计算g值
        gLayer = multiply(estOutput[layer][:, :-1],
                          multiply(1 - estOutput[layer][:, :-1], temp))  # 计算该层的g值
        # print("gLayer:\n", str(gLayer))
        gs.append(gLayer)
        gs.insert(0, gLayer)  # 因为是倒序求的gLayer，所以在添加时添加到数组首

        # print("estOutput[layer - 1]\n", str(estOutput[layer - 1]))
        deltaParamLayer = multiply(gLayer.T, estOutput[layer - 1])  # 计算该层所有神经元所有参数的梯度
        # print("deltaParamLayer:\n", str(deltaParamLayer))
        deltaParams.insert(0, deltaParamLayer)  # 存储该层神经元的参数的梯度
        # print("###############layer %d end###############" % (layer))

    # print("gs:\n", str(gs))
    # print("deltaParams:\n", str(deltaParams))
    # print("######################calcGradient end######################")
    return deltaParams


def initParams(totalLayer, inputLayerWid, outputLayerWid, hidLayerWid):
    """初始化参数集合

    某层参数矩阵形式应该如下：
                下层    第一个神经元    第二个神经元    ...    第m个神经元    哑结点（固定输入值-1）
          上层
    第一个神经元          W11              W12         ...       W1m          阈值θ1
    第二个神经元          W21              W22         ...       W2m          阈值θ2
    ...
    第n个神经元           Wn1              Wn2         ...       Wnm          阈值θn
    Ps: Wnm代表上层第n个神经元连接到下层第m个神经元的连接权

    Args:
        totalLayer: 神经网络总层数
        inputLayerWid: 输入层神经网络个数
        outputLayerWid: 输出层神经网络个数
        hidLayerWid: 每个隐藏层神经网络个数

    Returns:
        所有参数的集合，每层有一个参数矩阵
    """
    params = []

    for layer in range(totalLayer):
        if layer == 0:
            continue  # 输入层没有任何参数
        if layer == totalLayer - 1:
            params.append(mat(random.rand(outputLayerWid,
                                          hidLayerWid[
                                              layer - 2] + 1)))  # 为输出层与最后一个隐藏层生成随机的连接权值, +1是为了将阈值和连接权值组合在一起，最后一列是阈值
            break

        if layer == 1:
            params.append(mat(random.rand(hidLayerWid[layer - 1],
                                          inputLayerWid + 1)))  # 为第一层隐藏层与输入层之间生成随机的连接权值, +1是为了将阈值和连接权值组合在一起，最后一列是阈值
        else:
            params.append(mat(random.rand(hidLayerWid[layer - 1],
                                          hidLayerWid[layer - 2] + 1)))  # 为非第一层隐藏层生成连接权值, +1是为了将阈值和连接权值组合在一起，最后一列是阈值

    print("conWeights and threshold:", str(params))
    return params


def standBP(dataX, dataY, numHidLayer=1, hidLayerWid=[], activFunc=sigmoid, alpha = 2, itera = 10000):
    """标准BP算法

    通过训练数据集合X和Y，不断训练多层前馈神经网络的参数集合直至收敛稳定下来。

    Args:
        dataX: 输入数据矩阵X
        dataY: 输出数据矩阵Y
        numHidLayer: 隐藏层数
        hidLayerWid: 各个隐藏层的宽度，即神经元的个数
        activFunc: 激活函数
        alpha: 学习速率
        itera: 学习次数

    Returns:
    """
    dataX = mat(dataX)
    dataY = mat(dataY)

    numTrainX, inputLayerWid = shape(dataX)  # 获取输入层神经元的个数
    numTrainY, outputLayerWid = shape(dataY)  # 获取输出层神经元的个数
    if numTrainX != numTrainY:
        print("Wrong numTrain.")
        return

    totalLayer = 2 + numHidLayer  # 神经网络总层数
    params = initParams(totalLayer, inputLayerWid,
                        outputLayerWid, hidLayerWid)  # 初始化神经网络的参数集合

    newDataX = mat(-ones((numTrainX, inputLayerWid + 1)))
    newDataX[:, :inputLayerWid] = dataX  # 为输入矩阵添加一个全部有-1组成的列对应阈值列，被称作哑结点
    print("newDataX:\n", str(newDataX))

    output = calcOutput(newDataX[0, :], params, activFunc)  # 根据当前参数集合计算出来的每层神经元的输出
    print("output:\n", str(output))
    printLayerData(output)

    # deltaParams = calcGradient(dataY[0, :], output, params)  # 根据当前神经网络的输出，计算当前参数调整的梯度
    # print("deltaParams:\n", str(deltaParams))
    #
    # for k in range(len(deltaParams)):
    #     params[k] = params[k] + 0.01 * deltaParams[k]  # 根据梯度计算新的参数集合
    # print("new params:\n", str(params))

    alpha = alpha
    for i in range(itera):
        if i % 1000 == 0:
            print("*******************************************%d standBP*******************************************" % i)
        for j in range(numTrainX):  # 标准BP算法每次更新只针对单个样例，参数更新非常频繁
            # print("the training example:\n",
            #       str(newDataX[j, :-1]), str(dataY[j, :]))
            output = calcOutput(newDataX[j, :], params, activFunc)  # 根据当前参数集合计算出来的每层神经元的输出

            deltaParams = calcGradient(dataY[j, :], output, params)  # 根据当前神经网络的输出，计算当前参数调整的梯度

            if i % 1000 == 0:
                print("output:\n", str(output))
                # print("params:\n", str(params))
                # print("deltaParams:\n", str(deltaParams))

            for k in range(len(deltaParams)):
                params[k] = params[k] + alpha * deltaParams[k]  # 根据梯度计算新的参数集合
            # print("new params:\n", str(params))

    output = calcOutput(newDataX, params, activFunc)  # 根据最终调整好的参数计算训练集的输出
    print("final output:\n", str(output[-1]))


if __name__ == '__main__':
    # dataX = random.rand(4, 5)
    # dataY = random.rand(4, 3)
    """
    训练一个线性可分割分类问题
    """
    """
    dataX, dataY = loadData()
    print("dataX:", str(dataX))
    print("dataY:", str(dataY))
    numHidLayer = 1  # 一个隐藏层
    hidLayerWid = [2]  # 隐层的神经元个数为2
    standBP(dataX, dataY, numHidLayer, hidLayerWid)
    """

    """
    训练典型非线性分类问题，异或问题
    """
    dataX, dataY = loadXORData()
    print("dataX:", str(dataX))
    print("dataY:", str(dataY))
    numHidLayer = 2  # 两个隐藏层
    hidLayerWid = [2, 3]  # 隐层的神经元个数分别为2和3
    standBP(dataX, dataY, numHidLayer, hidLayerWid, sigmoid, 3, 20000)