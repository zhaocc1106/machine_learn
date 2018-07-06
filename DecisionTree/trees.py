#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
"""
Decision Tree算法的实现

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/7/6 14:47
"""
from math import log
import operator

def calcShannonEnt(dataSet):
    """计算香农熵

    Args:
        dataSet: 需要计算香农熵的数据集

    Returns:
        香农熵
    """
    numEntries = len(dataSet)
    labelCounts = {}
    #calculate the sum for every kind begin.
    for featVect in dataSet:
        currentLabel = featVect[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    #calculate the sum for every kind end.

    for key in labelCounts:
        prop = float(labelCounts[key]) / numEntries #calculate the prop for every kind
        shannonEnt -= prop * log(prop, 2)           #calculate the shannonEnt
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']
               ]
    featLabels = ['no surfacing', 'flippers']
    return dataSet, featLabels


def splitDataSet(dataSet, feat, val):
    retDataSet = []
    for featVec in dataSet:
        if featVec[feat] == val:
            reducedFeatVec = featVec[:feat]
            reducedFeatVec.extend(featVec[feat+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# choose the best feature to split data set.
def chooseBestFeature(dataSet):
    """寻找最佳分割数据集方式

    根据信息增益计算找到分割当前数据集最佳分割方式，即选择哪个特征进行分割？

    Args:
        dataSet:

    Returns:

    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        featureSet = set(featureList)           #create unique feature list
        newEntropy = 0.0
        for featureValue in featureSet:
            subDataSet = splitDataSet(dataSet, i, featureValue)     #calculate shannonEnt for every subDataSet
            prop = len(subDataSet) / float(len(dataSet))
            #print("\nsubDataSet splited by feature %d value %d:\n" %(i, featureValue))
            #print(subDataSet)
            newEntropy += prop * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy     #calculate info gain.
        #print("\nthe info gain splited by feature %d is %f" %(i, infoGain))
        if (bestInfoGain < infoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def mayjorityCount(classList):
    """从类型标签列表中找到最多的类型当作当前类型标签

    Args:
        classList: 类型标签列表

    Returns:
        类型标签
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, featLabels):
    """创建决策树

    使用信息增益逐步递归构建决策树

    Args:
        dataSet: 特征数据集合
        featLabels: 类型标签列表

    Returns:
        决策树
    """
    labels = featLabels[:]
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(dataSet):       #stop split tree when all classes is the same.
        return classList[0]
    if len(dataSet[0]) == 1:
        return mayjorityCount(classList)                    #use the mayjority to represent the class.
    bestFeat = chooseBestFeature(dataSet)
    bestLabel = labels[bestFeat]
    myTrees = {bestLabel:{}}
    del(labels[bestFeat])
    featList = [example[bestFeat] for example in dataSet]
    featSet = set(featList)
    for value in featSet:
        subLabels = labels[:]
        myTrees[bestLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)    #create the child tree
    # print("\nthe sub trees is")
    print(myTrees)
    return myTrees


def classify(inputTree, featLabels, testVec):
    """对testVec数据进行分类

    使用训练数据集合训练出来的决策树inputTree对测试数据testVec进行分类

    Args:
        inputTree: 决策树
        featLabels: 类型标签列表
        testVec: 测试数据

    Returns:
        类型标签
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":                #this is dict, need continue classify.
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]                             #this is a leaf, the value is the classLabel
    return classLabel