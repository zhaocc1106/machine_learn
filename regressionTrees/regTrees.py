from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    """
    从文件中读取测试训练数据以及分类标签数据
    :param fileName: 数据文件名
    :return dataArr: 数据矩阵
    """
    dataArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataArr.append(list(map(float, lineArr)))
    return dataArr

# ************************************************BEGIN: 回归树相关************************************************
def regLeaf(dataSet):
    """
    返回回归树的叶子节点的常数值
    :param dataSet: 划分给叶子节点的数据集
    :return: 叶子节点的常数值
    """
    return mean(dataSet[:, -1])                     # 使用平均值代表叶子节点的值

def regErr(dataSet):
    """
    回归树的误差计算公式
    方差是平方误差的均值（均方差），而这里需要的是平方误差的总值（总方差）。
    总方差可以通过均方差乘以数据集中样本点的个数来得到。
    :param dataSet: 需要计算误差的数据集合
    :return: 计算出的误差
    """
    return var(mat(dataSet)[:, -1]) * shape(mat(dataSet))[0]

def regTreeEval(model, inDat):
    """
    返回回归树叶子节点的值作为预测的值
    :param model: 叶子节点模型，这里是常量值
    :param inDat: 输入数据，这里因为使用回归树，回归树的叶子节点是常量值，所以该值没有用到
    :return: 预测值
    """
    # print("regTreeEval model", str(model))
    return float(model)
# ************************************************END: 回归树相关************************************************

# ************************************************BEGIN: 模型树相关************************************************
def linearSolve(dataSet):
    """
    使用线性回归分析数据，并返回最佳回归系数
    :param dataSet: 数据矩阵
    :return ws: 最佳线性回归系数
    """
    m, n = shape(dataSet)
    X = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    # 矩阵的求逆公式为X^-1 = X* / det(X)    X*为伴随矩阵（每个元素的代数余子式组成的矩阵）   det(X)为矩阵的秩
    if linalg.det(xTx) == 0.0:
        print("The matrix is singular, connot do inverse")
        return
    ws = (xTx).I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataSet):
    """
    返回模型树叶子节点的线性模型的回归系数ws
    :param dataSet:
    :return ws:最佳线性回归系数
    """
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    """
    计算线性回归的误差，使用估计值与真实值的平方差的总值来计量
    :param dataSet: 数据矩阵
    :return: 计算出的误差
    """
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(yHat - Y, 2))

def modelTreeEval(model, inDat):
    """
    返回使用模型树叶子节点的线性模型回归系数和输入值inDat计算出预测值
    :param model: 叶子节点模型，这里是线性模型的回归系数
    :param inDat: 输入数据，因为这里使用模型树，所以需要通过线性模型回归系数和输入值inDat计算出预测值
    :return: 预测值
    """
    # print("modelTreeEval ", str(float(mat(inDat) * model)))
    return float(mat(inDat) * model)

def plotModel(modelTree, min, max, ax):
    """
    将模型树的回归线画出来
    :param modelTree: 模型树
    :param min: x轴的最小值
    :param max: x轴的最大值
    :param ax: 画布
    :return:
    """
    if isTree(modelTree):                                           # 如果当前节点是树，则递归画出其子树回归线
        plotModel(modelTree['lTree'], min, modelTree['spVal'], ax)
        plotModel(modelTree['rTree'], modelTree['spVal'], max, ax)
    else:
        X = arange(min, max, 0.01)
        # print("min:", min, " max:", max)
        Xmat = mat(X)
        dataMat = mat(zeros((shape(Xmat)[1], 2)))
        dataMat[:, 0] = 1.0
        dataMat[:, 1] = Xmat.T
        # print("modelTree:", str(modelTree))
        Y = transpose(dataMat * mat(modelTree)).tolist()[0]
        # print("X:", str(X))
        # print("Y:", str(Y))
        ax.plot(X, Y)

def plotDataSetWithModel(dataSet, modelTree, x=0, y=1):
    """
    绘制模型树的测试数据，及模型树的线性回归线
    :param dataSet: 测试数据集
    :param modelTree: 生成的模型树
    :return:
    """
    fit = plt.figure()
    ax = fit.add_subplot(111)
    ax.scatter(array(dataSet).T[x], array(dataSet).T[y], s=10, c='b', alpha = .5)
    plotModel(modelTree, amin(array(dataSet)[:, x], 0), amax(array(dataSet)[:, x], 0), ax)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
# ************************************************END: 模型树相关************************************************

def chooseBestSplit(dataSet, leafType, errType, ops):
    """
    根据叶子类型，和误差计算公式找到最佳分割数据集的方式，返回分割数据的feature index和feature value
    :param dataSet: 数据集合
    :param leafType: 叶子类型
    :param errType: 误差计算公式
    :param ops: 包含构建树的其他参数
    :return feat: 分割数据的feature index
    :return value: 分割数据的feature value，如果满足停止划分条件，则该值为叶子节点的的值
    """
    dataSet = mat(dataSet)                              # 矩阵化
    tolS = ops[0]; tolN = ops[1]                        # tolS表示容忍的误差大小，tolN表示分割后的数据集最少的个数
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:     # 如果数据集合的分类值均相同，则代表这组数据属于同一个叶子节点
        return None, leafType(dataSet)

    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf                                         # 初始误差无限大
    bestIndex = 0; bestVal = 0

    for featIndex in range(n - 1):
        for featVal in set(dataSet[:, featIndex].T.tolist()[0]):
            # print("featIndex:", featIndex, " featVal:", featVal)
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, featVal)
            # print("mat0:", str(mat0))
            # print("mat1:", str(mat1))
            if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
                continue                                # 如果划分出的数据集合过小，则不使用
            newS = errType(mat0) + errType(mat1)
            # print("newS:", newS)
            # print("mat0:", str(mat0))
            # print("mat1:", str(mat1))
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestVal = featVal

    if (S - bestS) < tolS:
        return None, leafType(dataSet)                   # 如果划分后的误差降低的范围小于能够容忍的误差大小，则代表这组数据属于同一个叶子节点
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestVal)
    if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
        return None, leafType(dataSet)                   # 如果划分出的数据集合过小，则代表这组数据属于同一个叶子节点
    return bestIndex, bestVal

def binSplitDataSet(dataSet, feature, value):
    """
    对feature特征，以value为分割值，分割dataSet数据集
    :param dataSet: 待分割数据集
    :param feature: 特征index
    :param value: 分割值
    :return mat0: 分割出的矩阵0
    :return mat1: 分割出的矩阵1
    """
    mat0 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    return mat0, mat1

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=[1, 4]):
    """
    构建回归树(regression tree)或者模型树(model tree)
    回归树：叶子节点是常数值的树
    模型树：叶子节点是一个线性模型的树
    具体构建哪种树由leafType决定，默认构建回归树
    :param dataSet: 数据集
    :param leafType: 叶子类型，决定构建哪种树
    :param errType: 代表误差计算函数
    :param ops: 包含构建树的其他参数
    :return retTree: 构建成功的树
    """
    dataSet = mat(dataSet)
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val                                              # 如果满足停止条件时，返回叶子节点的值

    retTree = {}                                                # 用一个集合代表一棵树
    retTree['spInd'] = feat                                     # spInd元素代表分割数据的特征index
    retTree['spVal'] = val                                      # spVal元素代表分割数据的特征值
    lSet, rSet = binSplitDataSet(dataSet, feat, val)            # 使用feat index和value分割左右子树的数据集合
    retTree['lTree'] = createTree(lSet, leafType, errType, ops) # 使用左子树的数据集递归构建左子树
    retTree['rTree'] = createTree(rSet, leafType, errType, ops) # 使用右子树的数据集递归构建右子树
    return retTree

def isTree(obj):
    """
    判断节点是否是树
    :param obj: 节点
    :return:
    """
    return (type(obj).__name__ == 'dict')                       # 如果节点是dictionary，我们认为它是树

def getMean(tree):
    """
    对节点的树进行减枝，只留下根节点
    :param tree: 需要减枝的树
    :return: 返回减枝后树的值
    """
    if not isTree(tree):
        return tree                                             # 如果不是树，直接返回值
    if isTree(tree['lTree']):
        tree['lTree'] = getMean(tree['lTree'])                  # 递归求子树的减枝后的值
    if isTree(tree['rTree']):
        tree['rTree'] = getMean(tree['rTree'])
    return (tree['lTree'] + tree['rTree']) / 2                  # 左右子树的均值当作该树的值

def prune(tree, testData):
    """
    对树进行后减枝，由于当ops选择不合理时，即预减枝的效果不佳时，会导致过拟合。
    降低过拟合的方法就是使用测试数据对数进行后减枝。
    :param tree: 要减枝的树
    :param testData: 测试数据矩阵
    :return retTree: 减枝后的树
    """
    if shape(testData)[0] == 0:                                 # 如果该树没有划分到任何数据集，则认为该树是过拟合了，将子树全部剪掉
        return getMean(tree)

    if isTree(tree['lTree']) or isTree(tree['rTree']):          # 如果子树至少有一个是树，则需要分割测试数据给子树用于减枝
        lData, rData = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # print("tree[spInd]:", str(tree['spInd']))
        # print("tree['spVal']:", str(tree['spVal']))
        # print("lData:", str(lData))
        # print("rData:", str(rData))

    if isTree(tree['lTree']):
        tree['lTree'] = prune(tree['lTree'], lData)             # 如果子树是树，则递归对子树进行减枝操作
    if isTree(tree['rTree']):
        tree['rTree'] = prune(tree['rTree'], rData)

    if not isTree(tree['lTree']) and not isTree(tree['rTree']):
        lData, rData = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errNoMerge = sum(power(lData[:, -1] - tree['lTree'], 2)) + \
            sum(power(rData[:, -1] - tree['rTree'], 2))         # 计算子树没有合并之前各自误差的总方差

        treeMean = (tree['lTree'] + tree['rTree']) / 2.0
        errMerge = sum(power(testData[:, -1] - treeMean, 2))    # 计算子树合并之后误差的总方差
        # print("errNoMerge:", errNoMerge, " errMerge:", errMerge)

        if  errMerge < errNoMerge:                               # 如果合并后效果更好，则合并，即掉左右枝
            print("merge... \nlTree:", str(tree['lTree']), "\nrTree:", str(tree['rTree']))
            return getMean(tree)
        else:
            return tree                                         # 合并后误差更大，则不减枝
    else:
        return tree

def plotDataSet(dataSet, x=0, y=1):
    """
    将数据集进行绘制
    :param dataSet: 数据矩阵
    :return:
    """
    fit = plt.figure()
    ax = fit.add_subplot(111)
    ax.scatter(array(dataSet).T[x], array(dataSet).T[y], s=10, c='b', alpha = .5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def treeForecast(tree, inData, modelEval=regTreeEval):
    """
    对于输入值，通过树回归预测值
    :param tree: 树
    :param inData: 输入数据
    :param modelEval: 树的不同类型，对应不同类型叶子节点，对应不同预测值的计算方式
    :return: 预测出的值
    """
    # print("inData:", str(inData))
    if not isTree(tree):
        return modelEval(tree, inData)                          # 如果递归到了叶子节点，直接计算出预测值

    # print("tree['spInd']:", str(tree['spInd']))
    # print("tree['spVal']:", str(tree['spVal']))
    if inData[tree['spInd']] <= tree['spVal']:                  # 走左子树
        return treeForecast(tree['lTree'], inData, modelEval)
    else:
        return treeForecast(tree['rTree'], inData, modelEval)          # 走右子树

def createForecast(tree, testData, modelEval=regTreeEval):
    """
    对于一批测试数据，通过树回归计算每个测试数据的预测值
    :param tree: 树
    :param testData: 测试数据矩阵
    :param modelEval: 树的不同类型，对应不同类型叶子节点，对应不同预测值的计算方式
    :return: 
    """
    m, n = shape(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i] = treeForecast(tree, array(testData)[i, :], modelEval)
        # print("yHat[i]:", str(yHat[i]))
    return yHat

if __name__ == '__main__':
    # 测试ex00.txt数据
    """
    dataSet = loadDataSet("ex00.txt")
    print("dataSet:", str(dataSet))
    m, n = shape(mat(dataSet))
    dataMat = mat(ones((m, n + 1)))
    dataMat[:, 1:3] = mat(dataSet)
    print("dataMat:", str(dataMat))
    plotDataSet(dataMat, 1, 2)
    regTree = createTree(dataMat, ops=[1, 4])
    print("regTree:", str(regTree))
    """

    # 测试ex0.txt数据
    """
    dataSet = loadDataSet("ex0.txt")
    print("dataSet:", str(dataSet))
    m, n = shape(mat(dataSet))
    print("dataMat:", str(dataSet))
    plotDataSet(dataSet, 1, 2)
    regTree = createTree(dataSet, ops=[1, 4])
    print("regTree:", str(regTree))
    """

    # 测试ex2.txt数据
    """
    dataSet = loadDataSet("ex2.txt")
    print("dataSet:", str(dataSet))
    m, n = shape(mat(dataSet))
    dataMat = mat(ones((m, n + 1)))
    dataMat[:, 1:3] = mat(dataSet)
    print("dataMat:", str(dataMat))
    plotDataSet(dataMat, 1, 2)
    # betterTree = createTree(dataMat, ops=[10000, 4])
    # print("Better regTree:", str(betterTree))
    """

    # 生成一个过拟合的树，测试减枝效果
    """
    regTree = createTree(dataMat, ops=[1, 4])
    print("Overfitting regTree:", str(regTree))

    # 对过拟合的树进行减枝
    dataSet = loadDataSet("ex2test.txt")
    m, n = shape(mat(dataSet))
    dataMat = mat(ones((m, n + 1)))
    dataMat[:, 1:3] = mat(dataSet)
    plotDataSet(dataMat, 1, 2)
    prunedTree = prune(regTree, dataMat)
    print("Pruned tree:", str(prunedTree))
    """

    # 对exp2数据生成模型树
    """
    dataSet = loadDataSet("exp2.txt")
    m, n = shape(mat(dataSet))
    dataMat = mat(ones((m, n + 1)))
    dataMat[:, 1:3] = mat(dataSet)
    modelTree = createTree(dataMat, leafType=modelLeaf, errType=modelErr, ops=[1, 10])
    print("modelTree:", str(modelTree))
    plotDataSetWithModel(dataMat, modelTree, 1, 2)                  # 绘制模型树的数据点以及模型树的线性模型回归线
    """

    # 对bikeSpeedVsIq_train数据分别生成回归树模型，模型树模型，以及线性回归模型，并通过bikeSpeedVsIq_test测试数据比较不同模型的效果
    dataSet = loadDataSet("bikeSpeedVsIq_train.txt")
    m, n = shape(mat(dataSet))
    dataMat = mat(ones((m, n + 1)))
    dataMat[:, 1:3] = mat(dataSet)
    regTree = createTree(dataMat, leafType=regLeaf, errType=regErr, ops=[1, 20])        # 生成回归树
    print("regTree:", str(regTree))
    modelTree = createTree(dataMat, leafType=modelLeaf, errType=modelErr, ops=[1, 20])  # 生成模型树
    print("modelTree:", str(modelTree))
    ws = linearSolve(dataMat)[0]                                                        # 生成线性模型
    print("ws:", str(ws))

    dataSet = loadDataSet("bikeSpeedVsIq_test.txt")
    m, n = shape(mat(dataSet))
    dataMat = mat(ones((m, n)))
    dataMat[:, 1] = mat(dataSet)[:, 0]

    yHat0 = createForecast(regTree, dataMat, regTreeEval)                               # 根据生成的回归树计算预测值
    yHat1 = createForecast(modelTree, dataMat, modelTreeEval)                           # 根据生成的模型树计算预测值
    yHat2 = dataMat * ws                                                                # 根据生成的线性模型计算预测值

    print("yHat0:", str(yHat0.T), "\nyHat1:", str(yHat1.T), "\nyHat2", str(yHat2.T))

    # 计算每种模型计算出的预测值和真实值的相关系数的大小，R²越接近1，代表效果越好
    cof0 = corrcoef(yHat0, array(dataSet)[:, 1], rowvar=0)[0, 1]
    cof1 = corrcoef(yHat1, array(dataSet)[:, 1], rowvar=0)[0, 1]
    cof2 = corrcoef(yHat2, array(dataSet)[:, 1], rowvar=0)[0, 1]

    print("cof0:", str(cof0), "\ncof1:", str(cof1), "\ncof2:", str(cof2))

    # 能够测出来模型树的效果要更好，绘出模型树相关的回归线
    plotDataSetWithModel(dataSet, modelTree, 0, 1)

    