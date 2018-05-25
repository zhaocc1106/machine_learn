from numpy import *
import matplotlib.pyplot as plt

# 加载测试数据
def loadDataFile():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

# sigmoid 函数,类似于阶跃函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

# 梯度上升算法计算逻辑回归最佳系数
# arg1: 测试数据数组
# arg2: 测试数据分类标签数组
def gradAscent(dataMatIn, labelMatIn):
    # 转成矩阵
    dataMat = mat(dataMatIn)
    labelMat = mat(labelMatIn).transpose()

    # 利用梯度上升算法计算最佳回归系数
    m, n = shape(dataMat)
    print('m:', str(m), "n", str(n))
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))

    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = mat(labelMat - h)    # 错误差值
        #print("labelMat:", str(labelMat), "\nerror:", str(error))
        weights = weights + alpha * dataMat.transpose() * error  # 根据错误差值不断调整回归系数
        print("weights:", str(weights))

    return weights

# 生成logistic回归分类最佳拟合直线
def plotBestFit(weights):
    dataMat, labelMat = loadDataFile()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]

    # 分类所有数据点，生成坐标
    xcord1 = []; ycord1 = []    # 分类为1的所有数据点
    xcord2 = []; ycord2 = []    # 分类为0的所有数据点
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataMat[i][1]); ycord1.append(dataMat[i][2])
        else:
            xcord2.append(dataMat[i][1]); ycord2.append(dataMat[i][2])

    # 绘制所有不同分类的数据点
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # 绘制分类线
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

# 通过梯度上升算法计算逻辑回归系数的逻辑回归分类算法测试
def logsitcRegressionViaGradAscentTest():
    dataMat, labelMat = loadDataFile()
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights.getA())

# 随机梯度上升算法
def stocGradAscent0(dataMat, labelMat):
    dataArr = array(dataMat)
    m, n = shape(dataArr)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataArr[i] * weights))
        error = array(labelMat[i] - h)    # 错误差值
        print("error:", str(error), "dataMax[i]:", str(dataArr[i]), "weights:", str(weights))
        weights = weights + alpha * error * dataArr[i]  # 根据错误差值不断调整回归系数
    return weights

# 通过随机梯度上升算法计算逻辑回归系数的逻辑回归分类算法测试
def logsitcRegressionViaStocGradAscent0Test():
    dataMat, labelMat = loadDataFile()
    weights = stocGradAscent0(dataMat, labelMat)
    plotBestFit(weights)

# 改进的随机梯度算法
def stocGradAscent1(dataMat, labelMat, numberIter=150):
    dataArr = array(dataMat)
    m, n = shape(dataArr)
    weights = ones(n)
    for j in range(numberIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01    # 调整alpha
            randomIndex = int(random.uniform(0, len(dataIndex)))    #选取随机index
            h = sigmoid(sum(dataArr[randomIndex] * weights))
            error = array(labelMat[randomIndex] - h)
            weights = weights + alpha * error * dataArr[randomIndex]
            del(dataIndex[randomIndex])
    return weights

# 通过改进随机梯度上升算法计算逻辑回归系数的逻辑回归分类算法测试
def logsitcRegressionViaStocGradAscent1Test():
    dataMat, labelMat = loadDataFile()
    weights = stocGradAscent1(dataMat, labelMat, 500)
    plotBestFit(weights)

# 使用sigmoid函数和最佳回归系数对某数据进行分类
def classifyVector(inX, weights):
    ret = sigmoid(sum(inX * weights))
    if (ret > 0.5):
        return 1.0
    else:
        return 0.0

# 疝气死亡预测
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')

    # 创建训练数据集
    trainSet = []; trainLabels = []
    for line in frTrain.readlines():
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(curLine[i]))
        trainSet.append(lineArr)
        trainLabels.append(float(curLine[21]))

    # 使用训练数据集计算最佳回归系数
    weights = stocGradAscent1(trainSet, trainLabels)

    # 使用测试数据集计算预测错误概率
    errorCount = 0; numTestVect = 0.0
    for line in frTest.readlines():
        numTestVect += 1.0
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(curLine[i]))
        if int(classifyVector(array(lineArr), weights)) != int(curLine[21]):
            errorCount += 1
    errorRate = float(errorCount) / numTestVect
    print("error rate:", str(errorRate))
    return errorRate

# 多次进行疝气死亡预测并计算平均错误概率
def mutiTest():
    errorRate = 0.0
    for i in range(10):
        errorRate += colicTest()
    print("after test 10 times, the average error rate is ", str(errorRate / float(10)))

if __name__ == '__main__':
    #logsitcRegressionViaGradAscentTest()
    #logsitcRegressionViaStocGradAscent0Test()
    #logsitcRegressionViaStocGradAscent1Test()
    mutiTest()
