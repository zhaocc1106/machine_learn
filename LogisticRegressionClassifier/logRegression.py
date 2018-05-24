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

if __name__ == '__main__':
    logsitcRegressionViaGradAscentTest()



