from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    """
    从文件中读取测试训练数据以及分类标签数据
    :param fileName: 数据文件名
    :return dataMat: 数据矩阵
    :return classLabels: 标签数组
    """
    dataArr = [];
    labels = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        length = len(lineArr)
        tmpArr = []
        for i in range(length - 1):
            tmpArr.append(float(lineArr[i]))
        dataArr.append(tmpArr)
        labels.append(float(lineArr[-1]))
    return dataArr, labels

def standRgres(dataArr, labels):
    """
    普通最小二乘法计算最佳回归系数
    平方误差：m∑i (yi - Xi.T * w)^2      X为列向量
    对上公式求导等于0可以计算出w的公式如下
    w = (X.T * X) ^ (-1) * X.T * y      X为矩阵
    :param dataArr: 数据矩阵
    :param labels: 标签数组，即y的值
    :return w: 返回w列向量
    """
    dataMat = mat(dataArr); labelMat = mat(labels)
    X = dataMat
    y = labelMat.T
    xTx = X.T * X
    # 矩阵的求逆公式为X^-1 = X* / det(X)    X*为伴随矩阵（每个元素的代数余子式组成的矩阵）   det(X)为矩阵的秩
    if linalg.det(xTx) == 0.0:
        print("The matrix is singular, connot do inverse")
        return
    w = (xTx).I * (X.T * y)
    print("w:", str(w))
    return w

def showRegres(dataArr, labels, w):
    """
    展示二维数据以及它的最佳拟合直线
    :param dataArr: 数据矩阵
    :param labels: 标签数组
    :param w: 最佳回归系数
    :return:
    """
    fit = plt.figure()
    ax = fit.add_subplot(111)

    # 绘制所有的点
    ax.scatter((array(dataArr).T)[1], array(labels), s=10, c="red")

    # 绘制最佳拟合直线
    # Y = X * w
    X = arange(0.0, 1.0, 0.01)
    Xmat = mat(X)
    dataMat = mat(zeros((100, 2)))
    dataMat[:, 0] = 1.0
    dataMat[:, 1] = Xmat.T
    Y = transpose(dataMat * mat(w)).tolist()[0]
    print("X:", str(X))
    print("Y:", str(Y))
    ax.plot(X, Y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def calcCorrCoef(dataArr, labels, w):
    """
    计算相关系数
    :param dataArr: 数据矩阵
    :param labels: 标签数组
    :param w: 最佳回归系数
    :return cc: 相关系数
    """
    X = mat(dataArr)
    Y = X * w
    # cocccoef计算相关系数，必须是两个行向量
    cc = corrcoef(Y.T, labels)
    return cc[0, 1]

if __name__ == "__main__":
    dataArr, labels = loadDataSet("ex0.txt")
    print("dataArr:", str(dataArr))
    print("labels:", str(labels))
    w = standRgres(dataArr, labels)
    showRegres(dataArr, labels, w)
    cc = calcCorrCoef(dataArr, labels, w)
    print("cc:", str(cc))