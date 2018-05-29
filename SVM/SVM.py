from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    """
    从文件中读取测试训练数据以及分类标签数据
    :param fileName: 数据文件名
    :return dataMat: 数据矩阵
    :return classLabels: 标签数组
    """
    dataArr = []; classLabels=[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataArr.append([float(lineArr[0]), float(lineArr[1])])
        classLabels.append(float(lineArr[2]))
    return dataArr, classLabels

def showDataSet(dataMat, classLabels):
    """
    将数据及标签进行可视化展示
    :param dataMat: 输入数据矩阵
    :param classLabels: 输入类型标签
    """
    data_plus = []  # 正类型数据样本
    data_minus = [] # 负类型数据样本

    # 分类
    for i in range(len(dataMat)):
        if classLabels[i] == 1:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])

    fit = plt.figure()
    ax = fit.add_subplot(111)
    # 转成numpy的array才能transpose
    data_plus_np = array(data_plus)
    data_mins_np = array(data_minus)
    ax.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], s=30, c='red')
    ax.scatter(transpose(data_mins_np)[0], transpose(data_mins_np)[1], s=30, c='green')
    plt.show()

def selectJRand(i, m):
    j = i
    while j==i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    """
    根据aj的范围(L~H)调整aj
    :param aj: 待调整的aj
    :param H: aj的上限
    :param L: aj的下限
    :return aj: 调整后的aj
    """
    if aj > H:
        return H
    if aj < L:
        return L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    简化版svm算法
    :param dataMatIn: 输入矩阵
    :param classLabels: 输入标签数组
    :param C: 松弛变量，用于控制“最大化间隔”和“保证大多数点的函数间隔小于1.0”这两个目标的权重
    :param toler: 容错率
    :param maxIter: 调整循环次数
    :return b: 调整完毕的b值
    :return alphas: 调整完毕的alphas值
    """
    dataMat = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    m,n = shape(dataMat)

    # 初始化alphas和b
    alphas = mat(zeros((m, 1))); b = 0

    # 循环调整alphas 和 b
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # 根据fx = wx + b计算fx的值，即计算分类值，其中w是行向量， x是列向量
            # 通过拉格朗日乘子法优化可以得出w的值为(1~n)∑i alphai*yi*xi
            fXi = float(multiply(alphas, labelMat).T *\
                        (dataMat * dataMat[i, :].T)) + b
            # 计算上式的分类值和期望分类值的差距Ei
            Ei  = fXi - float(labelMat[i])

            # 通过KKT条件推导，如下几个条件不满足，需要调整alpha
            # 1) yi * fxi > 1 且ai > 0                i点在边界内，ai应该为0
            # 2) yi * fxi < 1 且ai < c                i点在两条边界之间，ai应该为c
            # 3) yi * fxi == 1 且ai == 0 或 ai ==c    i点在边界线上，ai应该在0~c之间
            # 加上toler容错之后的条件如下???
            if ((labelMat[i] * Ei < -toler and alphas[i] < C ) or \
                    (labelMat[i] * Ei > toler and alphas[i] > 0)):
                j = selectJRand(i, m)   # 随机选择另一个点j
                print("\n")
                print("\n")
                print("########### alpha and b adjust begin for [%d %d] dot ###########" % (i, j))
                # 计算j点分类值
                fXj = float(multiply(alphas, labelMat).T * \
                            (dataMat * dataMat[j, :].T)) + b
                # 计算分类值和期望分类值的差距Ej
                Ej = fXj - float(labelMat[j])

                # 保存旧的alpha，=是引用幅值，所以需要copy一份新的
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                print("alphaIold:%f alphaJold:%f labelMat[i]:%d labelMat[j]:%d" % (alphaIold, alphaJold, labelMat[i], labelMat[j]))

                # 计算调整alpha的上限和下限
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphaJold - alphaIold)
                    H = min(C, C + alphaJold - alphaIold)
                else:
                    L = max(0, alphaJold + alphaIold - C)
                    H = min(C, alphaJold + alphaIold)
                print("L~H:[%f~%f]" % (L, H))

                if L == H:
                    print("L==H")
                    continue

                # 计算η
                # η = 2 * xi * xj.T - xi * xi.T - xj * xj.T
                eta = 2.0 * dataMat[i, :] * dataMat[j, :].T -\
                    dataMat[i, :] * dataMat[i, :].T - \
                    dataMat[j, :] * dataMat[j, :].T
                print("eta:%f" % eta)
                if eta >= 0:
                    print("eta>=0")
                    continue

                # 计算新的alphas[j]
                # alphaJnew = alphaJold - yj(Ei - Ej) / η
                alphas[j] = alphaJold - labelMat[j] * (Ei - Ej) / eta
                # 修建alphaJnew
                alphas[j] = clipAlpha(alphas[j], H, L)
                print("alphaJnew:%f alphaJold:%f" % (alphas[j], alphaJold))
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not move enough")
                    continue

                # 计算新的alphas[i]
                # alphaInew = alphaIold + yi*yj*(alphaJold - alphaJnewClipped)
                # 可以看出来alphaI的变化和alphaJ的变化量是相同的，但是方向可能相反
                alphas[i] = alphaIold + labelMat[i] * labelMat[j] * (alphaJold - alphas[j])

                print("after adjust [alphaInew:%f alphaJnew:%f]" % (alphas[i], alphas[j]))

                # 计算b1
                # b1New = bOld - Ei - yj * (alphaInew - alphaIold) * xi.T * xi - yj * (alphaJnew - alphaJold) * xj.T *xi
                b1 = b - Ei - labelMat[j] * (alphas[i] - alphaIold) * dataMat[i, :] * dataMat[i, :].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMat[i, :] * dataMat[j, :].T
                # b2New = bOld - Ej - yi * (alphaInew - alphaIold) * xi.T * xj - yj * (alphaJnew - alphaJold) * xj.T *xj
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i, :] * dataMat[j, :].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMat[j, :] * dataMat[j, :].T

                # 根据alphaInew和alphaJnew的范围求解新的b值
                if (0 < alphas[i] and alphas[i] < C):
                    b = b1
                elif (0 < alphas[j] and alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1.0
                print("after adjust [b:%f]" % b)
                #print("########### alpha and b adjust end for [%d %d] dot ###########" % (i, j))
        # 最后稳定的状态是为所有数据点循环了maxIter遍发现alphas没有进行过调整
        print("alphaPairsChanged:%d" % alphaPairsChanged)
        if (alphaPairsChanged == 0):
            print("alpha pairs not changed")
            iter += 1
        else:
            iter = 0
        print("iter:", str(iter))
    print("########### alpha and b adjust end ###########")
    print("b:", str(b))
    print("alphas:", str(alphas))
    return b, alphas

def showClassifer(dataArr, labelArr, b, alphas):
    """
    展示分类结果，画出最佳分割线，圈出支持向量
    :param dataArr: 数据点矩阵
    :param labelArr: 数据点对应的标签
    :param b: 通过smo调整出的b的最佳值
    :param alphas: 通过smo调整出的alphas最佳值
    """
    labelMat = mat(labelArr).T
    alphasMat = mat(alphas)
    dataMat = mat(dataArr)

    # 绘制所有不同分类的数据点
    data_plus = []  # 正类型数据样本
    data_minus = []  # 负类型数据样本
    for i in range(len(dataMat)):
        if classLabels[i] == 1:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])

    fit = plt.figure()
    ax = fit.add_subplot(111)
    data_plus_np = array(data_plus) # 转成numpy的array才能transpose
    data_mins_np = array(data_minus)
    ax.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], s=30, c='red')
    ax.scatter(transpose(data_mins_np)[0], transpose(data_mins_np)[1], s=30, c='green')


    # 绘制分类线
    w = multiply(alphasMat, labelMat).T * dataMat # 计算直线的法向量w
    # w*x + b = 0为分割直线方程
    # a1 * x + a2 * y + b = 0
    x = arange(0.0, 10.0, 0.1)
    wList = w.tolist()[0]
    a1 = wList[0]
    a2 = wList[1]
    print("a1:", str(a1), "a2:", str(a2))
    y = ((-b - a1 * x) / a2).tolist()[0]
    ax.plot(x, y)
    #print("x:", str(x))
    #print("y:", str(y))

    # 寻找边界线上的点，即支持向量点，并圈出来
    print("support vector:")
    for i in range(100):
        if alphas[i] > 0.0:
            print(str(dataArr[i]), " ", str(classLabels[i]))
            x = dataArr[i][0]
            y = dataArr[i][1]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')

    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    dataArr, classLabels = loadDataSet('testSet.txt')
    #showDataSet(dataArr, classLabels)
    b, alphas = smoSimple(dataArr, classLabels, 0.6, 0.001, 40)
    showClassifer(dataArr, classLabels, b, alphas)
