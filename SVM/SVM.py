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
    classLabels = []
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
    data_minus = []  # 负类型数据样本

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
    ax.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], s=30,
               c='red')
    ax.scatter(transpose(data_mins_np)[0], transpose(data_mins_np)[1], s=30,
               c='green')
    plt.show()


def selectJRand(i, m):
    j = i
    while j == i:
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


def showClassifer(dataArr, labelArr, b, alphas, C):
    """
    展示分类结果，画出最佳分割线，圈出支持向量
    :param dataArr: 数据点矩阵
    :param labelArr: 数据点对应的标签
    :param b: 通过smo调整出的b的最佳值
    :param alphas: 通过smo调整出的alphas最佳值
    :param C: 松弛变量
    """
    labelMat = mat(labelArr).T
    alphasMat = mat(alphas)
    dataMat = mat(dataArr)

    # 绘制所有不同分类的数据点
    data_plus = []  # 正类型数据样本
    data_minus = []  # 负类型数据样本
    for i in range(len(dataMat)):
        if labelArr[i] == 1:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])

    fit = plt.figure()
    ax = fit.add_subplot(111)
    data_plus_np = array(data_plus)  # 转成numpy的array才能transpose
    data_mins_np = array(data_minus)
    ax.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], s=30,
               c='red')
    ax.scatter(transpose(data_mins_np)[0], transpose(data_mins_np)[1], s=30,
               c='green')

    # 绘制分类线
    w = multiply(alphasMat, labelMat).T * dataMat  # 计算直线的法向量w
    # w*x + b = 0为分割直线方程
    # a1 * x + a2 * y + b = 0
    x = arange(0.0, 10.0, 0.1)
    wList = w.tolist()[0]
    a1 = wList[0]
    a2 = wList[1]
    print("a1:", str(a1), "a2:", str(a2))
    y = ((-b - a1 * x) / a2).tolist()[0]
    ax.plot(x, y)
    # print("x:", str(x))
    # print("y:", str(y))

    # 寻找边界线上的点，即支持向量点，并圈出来
    print("support vector:")
    for i in range(100):
        if alphas[i] > 0.0 and alphas[i] < C:
            print(str(dataArr[i]), " ", str(labelArr[i]))
            x = dataArr[i][0]
            y = dataArr[i][1]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5,
                        edgecolor='red')

    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()


###################################简化版SMO算法###################################
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
    dataMat = mat(dataMatIn);
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMat)

    # 初始化alphas和b
    alphas = mat(zeros((m, 1)));
    b = 0

    # 循环调整alphas 和 b
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # 根据fx = wx + b计算fx的值，即计算分类值，其中w是行向量， x是列向量
            # 通过拉格朗日乘子法优化可以得出w的值为(1~n)∑i alphai*yi*xi
            fXi = float(multiply(alphas, labelMat).T * \
                        (dataMat * dataMat[i, :].T)) + b
            # 计算上式的分类值和期望分类值的差距Ei
            Ei = fXi - float(labelMat[i])

            # 通过KKT条件推导，如下几个条件不满足，需要调整alpha
            # 1) yi * fxi > 1 且ai > 0                i点在边界内，ai应该为0
            # 2) yi * fxi < 1 且ai < c                i点在两条边界之间，ai应该为c
            # 3) yi * fxi == 1 且ai == 0 或 ai ==c    i点在边界线上，ai应该在0~c之间
            # 加上toler容错之后的条件如下???
            if ((labelMat[i] * Ei < -toler and alphas[i] < C) or \
                    (labelMat[i] * Ei > toler and alphas[i] > 0)):
                j = selectJRand(i, m)  # 随机选择另一个点j
                print("\n")
                print("\n")
                print(
                    "########### alpha and b adjust begin for [%d %d] dot ###########" % (
                    i, j))
                # 计算j点分类值
                fXj = float(multiply(alphas, labelMat).T * \
                            (dataMat * dataMat[j, :].T)) + b
                # 计算分类值和期望分类值的差距Ej
                Ej = fXj - float(labelMat[j])

                # 保存旧的alpha，=是引用幅值，所以需要copy一份新的
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                print(
                    "alphaIold:%f alphaJold:%f labelMat[i]:%d labelMat[j]:%d" % (
                    alphaIold, alphaJold, labelMat[i], labelMat[j]))

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
                eta = 2.0 * dataMat[i, :] * dataMat[j, :].T - \
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
                alphas[i] = alphaIold + labelMat[i] * labelMat[j] * (
                            alphaJold - alphas[j])

                print("after adjust [alphaInew:%f alphaJnew:%f]" % (
                alphas[i], alphas[j]))

                # 计算b1
                # b1New = bOld - Ei - yj * (alphaInew - alphaIold) * xi.T * xi - yj * (alphaJnew - alphaJold) * xj.T *xi
                b1 = b - Ei - labelMat[j] * (alphas[i] - alphaIold) * dataMat[i,
                                                                      :] * dataMat[
                                                                           i,
                                                                           :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMat[i,
                                                             :] * dataMat[j,
                                                                  :].T
                # b2New = bOld - Ej - yi * (alphaInew - alphaIold) * xi.T * xj - yj * (alphaJnew - alphaJold) * xj.T *xj
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i,
                                                                      :] * dataMat[
                                                                           j,
                                                                           :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMat[j,
                                                             :] * dataMat[j,
                                                                  :].T

                # 根据alphaInew和alphaJnew的范围求解新的b值
                if (0 < alphas[i] and alphas[i] < C):
                    b = b1
                elif (0 < alphas[j] and alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1.0
                print("after adjust [b:%f]" % b)
                # print("########### alpha and b adjust end for [%d %d] dot ###########" % (i, j))
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


###################################完整版Platt SMO算法###################################
class optStruct:
    """
    SMO算法相关数据结构，构成系统环境变量，里面的属性解释如下：
    X: 数据矩阵
    labelMat: 类别标签数组
    C: 松弛变量
    toler: 容错率
    m: 数据矩阵的行数
    alphas: 不断调整alphas矩阵
    b: 不断调整的b值
    eCache: 二维数组，第一列值代表是否有效的flag，第二列存储对应i节点当前的误差Ei
    """

    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = mat(dataMatIn)
        self.labelMat = mat(classLabels).transpose()
        self.C = C
        self.toler = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))


def calcEk(oS, k):
    """
    计算k点的误差值Ek
    :param oS: 当前系统环境变量
    :param k: 计算k点误差
    :return Ek: 返回误差值
    """
    # 根据fx = wx + b计算fx的值，即计算分类值，其中w是行向量， x是列向量
    # 通过拉格朗日乘子法优化可以得出w的值为(1~n)∑i alphai*yi*xi
    fXk = float(multiply(oS.alphas, oS.labelMat).T * \
                (oS.X * oS.X[k, :].T)) + oS.b
    # 计算上式的分类值和期望分类值的差距Ei
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def calcW(oS):
    """
    计算当前w值
    w的值为(1~n)∑i alphai*yi*xi
    :param oS: 当前系统环境变量
    :return w: 返回直线方程的法向量w
    """
    m, n = shape(oS.X)
    """
    w = (multiply(alphasMat, labelMat).T * dataMat).T
    该计算公式同下面的循环效果是一致的
    """
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(oS.alphas[i] * oS.labelMat[i].T, oS.X[i, :].T)
    return w


def selectJ(i, oS, Ei):
    """
    为i点寻找j点，找到的j点的alphaJnew是变化最大的点，加大alpha的变化步长，继而加快alphas和b的调整速度
    由于alphaJnew = alphaJold - yj(Ei - Ej) / η ，并且η是常量，所以就是求Ei - Ej值最大时的j点
    :param i: 为i点找j点
    :param oS: 当前系统环境变量
    :param Ei: i点的误差Ei
    :return j: 找到的j点
    :return Ej: j点的误差Ej
    """
    maxJ = -1;
    maxDeltaE = 0;
    Ej = 0
    oS.eCache[i] = [1, Ei]  # 更新环境变量中eCache某i点的Ei为有效值
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]  # 寻找eCache中有效的Ei所对应的坐标列表
    if len(validEcacheList > 1):
        for k in validEcacheList:
            if k == i:
                continue

            updateEk(oS, k)
            Ek = oS.eCache.A[k][1]
            deltaE = abs(Ei - Ej)
            if deltaE > maxDeltaE:
                maxJ = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxJ, Ej
    else:
        j = selectJRand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej


def updateEk(oS, k):
    """
    更新系统环境变量中k点的Ek值
    :param oS: 当前系统环境变量
    :param k: k点
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    """
    完整platt SMO算法中的优化例程，已知i点，调整alphas和b值
    :param i: i点
    :param oS: 当前环境变量
    :return alphaPairsChanged: 代表alphas矩阵是否更新过
    """

    Ei = calcEk(oS, i)  # 更新i点的Ei值

    # 通过KKT条件推导，如下几个条件不满足，需要调整alpha
    # 1) yi * fxi > 1 且ai > 0                i点在边界内，ai应该为0
    # 2) yi * fxi < 1 且ai < c                i点在两条边界之间，ai应该为c
    # 3) yi * fxi == 1 且ai == 0 或 ai ==c    i点在边界线上，ai应该在0~c之间
    # 加上toler容错之后的条件如下???
    if ((oS.labelMat[i] * Ei < -oS.toler and oS.alphas[i] < oS.C) or \
            (oS.labelMat[i] * Ei > oS.toler and oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # 随机选择另一个点j
        print("\n")
        print("\n")
        print(
            "########### alpha and b adjust begin for [%d %d] dot ###########" % (
            i, j))

        # 保存旧的alpha，=是引用幅值，所以需要copy一份新的
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        print("alphaIold:%f alphaJold:%f labelMat[i]:%d labelMat[j]:%d" % (
            alphaIold, alphaJold, oS.labelMat[i], oS.labelMat[j]))

        # 计算调整alpha的上限和下限
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, alphaJold - alphaIold)
            H = min(oS.C, oS.C + alphaJold - alphaIold)
        else:
            L = max(0, alphaJold + alphaIold - oS.C)
            H = min(oS.C, alphaJold + alphaIold)
        print("L~H:[%f~%f]" % (L, H))

        if L == H:
            print("L==H")
            return 0

        # 计算η
        # η = 2 * xi * xj.T - xi * xi.T - xj * xj.T
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - \
              oS.X[i, :] * oS.X[i, :].T - \
              oS.X[j, :] * oS.X[j, :].T
        print("eta:%f" % eta)
        if eta >= 0:
            print("eta>=0")
            return 0

        # 计算新的alphas[j]
        # alphaJnew = alphaJold - yj(Ei - Ej) / η
        oS.alphas[j] = alphaJold - oS.labelMat[j] * (Ei - Ej) / eta
        # 修建alphaJnew
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        print("alphaJnew:%f alphaJold:%f" % (oS.alphas[j], alphaJold))
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not move enough")
            return 0

        # 计算新的alphas[i]
        # alphaInew = alphaIold + yi*yj*(alphaJold - alphaJnewClipped)
        # 可以看出来alphaI的变化和alphaJ的变化量是相同的，但是方向可能相反
        oS.alphas[i] = alphaIold + oS.labelMat[i] * oS.labelMat[j] * (
                    alphaJold - oS.alphas[j])
        updateEk(oS, i)

        print("after adjust [alphaInew:%f alphaJnew:%f]" % (
        oS.alphas[i], oS.alphas[j]))

        # 计算b1
        # b1New = bOld - Ei - yj * (alphaInew - alphaIold) * xi.T * xi - yj * (alphaJnew - alphaJold) * xj.T *xi
        b1 = oS.b - Ei - oS.labelMat[j] * (oS.alphas[i] - alphaIold) * oS.X[i,
                                                                       :] * oS.X[
                                                                            i,
                                                                            :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j,
                                                                        :].T
        # b2New = bOld - Ej - yi * (alphaInew - alphaIold) * xi.T * xj - yj * (alphaJnew - alphaJold) * xj.T *xj
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i,
                                                                       :] * oS.X[
                                                                            j,
                                                                            :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j,
                                                                        :].T

        # 根据alphaInew和alphaJnew的范围求解新的b值
        if (0 < oS.alphas[i] and oS.alphas[i] < oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j] and oS.alphas[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0

        print("after adjust [b:%f]" % oS.b)
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter):
    """
    完整platt SMO算法外部循环
    :param dataMatIn: 输入数据矩阵
    :param classLabels: 类型标签数组
    :param C: 松弛变量
    :param toler: 容错率
    :param maxIter: 最大循环次数
    :return b: 调整出的最佳b值
    :return alphas: 调整出的最佳alphas值
    """
    oS = optStruct(dataMatIn, classLabels, C, toler)
    # print("oS.X:", str(oS.X))
    # print("oS.classLabels:", str(oS.labelMat))
    iter = 0
    entireSet = True
    alphasPairsChanged = 0
    while (iter < maxIter) and (alphasPairsChanged > 0 or entireSet):
        alphasPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphasPairsChanged = alphasPairsChanged + innerL(i, oS)
                print("fullSet, iter: %d, alphasPairsChanged: %d" % (
                iter, alphasPairsChanged))
            iter += 1
        else:
            """
            >>> arr = array([1, 2, 3, 4])
            >>> arr > 2
            array([False, False,  True,  True])
            >>> nonzero(arr > 2)
            (array([2, 3], dtype=int64),)
            """
            # 对于支持向量，0 < alpha < C
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]
            for i in nonBoundIs:
                alphasPairsChanged = alphasPairsChanged + innerL(i, oS)
                print("nonBoundIs, iter: %d, alphasPairsChanged: %d" % (
                iter, alphasPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # 完整集合和非边界集合交替进行
        elif alphasPairsChanged == 0:
            entireSet = True
        print("iter: %d" % iter)
    """
    w1 = multiply(oS.alphas, oS.labelMat).T * oS.X
    w2 = calcW(oS)
    print("w1:", str(w1))
    print("w2:", str(w2))
    """
    return oS.b, oS.alphas


###################################非线性SVM相关###################################
def singleKernelTrans(dataMat, A, kTup):
    """
    使用kTup指定的核函数计算数据矩阵dataMat针对向量A的核函数值
    :param dataMat: 数据矩阵
    :param A: 向量
    :param kTup: 核函数描述信息
    :return k: 一列核函数值，行数为数据矩阵的行数
    """
    m, n = shape(dataMat)
    k = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        # 如果线性svm，直接计算每个向量与i向量的点积作为K矩阵的第i列值
        k = dataMat * A.T
    elif kTup[0] == 'rbf':
        # 如果是径向基核函数，计算每个向量与i向量径向基核函数的值作为K矩阵的第i列值
        # 径向基核函数:exp(-1 * (||x -y||^2 / 2 * σ ^ 2))，其中x 与 y是向量，σ为确定到达率或者说是函数值跌落到0的速度参数
        for j in range(m):
            deltaRow = dataMat[j, :] - A
            k[j] = deltaRow * deltaRow.T
        k = exp(k / (-1 * kTup[1] ** 2))
    else:
        raise NameError("The kernel is not recognized")
    return k


def allKernelTrans(oS, kTup):
    """
    核转换函数
    :param oS: 当前系统环境变量
    :param kTup: 核函数信息 kTup[0]保存核函数的描述字符，kTup[1]保存核函数相关参数
    :return K: 返回存储核函数值的矩阵(m * m)
    """
    K = mat(zeros((oS.m, oS.m)))
    for i in range(oS.m):
        # 计算每个向量与i向量的核函数值k作为K的第i列值
        K[:, i] = singleKernelTrans(oS.X, oS.X[i, :], kTup)
    return K


class optStructWithK:
    """
    SMO算法相关数据结构，构成系统环境变量，同optStruct相比较，多了存储核函数值的矩阵K
    里面的属性解释如下：
    X: 数据矩阵
    labelMat: 类别标签数组
    C: 松弛变量
    toler: 容错率
    m: 数据矩阵的行数
    alphas: 不断调整alphas矩阵
    b: 不断调整的b值
    eCache: 二维数组，第一列值代表是否有效的flag，第二列存储对应i节点当前的误差Ei
    K: 存储核函数值的矩阵
    """

    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = mat(dataMatIn)
        self.labelMat = mat(classLabels).transpose()
        self.C = C
        self.toler = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = allKernelTrans(self, kTup)


def calcEkWithK(oS, k):
    """
    计算k点的误差值Ek,跟calcEk函数的不同之处在于使用核函数值代替向量的点积
    :param oS: 当前系统环境变量
    :param k: 计算k点误差
    :return Ek: 返回误差值
    """
    # 根据fx = wx + b计算fx的值，即计算分类值，其中w是行向量， x是列向量
    # 通过拉格朗日乘子法优化可以得出w的值为(1~n)∑i alphai*yi*xi
    # fx = (1~n)∑i alphai*yi*xi * x + b => fx = (1~n)∑i alphai*yi*<xi, x> + b => (1~n)∑i alphai*yi*K(xi, x) + b
    fXk = float(multiply(oS.alphas, oS.labelMat).T * \
                oS.K[:, k]) + oS.b
    # 计算上式的分类值和期望分类值的差距Ei
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJwithK(i, oS, Ei):
    """
    为i点寻找j点，找到的j点的alphaJnew是变化最大的点，加大alpha的变化步长，继而加快alphas和b的调整速度
    由于alphaJnew = alphaJold - yj(Ei - Ej) / η ，并且η是常量，所以就是求Ei - Ej值最大时的j点
    与selectJ函数区别在于使用updateEkWithK代替updateEk来更新Ek
    :param i: 为i点找j点
    :param oS: 当前系统环境变量
    :param Ei: i点的误差Ei
    :return j: 找到的j点
    :return Ej: j点的误差Ej
    """
    maxJ = -1;
    maxDeltaE = 0;
    Ej = 0
    oS.eCache[i] = [1, Ei]  # 更新环境变量中eCache某i点的Ei为有效值
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]  # 寻找eCache中有效的Ei所对应的坐标列表
    if len(validEcacheList > 1):
        for k in validEcacheList:
            if k == i:
                continue

            updateEkWithK(oS, k)
            Ek = oS.eCache.A[k][1]
            deltaE = abs(Ei - Ej)
            if deltaE > maxDeltaE:
                maxJ = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxJ, Ej
    else:
        j = selectJRand(i, oS.m)
        Ej = calcEkWithK(oS, j)
        return j, Ej


def updateEkWithK(oS, k):
    """
    更新系统环境变量中k点的Ek值
    与updateEk函数的区别在于使用calcEkWithK代替calcEk计算Ek
    :param oS: 当前系统环境变量
    :param k: k点
    """
    Ek = calcEkWithK(oS, k)
    oS.eCache[k] = [1, Ek]


def innerLwithK(i, oS):
    """
    完整platt SMO算法中的优化例程，已知i点，调整alphas和b值, 和innerL函数的不同之处在于：
    使用核函数计算η：η = 2 * xi * xj.T - xi * xi.T - xj * xj.T => η = 2 * K[i, j]- K[i, i] - K[j, j]
    使用核函数计算b1: b1New = bOld - Ei - yj * (alphaInew - alphaIold) * xi.T * xi - yj * (alphaJnew - alphaJold) * xj.T *xi
    => b1New = bOld - Ei - yj * (alphaInew - alphaIold) * K[i, i] - yj * (alphaJnew - alphaJold) * K[j, i]
    使用核函数计算b2: b2New = bOld - Ej - yi * (alphaInew - alphaIold) * xi.T * xj - yj * (alphaJnew - alphaJold) * xj.T *xj
    => b2New = bOld - Ej - yi * (alphaInew - alphaIold) * K[i, j] - yj * (alphaJnew - alphaJold) * K[j, j]
    :param i: i点
    :param oS: 当前环境变量
    :return alphaPairsChanged: 代表alphas矩阵是否更新过
    """

    Ei = calcEkWithK(oS, i)  # 更新i点的Ei值

    # 通过KKT条件推导，如下几个条件不满足，需要调整alpha
    # 1) yi * fxi > 1 且ai > 0                i点在边界内，ai应该为0
    # 2) yi * fxi < 1 且ai < c                i点在两条边界之间，ai应该为c
    # 3) yi * fxi == 1 且ai == 0 或 ai ==c    i点在边界线上，ai应该在0~c之间
    # 加上toler容错之后的条件如下???
    if ((oS.labelMat[i] * Ei < -oS.toler and oS.alphas[i] < oS.C) or \
            (oS.labelMat[i] * Ei > oS.toler and oS.alphas[i] > 0)):
        j, Ej = selectJwithK(i, oS, Ei)  # 随机选择另一个点j
        print("\n")
        print("\n")
        print(
            "########### alpha and b adjust begin for [%d %d] dot ###########" % (
            i, j))

        # 保存旧的alpha，=是引用幅值，所以需要copy一份新的
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        print("alphaIold:%f alphaJold:%f labelMat[i]:%d labelMat[j]:%d" % (
            alphaIold, alphaJold, oS.labelMat[i], oS.labelMat[j]))

        # 计算调整alpha的上限和下限
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, alphaJold - alphaIold)
            H = min(oS.C, oS.C + alphaJold - alphaIold)
        else:
            L = max(0, alphaJold + alphaIold - oS.C)
            H = min(oS.C, alphaJold + alphaIold)
        print("L~H:[%f~%f]" % (L, H))

        if L == H:
            print("L==H")
            return 0

        # 计算η
        # η = 2 * K[i, j]- K[i, i] - K[j, j]
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        print("eta:%f" % eta)
        if eta >= 0:
            print("eta>=0")
            return 0

        # 计算新的alphas[j]
        # alphaJnew = alphaJold - yj(Ei - Ej) / η
        oS.alphas[j] = alphaJold - oS.labelMat[j] * (Ei - Ej) / eta
        # 修建alphaJnew
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEkWithK(oS, j)
        print("alphaJnew:%f alphaJold:%f" % (oS.alphas[j], alphaJold))
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not move enough")
            return 0

        # 计算新的alphas[i]
        # alphaInew = alphaIold + yi*yj*(alphaJold - alphaJnewClipped)
        # 可以看出来alphaI的变化和alphaJ的变化量是相同的，但是方向可能相反
        oS.alphas[i] = alphaIold + oS.labelMat[i] * oS.labelMat[j] * (
                    alphaJold - oS.alphas[j])
        updateEkWithK(oS, i)

        print("after adjust [alphaInew:%f alphaJnew:%f]" % (
        oS.alphas[i], oS.alphas[j]))

        # 计算b1
        # b1New = bOld - Ei - yj * (alphaInew - alphaIold) * K[i, i] - yj * (alphaJnew - alphaJold) * K[j, i]
        b1 = oS.b - Ei - oS.labelMat[j] * (oS.alphas[i] - alphaIold) * oS.K[
            i, i] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, i]
        # b2New = bOld - Ej - yi * (alphaInew - alphaIold) * K[i, j] - yj * (alphaJnew - alphaJold) * K[j, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[
            i, j] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]

        # 根据alphaInew和alphaJnew的范围求解新的b值
        if (0 < oS.alphas[i] and oS.alphas[i] < oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j] and oS.alphas[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0

        print("after adjust [b:%f]" % oS.b)
        return 1
    else:
        return 0


def smoPWithK(dataMatIn, classLabels, C, toler, maxIter, kTup):
    """
    完整platt SMO算法外部循环
    与smoP的区别在于使用innerLwithK代替innerL，使用optStructWithK代替optStruct
    :param dataMatIn: 输入数据矩阵
    :param classLabels: 类型标签数组
    :param C: 松弛变量
    :param toler: 容错率
    :param maxIter: 最大循环次数
    :param kTup: 存储核函数相关信息 kTup[0]存储核函数描述信息，kTup[1]存储核函数相关参数
    :return b: 调整出的最佳b值
    :return alphas: 调整出的最佳alphas值
    """
    oS = optStructWithK(dataMatIn, classLabels, C, toler, kTup)
    # print("oS.X:", str(oS.X))
    # print("oS.classLabels:", str(oS.labelMat))
    iter = 0
    entireSet = True
    alphasPairsChanged = 0
    while (iter < maxIter) and (alphasPairsChanged > 0 or entireSet):
        alphasPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphasPairsChanged = alphasPairsChanged + innerLwithK(i, oS)
                print("fullSet, iter: %d, alphasPairsChanged: %d" % (
                iter, alphasPairsChanged))
            iter += 1
        else:
            """
            >>> arr = array([1, 2, 3, 4])
            >>> arr > 2
            array([False, False,  True,  True])
            >>> nonzero(arr > 2)
            (array([2, 3], dtype=int64),)
            """
            # 对于支持向量，0 < alpha < C
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]
            for i in nonBoundIs:
                alphasPairsChanged = alphasPairsChanged + innerLwithK(i, oS)
                print("nonBoundIs, iter: %d, alphasPairsChanged: %d" % (
                iter, alphasPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # 完整集合和非边界集合交替进行
        elif alphasPairsChanged == 0:
            entireSet = True
        print("iter: %d" % iter)
    """
    w1 = multiply(oS.alphas, oS.labelMat).T * oS.X
    w2 = calcW(oS)
    print("w1:", str(w1))
    print("w2:", str(w2))
    """
    return oS.b, oS.alphas


def showSupportVector(dataArr, labelArr, alphas):
    """
    展示分类结果，圈出支持向量
    :param dataArr: 数据点矩阵
    :param labelArr: 数据点对应的标签
    :param alphas: 通过smo调整出的alphas最佳值
    """
    labelMat = mat(labelArr).T
    alphasMat = mat(alphas)
    dataMat = mat(dataArr)

    # 绘制所有不同分类的数据点
    data_plus = []  # 正类型数据样本
    data_minus = []  # 负类型数据样本
    for i in range(len(dataMat)):
        if labelArr[i] == 1:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])

    fit = plt.figure()
    ax = fit.add_subplot(111)
    data_plus_np = array(data_plus)  # 转成numpy的array才能transpose
    data_mins_np = array(data_minus)
    ax.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], s=30,
               c='red')
    ax.scatter(transpose(data_mins_np)[0], transpose(data_mins_np)[1], s=30,
               c='green')

    # 寻找边界线上的点，即支持向量点，并圈出来
    print("support vector:")
    for i in range(100):
        if alphas[i] > 0.0:
            print(str(dataArr[i]), " ", str(labelArr[i]))
            x = dataArr[i][0]
            y = dataArr[i][1]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5,
                        edgecolor='red')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def rbfTest(k1=1.3):
    """
    测试径向基核数来分类非线性数据的效果
    :param k1:径向基函数的参数σ
    """
    trainDataArr, trainClassLabel = loadDataSet("testSetRBF.txt")
    kTup = ('rbf', k1)
    b, alphas = smoPWithK(trainDataArr, trainClassLabel, 0.6, 0.001, 40, kTup)
    showSupportVector(trainDataArr, trainClassLabel, alphas)
    dataMat = mat(trainDataArr)
    labelMat = mat(trainClassLabel).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    # 提取所有的支持向量及对应标签
    sVs = dataMat[svInd]
    labelSv = labelMat[svInd]
    print("there is %d support vector" % shape(sVs)[0])
    print("######## calculate the error rate for train data ########")
    m, n = shape(dataMat)
    trainErrorCount = 0.0
    # 使用支持向量计算训练数据分类错误概率
    for i in range(m):
        kernelEval = singleKernelTrans(sVs, dataMat[i, :], kTup)
        predict = float(multiply(alphas[svInd], labelSv).T * kernelEval) + b
        # print("for [%d] vector, the predict is %f" % (i, predict))
        if sign(predict) != sign(labelMat[i]):
            trainErrorCount += 1.0
    print("train error rate is %f" % float(trainErrorCount / float(m)))

    # 使用支持向量计算测试数据分类错误概率
    print("######## calculate the error rate for test data ########")
    testDataArr, testClassLabel = loadDataSet("testSetRBF2.txt")
    testErrorCount = 0.0
    dataMat = mat(testDataArr)
    labelMat = mat(testClassLabel).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = singleKernelTrans(sVs, dataMat[i, :], kTup)
        predict = float(multiply(alphas[svInd], labelSv).T * kernelEval) + b
        # print("for [%d] vector, the predict is %f" % (i, predict))
        if sign(predict) != sign(labelMat[i]):
            testErrorCount += 1.0
    print("test error rate is %f" % float(testErrorCount / float(m)))


if __name__ == '__main__':
    """
    # 测试线性分割svm
    dataArr, classLabels = loadDataSet('testSet.txt')
    showDataSet(dataArr, classLabels)
    # b, alphas = smoSimple(dataArr, classLabels, 0.6, 0.001, 40) # 简化版svm算法
    # b, alphas = smoP(dataArr, classLabels, 0.6, 0.001, 40) # 完整platt SMO算法

    # 包含核函数的完整platt SMO算法，核函数选择线性核函数
    b, alphas = smoPWithK(dataArr, classLabels, 0.6, 0.001, 40, kTup=('lin', 0))
    showClassifer(dataArr, classLabels, b, alphas, 0.6)
    """

    # 测试非线性分割svm
    rbfTest(0.7)
