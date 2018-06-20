from numpy import *
from bs4 import BeautifulSoup
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

def lwlr(testPoint, dataArr, labels, k=1.0):
    """
    局部加权线性回归（locally weighted linear regression），我们给预测的点附近的点赋予更大些的权重
    该回归的最佳系数w计算公式如下:
    w = (X.T*W*X)^(-1) * X.T*W*Y
    W是一个对角矩阵，对角元素对应每个数据点的权重
    使用高斯核函数w(i,i) = e^(|x(i) - x| / (-2(k^2))) 高斯核函数能搞使点附近的点得到更大的权重
    :param testPoint: 测试数据点
    :param dataArr: 全部数据点
    :param labels: 标签数组
    :param k: 高斯核的参数(决定了数据点附近的点赋予多大的权重)
    :return yHat: 预测值
    """
    dataMat = mat(dataArr)
    labelMat = mat(labels)
    m = shape(dataMat)[0]
    weights = mat(eye(m))           # 创建单位矩阵
    pointMat = mat(testPoint)
    for j in range(m):
        diffMat = dataMat[j, :] - pointMat
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    print("weights:", str(weights))
    X = dataMat
    y = labelMat.T
    xTx = X.T * (weights * X)
    # 矩阵的求逆公式为X^-1 = X* / det(X)    X*为伴随矩阵（每个元素的代数余子式组成的矩阵）   det(X)为矩阵的秩
    if linalg.det(xTx) == 0.0:
        print("The matrix is singular, connot do inverse")
        return
    w = (xTx).I * (X.T * weights * y)
    #print("w:", str(w))
    return testPoint * w

def lwlrTest(testArr, dataArr, labels, k=1.0):
    """
    lwlr测试函数，将testArr中的数据点预测出对应的值
    :param testArr: 测试数据
    :param dataArr: 训练数据
    :param labels: 训练数据的标签
    :param k: 高斯核函数的参数
    :return:
    """
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], dataArr, labels, k)
    return yHat

def showLwlr(dataArr, labels, yHat):
    """
    展示二维数据以及它的预测值的线
    :param dataArr: 数据矩阵
    :param labels: 标签数组
    :param yHat: 预测出的值
    :return:
    """
    fit = plt.figure()
    ax = fit.add_subplot(111)

    # 绘制所有的点
    ax.scatter((array(dataArr).T)[1], array(labels), s=10, c="red")

    # 绘制预测线
    dataMat = mat(dataArr)
    sortInd = dataMat[:, 1].argsort(0).T.tolist()[0]
    X = dataMat[sortInd][:, 1]
    Y = yHat[sortInd]
    # print("X:", str(dataMat[sortInd][:, 1]))
    # print("Y:", str(yHat[sortInd]))
    ax.plot(X, Y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def rssError(yArr, yHatArr):
    """
    计算误差
    :param yArr: 真实结果数组
    :param yHatArr: 预测结果数组
    :return error: 误差值
    """
    return ((array(yArr) - array(yHatArr)) ** 2).sum()

def ridgeRegres(xMat, yMat, lam=0.2):
    """
    岭回归计算w系数
    :param xArr: 数据
    :param yArr: 标签
    :param lam: lambda值
    :return w: 通过lambda计算出的回归系数
    """
    xTx = xMat.T * xMat
    m, n = shape(xMat)
    I = mat(eye(n))
    denom = xTx + lam * I
    #print("denom:", str(denom))
    if linalg.det(denom) == 0.0:
        print("The matrix is singular, cannot do inverse")
    w = denom.I * xMat.T * yMat
    # print("w:", str(w))
    return w

def standData(xArr, yArr):
    """
    标准化数据
    :param xArr: 需要标准化的xArr
    :param yArr: 需要标准化的yArr
    :return xMat: 标准化后的xMat
    :return yMat: 标准化后的yMat
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T

    yMean = mean(yMat, 0)   # 计算平均值
    yMat = yMat - yMean

    xMean = mean(xMat, 0)   # 计算平均值
    xMat = xMat - xMean
    xVar = var(xMat, 0)     # 计算平方差
    xMat = xMat / xVar
    return xMat, yMat

def ridgeTest(xArr, yArr):
    """
    岭回归测试，并绘制随着lambda的变大，w系数的变化情况
    :param xArr: 数据
    :param yArr: 标签
    :return:
    """
    xMat, yMat = standData(xArr, yArr)

    numTest = 30                    # 30个lambda值
    wMat = zeros((numTest, shape(xMat)[1]))

    # print("xMat:", str(xMat))
    # print("yMat:", str(yMat))
    for i in range(numTest):
        wMat[i, :] = ridgeRegres(xMat, yMat, exp(i - 10)).T

    # 绘制w系数值与i(log(lambda))的关系
    # fit = plt.figure()
    # ax = fit.add_subplot(111)
    # ax.plot(wMat)
    # plt.xlabel("log(lambda)")
    # plt.ylabel("w")
    # plt.show()
    return wMat

def stageWise(xArr, yArr, eps=0.01, numIt=100):
    """
    前向逐步线性回归，属于一种贪心算法，即每一步尽可能减少误差。
    :param xArr: 数据矩阵
    :param yArr: 标签
    :param eps: 调整步长
    :param numIt: 调整迭代次数
    :return returnMat: 系数矩阵，每一行代表每次迭代调整的系数
    """
    # 数据标准化
    xMat, yMat = standData(xArr, yArr)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))                   # 整个调整的系数矩阵
    ws = zeros((n, 1))                              # 记录每次迭代调整好的系数列向量
    wsTest = ws.copy()                              # 每次迭代中，记录不断调整和测试的系数列向量
    wsMax = ws.copy()                               # 每次迭代中，记录最佳的系数列向量
    for i in range(numIt):
        print("ws:", str(ws))
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += (sign) * eps
                yHat = xMat * wsTest
                err = rssError(yMat.A, yHat.A)
                if err < lowestError:
                    lowestError = err
                    wsMax = wsTest
        ws = wsMax.copy()
        print("lowestError:", str(lowestError))
        returnMat[i, :] = ws.T
    return returnMat

def showMat(mat, xLabel, yLabel):
    """
    展示矩阵行向量随着行数增加的变化
    :param mat: 矩阵
    """
    fit = plt.figure()
    ax = fit.add_subplot(111)
    ax.plot(mat)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()


def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    """
    函数说明:从页面读取数据，生成retX和retY列表
    Parameters:
        retX - 数据X
        retY - 数据Y
        inFile - HTML文件
        yr - 年份
        numPce - 乐高部件数目
        origPrc - 原价
    Returns:
        无
    """
    # 打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r="%d" % i)
    while (len(currentRow) != 0):
        currentRow = soup.find_all('table', r="%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                #print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r="%d" % i)

def setDataCollect(retX, retY):
    """
    函数说明:依次读取六种乐高套装的数据，并生成数据矩阵
    Parameters:
        无
    Returns:
        无
    """
    scrapePage(retX, retY, './lego/lego8288.html', 2006, 800, 49.99)     # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, './lego/lego10030.html', 2002, 3096, 269.99)  # 2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, './lego/lego10179.html', 2007, 5195, 499.99)  # 2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, './lego/lego10181.html', 2007, 3428, 199.99)  # 2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, './lego/lego10189.html', 2008, 5922, 299.99)  # 2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, './lego/lego10196.html', 2009, 3263, 249.99)

def crossValidation(xArr, yArr, numIt=10):
    """
    交叉验证岭回归
    :param xArr: 特征数据矩阵
    :param yArr: 标签数据
    :param numIt: 迭代次数
    :return w: 回归系数矩阵
    """
    m = len(yArr)
    indexList = list(range(m))
    matErrors = zeros((numIt, 30))             # 总共测试numIt组，每组有30个lambda
    for i in range(numIt):
        trainX = []; trainY = []
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):
            if j < 0.9 * m:                         # 将百分之九十的数据划到训练数据中
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)
        for k in range(30):
            # 标准化数据
            testXmat = mat(testX); trainXMat = mat(trainX)
            meanTrain = mean(trainXMat, 0)
            varTrain = var(trainXMat, 0)
            testXmat = (testXmat - meanTrain) / varTrain

            yHat = testXmat * mat(wMat[k, :]).T + mean(trainY)
            matErrors[i, k] = rssError(mat(testY).T, yHat)
    # print("matErrors:", str(matErrors))
    meanErrors = mean(matErrors, 0)
    # print("meanErrors:", str(meanErrors))
    minErr = float(min(meanErrors))
    print("ridge regression minErr:", minErr)
    bestWeights = wMat[nonzero(meanErrors == minErr)]
    return bestWeights

if __name__ == "__main__":
    # 测试普通最小二乘法
    """
    dataArr, labels = loadDataSet("ex0.txt")
    print("dataArr:", str(dataArr))
    print("labels:", str(labels))
    w = standRgres(dataArr, labels)
    showRegres(dataArr, labels, w)
    cc = calcCorrCoef(dataArr, labels, w)
    print("cc:", str(cc))
    """

    # 测试局部加权线性回归
    """
    testArr, labels = loadDataSet("ex0.txt")
    # k为1 欠拟合，效果和普通最小二乘法一致
    # k为0.003，过拟合，拟合的直线与数据点过于贴近
    # k为0.01，是一个很好的估计
    yHat = lwlrTest(testArr, testArr, labels, 0.01)
    print("yHat", str(yHat))
    showLwlr(testArr, labels, yHat)
    """

    # 使用局部加权线性回归估测鲍鱼的年龄
    """
    dataArr, labels = loadDataSet("abalone.txt")
    print("dataArr:", str(dataArr))
    print("labels:", str(labels))
    print("Begin trainning...")
    yHat01 = lwlrTest(dataArr[0:99], dataArr[0:99], labels[0:99], 0.1)
    yHat1 = lwlrTest(dataArr[0:99], dataArr[0:99], labels[0:99], 1)
    yHat10 = lwlrTest(dataArr[0:99], dataArr[0:99], labels[0:99], 10)
    print("error of yHat01:", str(rssError(labels[0:99], yHat01)))
    print("error of yHat1:", str(rssError(labels[0:99], yHat1)))
    print("error of yHat10:", str(rssError(labels[0:99], yHat10)))
    print("Begin testing...")
    yHat01 = lwlrTest(dataArr[100:199], dataArr[0:99], labels[0:99], 0.1)
    yHat1 = lwlrTest(dataArr[100:199], dataArr[0:99], labels[0:99], 1)
    yHat10 = lwlrTest(dataArr[100:199], dataArr[0:99], labels[0:99], 10)
    print("error of yHat01:", str(rssError(labels[100:199], yHat01)))
    print("error of yHat1:", str(rssError(labels[100:199], yHat1)))
    print("error of yHat10:", str(rssError(labels[100:199], yHat10)))
    print("Begin standRegression testing...")
    w = standRgres(dataArr[0:99], labels[0:99])
    yHat = mat(dataArr[100:199]) * w
    print("error of yHat:", str(rssError(labels[100:199], yHat.T.A)))
    """

    # 岭回归测试
    """
    dataArr, labels = loadDataSet("abalone.txt")
    wMat = ridgeTest(dataArr[0:99], labels[0:99])
    # 使用标准化的X和Y进行测试
    xMat, yMat = standData(dataArr[0:99], labels[0:99])
    print("xMat:", str(xMat), " yMat:", str(yMat))
    # print("wMat:", str(wMat))
    # 寻找训练数据最佳系数值
    minError = inf
    bestLambda = 0.0
    bestYhat = []
    for i in range(shape(wMat)[0]):
        print(str(wMat[i, :]))
        yHat = xMat * mat(wMat[i, :]).T
        print("yHat:", str(yHat))
        error = rssError(yMat, yHat)
        print("lambda:", str(exp(i - 10)), " error:", str(error))
        if minError > error:
            minError = error
            bestLambda = exp(i - 10)
            bestYhat = yHat
    print("train result minError:", str(minError), " bestLambda:", str(bestLambda), "bestYhat:", bestYhat.T)

    # 使用标准化的X和Y进行测试
    xMat, yMat = standData(dataArr[100:199], labels[100:199])
    print("xMat:", str(xMat), " yMat:", str(yMat))
    # print("wMat:", str(wMat))
    minError = inf
    bestLambda = 0.0
    bestYhat = []
    for i in range(shape(wMat)[0]):
        print(str(wMat[i, :]))
        yHat = xMat * mat(wMat[i, :]).T
        print("yHat:", str(yHat))
        error = rssError(yMat, yHat)
        print("lambda:", str(exp(i - 10)), " error:", str(error))
        if minError > error:
            minError = error
            bestLambda = exp(i - 10)
            bestYhat = yHat
    print("train result minError:", str(minError), " bestLambda:", str(bestLambda), "bestYhat:", bestYhat.T)
    """

    # 逐步线性回归算法测试
    """
    dataArr, labels = loadDataSet("abalone.txt")
    returnMat = stageWise(dataArr, labels, 0.01, 1000)
    showMat(returnMat, "n", "w")
    """

    # 乐高套装价格估计测试
    retX = []                                   # 保存乐高套装商品相关参数
    retY = []                                   # 保存乐高套装的价格
    setDataCollect(retX, retY)                  # 从网页中爬取乐高套装商品的资料
    print("retX:", str(retX))
    print("retY:", str(retY))
    m, n = shape(retX)
    retX1 = ones((m, n+1))
    retX1[:, 1:5] = mat(retX)
    w = standRgres(retX1, retY)                 # 使用普通线性回归计算回归系数
    print("price = %f%+f*year%+f*numParts%+f*newFlag%+f*origPrice" % (w[0], w[1], w[2], w[3], w[4]))
    yHat = retX1 * w
    err = rssError(mat(retY).T, yHat)
    print("stand regression err:", err)

    bestWeights = crossValidation(retX, retY, 10)  # 使用岭回归交叉测试计算最佳回归系数
    # print("bestWeights:", str(bestWeights))
    xMat = mat(retX);
    yMat = mat(retY).T
    meanX = mean(xMat, 0);
    varX = var(xMat, 0)
    unReg = bestWeights / varX                  # 由于岭回归数据经过标准化，因此需要还原
    """
    还原公式推算如下：
    设xMat'为xMat标准化后的数据, yHat'为yMat标准化后的数据
    则：yHat' = xMat' * weights
    且xMat' = (xMat - xMean) / xVar
    带入上式得：
    yMat' = (xMat - xMean) * (weights / xVar)
    且yHat' = yMat - yMean
    带入上式得：
    yMat = -xMean * (weights / xVar) + yMean + xMat * (weights / xVar)
    其中常数部分为-xMean * (weights / xVar) + yMean
    系数部分为weights / xVar
    """
    print("price = %f%+f*year%+f*numParts%+f*newFlag%+f*origPrice" % (
        (-1 * sum(multiply(meanX, unReg)) + mean(yMat)), unReg[0, 0], unReg[0, 1], unReg[0, 2], unReg[0, 3]))
