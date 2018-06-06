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
        length = len(lineArr)
        tmpArr = []
        for i in range(length - 1):
            tmpArr.append(float(lineArr[i]))
        dataArr.append(tmpArr)
        classLabels.append(float(lineArr[-1]))
    return dataArr, classLabels

def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    """
    对特征数据矩阵中某一列特征，使用阈值和比较符号进行分类
    :param dataMat: 特征数据矩阵
    :param dimen: 列值
    :param threshVal: 阈值
    :param threshIneq: 比较符号
    :return retArray: 分类结果数组
    """
    retArr = ones((shape(dataMat)[0], 1))
    if threshIneq == 'lt':
        retArr[dataMat[:, dimen] <= threshVal] = -1.0
    elif threshIneq == 'gt':
        retArr[dataMat[:, dimen] > threshVal] = -1.0
    return retArr

def buildStump(dataArr, classLabels, D):
    """
    根据当前样本权重向量D，构建最佳的单层决策树
    :param dataArr: 特征数据数组
    :param classLabels: 类型标签数组
    :param D: 样本权重向量
    :return bestStump: 最佳的单层决策树描述信息，包括使用第几列特征进行分类，该列特征数据的阈值，分类的比较符号
    :return minError: 最佳单层决策树的错误概率
    :return bestClasEst: 最佳分类结果
    """
    dataMat = mat(dataArr); labelMat = mat(classLabels).T
    m, n = shape(dataMat)
    numSteps = 10.0                                 # 步数
    bestStump = {}                                  # 最佳单层决策树
    bestClasEst = mat(zeros((m, 1)))                # 最佳分类结果
    minError = inf                                  # 最佳单层决策树的错误概率,初始值为正无穷

    for i in range(n):                              # 对于每列特征进行循环
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        stepLen = (rangeMax - rangeMin) / numSteps  # 根据步数计算步长

        for j in range(-1, int(numSteps) + 1):      # 步数从-1到+11
            threshVal = rangeMin + stepLen * float(j)

            for threshIneq in ['lt', 'gt']:         # 在lt和gt两者之间交替
                predictVal = stumpClassify(dataMat, i, threshVal, threshIneq)
                errArr = mat(ones((m, 1)))
                errArr[predictVal == labelMat] = 0  # 预测正确，置为零
                weightedError = D.T * errArr        # 计算预测错误的样本加权和
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictVal.copy() # 需要使用copy函数防止引用赋值
                    bestStump['dimen'] = i
                    bestStump['threshVal'] = threshVal
                    bestStump['threshIneq'] = threshIneq
    print("buildStump result: \nbestStump:", str(bestStump), "\nminError:",
          str(minError))
    return bestStump, minError, bestClasEst

def adaBoostTrain(dataArr, classLabel, numIt = 40):
    """
    基于单层决策树的ada boost训练过程
    :param dataArr: 特征数据
    :param classLabel: 类型标签数组
    :param numIt: 训练最大轮数
    :return weakClassArr: 多个弱分类器，并且带有每个分类器权值
    """
    weakClassArr = []                               # 弱分类器存储数组
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)                       # 存储样本权重向量，总和为1，初始值每个值为1/m
    aggClassEst = mat(zeros((m, 1)))                # 每轮分类器分类结果的加权和值
    for i in range(numIt):
        print("#######################[classifier %d]#######################" % i)
        #print("D:", str(D.T))
        bestStump, minError, bestClasEst = buildStump(dataArr, classLabel, D)
        # alpha = (1/2) * ln((1 - error) / error))
        alpha = float(0.5 * log((1 - minError) /\
                        max(minError, 1e-16)))      # 计算本轮测试alpha的值, 1e-16防止没有错误时除0溢出
        bestStump['alpha'] = alpha                  # 将本轮分类器alpha的值存储到分类器中
        #print("alpha:", str(alpha))
        weakClassArr.append(bestStump)

        # 当样本预测正确，D(i+1) = (D(i) * e^(-alpha)) / Sum(D)
        # 当样本预测错误，D(i+1) = (D(i) * e^(alpha)) / Sum(D)
        # 即预测错误的样本下一轮分类时样本权值更大，预测正确的样本下一轮分类时权值要变小
        expon = multiply(-1 * alpha * mat(classLabel).T, mat(bestClasEst))
        D = multiply(D, exp(expon))
        D = D / D.sum()

        aggClassEst += alpha * bestClasEst          # 计算前几轮分类器分类结果的加权和为最终分类结果
        errArr = ones((m, 1))
        errArr[sign(aggClassEst) == mat(classLabel).T] = 0
        errRate = errArr.sum() / m                  # 计算预测错误的概率
        print("training total error:", str(errRate))
        if (errRate == 0.0):                        # 如果预测错误率为0，则退出循环
            break
    return weakClassArr

def classfy(dataToClass, weakClassArr):
    """
    使用多个弱分类器通过ada boost集成强分类器来对数据进行分类
    :param dataToClass: 需要分类的特征数据
    :param weakClassArr: 弱分类器组合
    :return class: 分类结果
    """
    weakClassLen = len(weakClassArr)
    dataMat = mat(dataToClass)
    m = shape(dataMat)[0]
    aggClass = mat(zeros((m, 1)))
    for i in range(weakClassLen):
        predictVal = stumpClassify(dataMat, weakClassArr[i]['dimen'],
                                   weakClassArr[i]['threshVal'], weakClassArr[i]['threshIneq'])
        aggClass += weakClassArr[i]['alpha'] * predictVal
    return aggClass

if __name__ == '__main__':
    dataArr, classLabel = loadDataSet('horseColicTraining2.txt')
    print("dataArr:", str(dataArr), "\nclassLable:", str(classLabel))
    weakClassArr = adaBoostTrain(dataArr, classLabel, 100)
    print("weakClassArr:", str(weakClassArr))

    testDataArr, testLabel = loadDataSet('horseColicTest2.txt')
    aggClass = classfy(testDataArr, weakClassArr)
    print("aggClass:", sign(aggClass.T))
    m = shape(testDataArr)[0]
    errArr = multiply(sign(aggClass) != mat(testLabel).T, ones((m, 1)))
    errRate = errArr.sum() / m
    print("test total error rate:", str(errRate))