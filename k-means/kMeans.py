from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    """
    从文件中获取需要分类的数据
    :param fileName: 数据文件名
    :return dataArr: 数据矩阵
    """
    dataArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataArr.append(list(map(float, lineArr)))
    return dataArr


def randCent(dataSet, k):
    """
    创建K个质心点，用于创建初始的质心点集
    :param dataSet: 数据集合
    :param k: 将数据聚类为四个族
    :return centroids: 质心点
    """
    dataMat = mat(dataSet)
    m, n = shape(dataMat)
    centroids = zeros((k, n))
    for j in range(n):
        minJ = zeros((k, 1))
        minJ[:, 0] = min(dataMat[:, j])
        maxJ = zeros((k, 1))
        maxJ[:, 0] = max(dataMat[:, j])
        rangeJ = maxJ - minJ
        centroids[:, j] = (minJ + multiply(random.rand(k, 1), rangeJ))[:, 0]  # 生成k行1列的随机数，使用随机数生成在数据范围之内的任意值
    return centroids


def euclDist(vectA, vectB):
    """
    计算向量A B之间的欧氏距离
    :param vectA: 向量A
    :param vectB: 向量B
    :return: 欧氏距离
    """
    return sqrt(sum(power(vectA - vectB, 2)))


def kMeans(dataSet, k, distFunc=euclDist, createCent=randCent):
    """
    通过k均值分类算法对数据进行分类，分成k个簇
    :param dataSet: 需要分类的数据
    :param k: 分成k个簇
    :param distFunc: 计算距离的公式
    :param createCent: 创建初始质心点的函数
    :return centroids: 分类完成的质心点集
    :return clusterAssment: 每个数据点的分类评估，包括分到哪个簇，距离该簇质心点的距离
    """
    dataMat = mat(dataSet)
    m, n = shape(dataMat)
    clusterAssment = mat(zeros((m, 2)))
    clusterAssment[:, 0] = -1
    centroids = createCent(dataSet, k)  # 创建初始的质心点集

    clusterChanged = True  # 作为迭代条件，每次分类完成后，如果簇安排发生了变化，则继续
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 对数据集中每个点，查找距离最短的簇
            distMin = inf
            minIndex = -1
            for j in range(k):
                dist = distFunc(centroids[j, :], dataMat[i, :])  # 计算数据点距离每个簇的质心点的距离
                if dist < distMin:
                    distMin = dist
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterAssment[i, 0] = minIndex
                clusterAssment[i, 1] = distMin ** 2
                clusterChanged = True  # 簇的安排发生了变化

        # print("centroids:", str(centroids))
        # 计算各个簇的新的质心点
        for i in range(k):
            ptsInClus = dataMat[nonzero(clusterAssment[:, 0] == i)[0], :]
            # print("ptsInClus:", str(ptsInClus))
            centroids[i, :] = mean(ptsInClus, axis=0)
    return centroids, clusterAssment


def showData(dataSet, centroids, clusterAssment, k):
    """
    将数据的分类情况展示出来
    :param dataSet: 数据集
    :param centroids: 各个簇的质心点
    :param clusterAssment: 每个数据点分类情况
    :param k 个簇
    :return:
    """
    makers = ['.', ',', 'o', 'v', '^', '<', '>', \
              '1', '2', '3', '4', '8', 's', 'p', \
              '*', 'h', 'H', 'x', 'D', 'd', '|']
    if k > len(makers):
        print("k overflow")  # 不同分类使用不同符号代表，这里限制了最大分类个数
        return

    fit = plt.figure()
    ax = fit.add_subplot(111)

    # 用+号画出所有簇的质心点
    ax.scatter(centroids[:, 0].T.tolist(), centroids[:, 1].T.tolist(), c='r', marker='+', s=200)

    # 按照簇分类画出所有数据点
    dataMat = mat(dataSet)
    for i in range(k):
        ptsInClus = dataMat[nonzero(clusterAssment[:, 0] == i)[0], :]  # 找出该簇的所有点
        ax.scatter(ptsInClus[:, 0].T.tolist()[0], ptsInClus[:, 1].T.tolist()[0], c='b', alpha='0.5', \
                   marker=makers[i])  # 用特定的形状绘制出该簇所有点
    plt.show()

def biKmeans1(dataSet, k, distFunc=euclDist, createCent=randCent):
    """
    二分K-均值聚类算法，能够克服K-均值算法收敛于局部最小值的问题。通过实际测试，能够发现普通的k-均值有时分类的结果并不令人满意
    :param dataSet: 数据集合
    :param k: k个簇
    :param distFunc: 距离计算公式
    :param createCent: 创建初始质心点集的函数
    :return centroids: 分类完成的质心点集
    :return clusterAssment: 每个数据点的分类评估，包括分到哪个簇，距离该簇质心点的距离
    """
    dataMat = mat(dataSet)
    m, n = shape(dataMat)
    clusterAssment = zeros((m, 2))
    centroid0 = mean(dataMat, axis=0).tolist()[0]                               # 初始化的质心点是整个数据集的均值

    centroids = [centroid0]  # 初始化质心点集
    for j in range(m):
        clusterAssment[j, 1] = distFunc(dataMat[j, :], centroid0) ** 2          # 初始化每个数据点距离质心点的距离
    print("Initial centroids:", str(centroids), \
          "\nInitial clusterAssment:", str(clusterAssment))

    while len(centroids) < k:
        print("#############################[%d] 2-means#############################" % (len(centroids)))
        lowestSSE = inf                                                         # 初始化最小SSE(Sum of Squared Error)误差平方和为无穷大
        for i in range(len(centroids)):
            ptsInClus = dataMat[nonzero(clusterAssment[:, 0] == i)[0], :]       # 找出该簇的所有点
            # print("i: ", i, " ptsInClus:", str(ptsInClus))
            splitCent, splitClustAss = kMeans(ptsInClus, 2, distFunc, createCent) # 对该簇尝试进行2-均值划分
            # print("splitCent:", str(splitCent), "\nsplitClustAss:", str(splitClustAss))
            sseSplit = sum(splitClustAss[:, 1])                                 # 对划分到该簇的数据点求SSE
            sseNotSplit =\
                sum(clusterAssment[nonzero(clusterAssment[:, 0] != i)[0], 1])   # 对不是该簇的数据求SSE
            print("i:", i)
            print("sseSplit:", str(sseSplit), " sseNotSplit:", str(sseNotSplit))
            print("sumSplit:", str(sseSplit + sseNotSplit))
            if sseSplit + sseNotSplit < lowestSSE:                              # 找到能够使SSE最小的划分方案
                lowestSSE = sseSplit + sseNotSplit
                bestSplitIndex = i
                bestSplitCent = splitCent.copy()
                bestSplitClustAss = splitClustAss.copy()

        print("lowestSSE:", lowestSSE, "\nbestSplitIndex:", str(bestSplitIndex),\
              "\nbestSplitCent:", str(bestSplitCent))

        centroids[bestSplitIndex] = bestSplitCent[0, :]                         # 将被重新划分的簇的质心点进行替换
        centroids.append(bestSplitCent[1, :])                                   # 添加一个新的质心点在最后

        indexs0 = nonzero(bestSplitClustAss[:, 0] == 0)[0]                      # 划分到0簇中的所有数据点index
        indexs1 = nonzero(bestSplitClustAss[:, 0] == 1)[0]                      # 划分到1簇中的所有数据点index
        bestSplitClustAss[indexs0, 0] = bestSplitIndex                          # 将新划分的0、1簇中的数据点的簇序号替换成外层簇的序号
        bestSplitClustAss[indexs1, 0] = len(centroids) - 1
        clusterAssment[nonzero(clusterAssment[:, 0]\
                               == bestSplitIndex)[0], :] = bestSplitClustAss    # 将被重新划分的数据点更新一下划分簇的序号
        # print("centroids: ", str(centroids), "\nclusterAssment:", str(clusterAssment))
    return array(centroids), clusterAssment

def biKmeans2(dataSet, k, distFunc=euclDist, createCent=randCent):
    """
    二分K-均值聚类算法，能够克服K-均值算法收敛于局部最小值的问题，与biKmean1的区别在于比较好的划分的标准不同。
    biKmean1是通过循环2划分每个簇，划分哪个簇后导致SSE最小，就是最佳的划分。
    biKmean2是判断哪个簇的当前SSE最大，就2划分哪个簇。
    通过测试biKmean2比biKmean1效果出错率低很多
    :param dataSet: 数据集合
    :param k: k个簇
    :param distFunc: 距离计算公式
    :param createCent: 创建初始质心点集的函数
    :return centroids: 分类完成的质心点集
    :return clusterAssment: 每个数据点的分类评估，包括分到哪个簇，距离该簇质心点的距离
    """
    dataMat = mat(dataSet)
    m, n = shape(dataMat)
    clusterAssment = zeros((m, 2))
    centroid0 = mean(dataMat, axis=0).tolist()[0]                               # 初始化的质心点是整个数据集的均值

    centroids = [centroid0]                                                     # 初始化质心点集
    for j in range(m):
        clusterAssment[j, 1] = distFunc(dataMat[j, :], centroid0) ** 2          # 初始化每个数据点距离质心点的距离
    print("Initial centroids:", str(centroids), \
          "\nInitial clusterAssment:", str(clusterAssment))

    while len(centroids) < k:
        print("#############################[%d] 2-means#############################" % (len(centroids)))
        maxSSE = 0
        for i in range(len(centroids)):
            sseI = sum(clusterAssment[nonzero(clusterAssment[:, 0] == i)[0], :])# 计算I簇的SSE
            print("sseI:", str(sseI), " maxSSE:", str(maxSSE))
            if sseI > maxSSE:
                maxSSE = sseI
                bestSplitIndex = i
        print("maxSSE:", maxSSE, "\nbestSplitIndex:", str(bestSplitIndex))      # 找出SSE最大的簇进行进一步划分
        ptsInClus = dataMat[nonzero(clusterAssment[:, 0] ==
                                    bestSplitIndex)[0], :]                      # 找出该簇的所有点
        bestSplitCent, bestSplitClustAss = kMeans(ptsInClus, 2, distFunc, createCent)

        centroids[bestSplitIndex] = bestSplitCent[0, :]                         # 将被重新划分的簇的质心点进行替换
        centroids.append(bestSplitCent[1, :])                                   # 添加一个新的质心点在最后

        indexs0 = nonzero(bestSplitClustAss[:, 0] == 0)[0]                      # 划分到0簇中的所有数据点index
        indexs1 = nonzero(bestSplitClustAss[:, 0] == 1)[0]                      # 划分到1簇中的所有数据点index
        bestSplitClustAss[indexs0, 0] = bestSplitIndex                          # 将新划分的0、1簇中的数据点的簇序号替换成外层簇的序号
        bestSplitClustAss[indexs1, 0] = len(centroids) - 1
        clusterAssment[nonzero(clusterAssment[:, 0]\
                               == bestSplitIndex)[0], :] = bestSplitClustAss    # 将被重新划分的数据点更新一下划分簇的序号
    return array(centroids), clusterAssment

if __name__ == "__main__":

    # 使用testSet测试普通kMeans
    # dataSet = loadDataSet('testSet2.txt')
    # print("dataSet:", str(dataSet))
    # centroids, clusterAssment = kMeans(dataSet, 3)
    # print("centroids:", str(centroids), '\nclusterAssment:', str(clusterAssment))
    # showData(dataSet, centroids, clusterAssment, 3)

    # 使用testSet2测试biKmeans
    # dataSet = loadDataSet('testSet2.txt')
    # print("dataSet:", str(dataSet))
    # centroids, clusterAssment = biKmeans1(dataSet, 3)
    # print("centroids:", str(centroids), '\nclusterAssment:', str(clusterAssment))
    # showData(dataSet, centroids, clusterAssment, 3)

    # 使用testSet2测试biKmeans2
    dataSet = loadDataSet('testSet2.txt')
    print("dataSet:", str(dataSet))
    centroids, clusterAssment = biKmeans2(dataSet, 3)
    print("centroids:", str(centroids), '\nclusterAssment:', str(clusterAssment))
    showData(dataSet, centroids, clusterAssment, 3)
    # 通过测试biKmean2比biKmean1效果出错率低很多

