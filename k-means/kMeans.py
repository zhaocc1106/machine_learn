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

def euclDist(vectA, vectB):
    """
    计算向量A B之间的欧氏距离
    :param vectA: 向量A
    :param vectB: 向量B
    :return: 欧氏距离
    """
    return sqrt(sum(power(vectA - vectB, 2)))

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
        centroids[:, j] = (minJ + multiply(random.rand(k, 1), rangeJ))[:, 0]      # 生成k行1列的随机数，使用随机数生成在数据范围之内的任意值
    return centroids

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
    centroids = createCent(dataSet, k)                          # 创建初始的质心点集
    print("centroids:", str(centroids))

    clusterChanged = True                                       # 作为迭代条件，每次分类完成后，如果簇安排发生了变化，则继续
    while clusterChanged:
        clusterChanged = False
        for i in range(m):                                      # 对数据集中每个点，查找距离最短的簇
            distMin = inf
            minIndex = -1
            for j in range(k):
                dist = distFunc(centroids[j, :], dataMat[i, :]) # 计算数据点距离每个簇的质心点的距离
                if dist < distMin:
                    distMin = dist
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterAssment[i, 0] = minIndex
                clusterAssment[i, 1] = distMin
                clusterChanged = True                           # 簇的安排发生了变化

        # 计算各个簇的新的质心点
        for i in range(k):
            ptsInClus = dataMat[nonzero(clusterAssment[:, 0]==i)[0], :]
            print("ptsInClus:", str(ptsInClus))
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
        print("k overflow")                                                 # 不同分类使用不同符号代表，这里限制了最大分类个数
        return

    fit = plt.figure()
    ax = fit.add_subplot(111)

    # 用+号画出所有簇的质心点
    ax.scatter(centroids[:, 0].T.tolist(), centroids[:, 1].T.tolist(), c='r', marker='+', s=200)

    # 按照簇分类画出所有数据点
    dataMat = mat(dataSet)
    for i in range(k):
        ptsInClus = dataMat[nonzero(clusterAssment[:, 0] == i)[0], :]       # 找出该簇的所有点
        ax.scatter(ptsInClus[:, 0].T.tolist()[0], ptsInClus[:, 1].T.tolist()[0], c='b', alpha='0.5', \
                   marker=makers[i])                                        # 用特定的形状绘制出该簇所有点
    plt.show()

if __name__ == "__main__":
    dataSet = loadDataSet('testSet.txt')
    print("dataSet:", str(dataSet))
    centroids, clusterAssment = kMeans(dataSet, 4)
    print("centroids:", str(centroids), '\nclusterAssment:', str(clusterAssment))
    showData(dataSet, centroids, clusterAssment, 4)