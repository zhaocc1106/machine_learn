#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
利用SVD算法提升推荐系统的效率

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/7/22 16:54
"""
from numpy import *

def loadData():
    data = mat([[4, 4, 0, 2, 2],
                [4, 0, 0, 3, 3],
                [4, 0, 0, 1, 1],
                [1, 1, 1, 2, 0],
                [2, 2, 2, 0, 0],
                [1, 1, 1, 0, 0],
                [5, 5, 5, 0, 0]])
    return data


def loadData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def euclidSim(inA, inB):
    """使用欧氏距离计算计算向量之间的相似度

    Args:
        inA: 向量A
        inB: 向量B

    Returns:
        向量A与B的相似度
    """
    return 1.0 / (1.0 + linalg.norm(inA - inB))                 # norm函数求解矩阵的范数


def pearsSim(inA, inB):
    """利用皮尔逊相关系数计算向量之间的相似度

    Args:
        inA: 向量A
        inB: 向量B

    Returns:
        向量A与B的相似度
    """
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0, 1]       # corrcoef结果在-1到+1之间，通过线性变化将范围变化到0到1之间


def cosSim(inA, inB):
    """利用余弦函数计算向量之间的相似度

    Args:
        inA: 向量A
        inB: 向量B

    Returns:
        相似度
    """
    # cos(alpha) = (A.T * B) / (||A|| + ||B||)
    num = inA.T * inB                                           # 计算矩阵的点积，即向量的内积
    normMult = linalg.norm(inA) * linalg.norm(inB)              # 计算A与B的范数乘积
    return (0.5 + 0.5 * (num / normMult)).A[0, 0]               # 将范围变化到0到1之间


def standRate(dataMat, userId, simMean, itemId):
    """标准打分函数

    使用原始数据集，为某用户未打分的项进行预测和打分

    Args:
        dataMat: 原始数据集
        userId: 用户ID
        simMean: 近似度计算函数
        itemId: 物品的ID

    Returns:
        打分引擎为userID用户对itemID物品预测的打分
    """
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[userId, j]
        if userRating == 0.0:
            continue                                            # 忽略用户未打分的物品项
        overLap = nonzero(logical_and(
            dataMat[:, itemId].A > 0, dataMat[:, j].A > 0))[0]  # 寻找出所有用户，这些用户对itemId和j物品都进行了打分
        # print("overLap:", str(overLap))
        if len(overLap) == 0:
            print("overLap is null, similarity is 0")
            similarity = 0
        else:
            similarity = simMean(dataMat[overLap, itemId],
                                 dataMat[overLap, j])           # 计算物品itemId与物品j的相似度
        simTotal += similarity                                  # 计算相似度的总和
        print("The similarity of [", j, itemId, "] is", similarity)
        ratSimTotal += userRating * similarity                  # 计算基于相似度的用户打分的总和
    if simTotal == 0:
        return 0.0                                              # 没有任何物品与itemId物品有相似
    else:
        return ratSimTotal / simTotal                           # 计算出用户打分


def recommend(dataMat, user, N = 3, simMean = cosSim, ratMethod = standRate):
    """推荐引擎

    基于所有的打分结果，为用户user，推荐出N个可能打分最高的物品

    Args:
        dataMat: 打分矩阵
        user: 用户ID
        N: 推荐N个最佳物品
        simMean: 相似度计算函数
        ratMethod: 预测打分函数

    Returns:
        预测出N个用户打分最高的物品ID
    """
    unratedItems = nonzero(dataMat[user,: ].A == 0)[1]          # 找出所有用户还没进行打分的物品
    if len(unratedItems) == 0:
        print("no unrated items")
    print("unratedItems:", str(unratedItems))
    itemRates = []                                              # 记录所有预测的打分
    for i in unratedItems:
        rate = ratMethod(dataMat, user, simMean, i)             # 循环预测所有未打分的物品的得分
        print("[item: ", i, " rate: ", rate, "]")
        itemRates.append((i, rate))                             # 将预测出的得分与物品ID结合成元素存放到itemRates中
    print("itemRates:", str(itemRates))
    return sorted(itemRates, key = lambda jj: jj[1],
                  reverse=True)[:N]                             # 从itemRates中找出分数最高的N个元素


def svdRate(dataMat, userId, simMean, itemId):
    """通过svd简化大数据，并为userId用户对itemId物品进行预测打分

    Args:
        dataMat: 大数据矩阵
        userId: 用户Id
        simMean: 相似度计算函数
        itemId: 物品Id

    Returns:
        打分引擎为userID用户对itemID物品预测的打分
    """
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = linalg.svd(dataMat)
    Sig4 = mat(eye(4) * Sigma[: 4])                                 # 通过计算奇异值平方和的90%，得出4个奇异值平方和就可以满足
    # xformedItems = (dataMat.T * U[:, :4] * Sig4.I).T
    """
    # DataMat = U * Sigma * VT
    # U为左奇异矩阵，VT为右奇异矩阵，Sigma为奇异值对角矩阵
    # DataMat ≈ U[:, : R] * SigR * VT[:R, :]
    # 左奇异矩阵U能行降维，右奇异矩阵VT能进行列降维
    # UT[: R, :] * DataMat ≈ SigR * VT[:R, :]行降维到R * N
    # DataMat * V[:, :R] ≈ U[:, : R] * SigR列降维到M * R
    """
    xformedItems = Sig4 * VT[: 4,: ]                                # UT[: R, :] * DataMat ≈ SigR * VT[:R, :]
    # print("U:\n", str(U))
    # print("Sigma:\n", str(Sigma))
    # print("VT:\n", str(VT))
    # for i in range(4):
    #     sigmaReduced[i, i] = Sigma[i]
    # print("sigmaReduced:\n", str(sigmaReduced))                     # Sig4 == sigmaReduced： ndarray * 操作是元素积操作

    for j in range(n):
        userRating = dataMat[userId, j]
        if userRating == 0.0 or j == itemId:
            continue                                                # 忽略用户未打分的物品项
        similarity = simMean(xformedItems[:, itemId],
                             xformedItems[:, j])                         # 计算物品itemId与物品j的相似度
        simTotal += similarity                                      # 计算相似度的总和
        print("The similarity of [", j, itemId, "] is", similarity)
        ratSimTotal += userRating * similarity                      # 计算基于相似度的用户打分的总和
    if simTotal == 0:
        return 0.0                                                  # 没有任何物品与itemId物品有相似
    else:
        return ratSimTotal / simTotal                               # 计算出用户打分


if __name__ == '__main__':
    dataMat = mat(loadData2())
    print("dataMat: \n", str(dataMat))
    # simCos = cosSim(dataMat[:, 0], dataMat[:, 1])
    # simEuclid = euclidSim(dataMat[:, 0], dataMat[:, 1])
    # simPears = pearsSim(dataMat[:, 0], dataMat[:, 1])
    # print("simCos: ", str(simCos))
    # print("simEuclid: ", str(simEuclid))
    # print("simPears: ", str(simPears))
    #
    # recItems = recommend(dataMat, 1, 3)
    # print("recItems: ", str(recItems))
    #
    recItems = recommend(dataMat, 1, 3, cosSim, standRate)
    print("recItems: ", str(recItems))
    # svdRate(dataMat, 1, cosSim, 2)