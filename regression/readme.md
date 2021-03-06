﻿# 回归预测

1. 普通最小二乘法

   设yi = xi.T * W	W为系数列向量<br>	即Y(m×1) = X(m×n) * W(n×1)
   m代表有m组数据，n代表每组数据xi由n个特征<br> 平方误差:	m∑i (yi - xi.T *
   w)^2<br> 求导等于0：	w = (X.T * X) ^ (-1) * X.T * y X为矩阵<br>
   计算出w的值即求出了最佳回归系数。

2. 局部加权线性回归

    局部加权线性回归（locally weighted linear regression），我们给预测的点附近的点赋予更大些的权重回归的最佳系数w计算公式如下:<br> 
    w = (X.T*W*X)^(-1) * X.T*W*Y<br>
    W是一个对角矩阵，对角元素对应每个数据点的权重W的计算公式如下:<br>
    使用高斯核函数w(i,i) = e^(|x(i) - x| / (-2(k^2)))<br>
    高斯核函数能够使点附近的点得到更大的权重。

3. 通过缩减系数来理解数据

    数据标准化：
    数据的标准化(normalization)是将数据按比例缩放，使之落入一个小的特定区间。在某些比较和评价的指标处理中经常会用到，去除数据的单位限制，将其转化为无量纲的纯数值。
    在选择岭回归和Lasso时候，标准化是必须的。原因是正则化是有偏估计，会对权重进行惩罚。在量纲不同的情况，正则化会带来更大的偏差。<br>
    数据标准化公式：<br> 
    Y = Y - mean(Y)	mean代表数据的平均值<br> 
    X = (X - mean(X)) / var(X)	var代表数据的方差

    3.1 岭回归<br> 岭回归（ridge regression）<br>
    回归系数w的计算公式如下：<br> w = (X.T * X + λ*I) ^ (-1) * X.T * y
    I为单位矩阵<br>
    通过引入λ惩罚项，能够减少不重要的参数，这个技术在统计学中也叫做缩减（shrinkage）,通过迭代找到预测误差最小时的λ即可求出最优回归系数。迭代的过程中λ应该已指数级别进行增长。
        
    3.2 前向逐步线性回归
    前向逐步线性回归算法属于一种贪心算法，即每一步都尽可能减少误差。我们计算回归系数，不再是通过公式计算，而是通过每次微调各个回归系数，然后计算预测误差。那个使误差最小的一组回归系数，就是我们需要的最佳回归系数。
    
    3.3 由于缩减系数用到了数据标准化，所以需要进行数据还原<br>
    还原公式推算如下：<br> 
    设xMat'为xMat标准化后的数据,yHat'为yMat标准化后的数据 则：<br> 
    yHat' = xMat' * weights<br> 
    且xMat' = (xMat - xMean) / xVar<br> 
    带入上式得： yMat' = (xMat - xMean) * (weights / xVar)<br> 
    且yHat' = yMat - yMean 带入上式得：<br> 
    yMat = -xMean * (weights / xVar) + yMean + xMat * (weights / xVar)<br>
    其中常数部分为-xMean * (weights / xVar) + yMean<br> 
    系数部分为weights / xVar
