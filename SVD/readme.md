# 利用SVD简化数据
## 回顾特征值和特征向量
　　我们首先回顾下特征值和特征向量的定义如下：<br>
　　　　　　　　　　　　　　　　　　　　　　　　　　Ax=λx<br>
　　其中A是一个n×n的矩阵，x是一个n维向量，则我们说λ是矩阵A的一个特征值，而x是矩阵A的特征值λ所对应的特征向量。<br>
　　求出特征值和特征向量有什么好处呢？ 就是我们可以将矩阵A特征分解。如果我们求出了矩阵A的n个特征值λ1≤λ2≤...≤λn,以及这n个特征值所对应的特征向量{w1,w2,...wn}，那么矩阵A就可以用下式的特征分解表示：<br>
　　　　　　　　　　　　　　　　　　　　　　　　　　A=WΣW−1<br>
　　其中W是这n个特征向量所张成的n×n维矩阵，而Σ为这n个特征值为主对角线的n×n维矩阵。<br>
　　一般我们会把实对称矩阵W的这n个特征向量标准化，即满足||wi||2=1, 或者说wTiwi=1，此时W的n个特征向量为标准正交基，满足WTW=I，即WT=W−1, 也就是说W为酉矩阵。<br>
　　这样我们的特征分解表达式可以写成<br>
　　　　　　　　　　　　　　　　　　　　　　　　　　A=WΣWT<br>
　　注意到要进行特征分解，矩阵A必须为方阵。那么如果A不是方阵，即行和列不相同时，我们还可以对矩阵进行分解吗？答案是可以，此时我们的SVD登场了。<br>
## SVD奇异值分解
　　奇异值分解(Singular Value Decomposition，以下简称SVD)是在机器学习领域广泛应用的算法，它不光可以用于降维算法中的特征分解，还可以用于推荐系统，以及自然语言处理等领域。是很多机器学习算法的基石。<br>
  　　SVD也是对矩阵进行分解，但是和特征分解不同，SVD并不要求要分解的矩阵为方阵。假设我们的矩阵A是一个m×n的矩阵，那么我们定义矩阵A的SVD为：
A=UΣVT
　　其中U是一个m×m的矩阵，Σ是一个m×n的矩阵，除了主对角线上的元素以外全为0，主对角线上的每个元素都称为奇异值，V是一个n×n的矩阵。U和V都是酉矩阵，即满足UTU=I,VTV=I<br>
## SVD的性质
　　对于奇异值,它跟我们特征分解中的特征值类似，在奇异值矩阵中也是按照从大到小排列，而且奇异值的减少特别的快，在很多情况下，前10%甚至1%的奇异值的和就占了全部的奇异值之和的99%以上的比例。也就是说，我们也可以用最大的k个的奇异值和对应的左右奇异向量来近似描述矩阵。也就是说：
Am×n=Um×mΣm×nVTn×n≈Um×kΣk×kVTk×n
## SVD用于降维
　　DataMat = U * Sigma * VT<br>
　　U为左奇异矩阵，VT为右奇异矩阵，Sigma为奇异值对角矩阵<br>
　　DataMat ≈ U[:, : R] * SigR * VT[:R, :]<br>
　　左奇异矩阵U能行降维，右奇异矩阵VT能进行列降维<br>
　　UT[: R, :] * DataMat ≈ SigR * VT[:R, :]行降维到R * N<br>
　　DataMat * V[:, :R] ≈ U[:, : R] * SigR列降维到M * R<br>
## 推荐系统
### 基于协同过滤的推荐引擎
　　近十年来，推荐引擎对因特网用户而言已经不是什么新鲜事物了。Amazon会根据顾客的购买历史向他们推荐物品，Netflix会向其用户推荐电影，新闻网站会对用户推荐新闻报道，这样的
例子还有很多很多。当然，有很多方法可以实现推荐功能，这里我们只使用一种称为协同过滤（collaborative filtering）的方法。协同过滤是通过将用户和其他用户的数据进行对比来实现推
荐的。<br>
　　这里的数据是从概念上组织成了类似图14-2所给出的矩阵形式。当数据采用这种方式进行组织时，我们就可以比较用户或物品之间的相似度了。这两种做法都会使用我们很快就介绍到的相
似度的概念。当知道了两个用户或两个物品之间的相似度，我们就可以利用已有的数据来预测未知的用户喜好。例如，我们试图对某个用户喜欢的电影进行预测，推荐引擎会发现有一部电影该
用户还没看过。然后，它就会计算该电影和用户看过的电影之间的相似度，如果其相似度很高，推荐算法就会认为用户喜欢这部电影。<br>
　　在上述场景下，唯一所需要的数学方法就是相似度的计算，这并不是很难。接下来，我们首先讨论物品之间的相似度计算，然后讨论在基于物品和基于用户的相似度计算之间的折中。最后，
我们介绍推荐引擎成功的度量方法。<br>
### 相似度计算
>欧氏距离计算相似度:<br>
相似度=1/(1+欧氏距离)<br>
皮尔逊相关系数计算相似度:<br>
相似度=0.5 + 0.5*corrcoef()<br>
余弦相似度:<br>
相似度=(A*B)/(||A||*||B||)<br>
### 利用SVD 提高推荐的效果
　　通过行降维将大数据缩减<br>
## 图像压缩
　　通过Am×n=Um×mΣm×nVTn×n≈Um×kΣk×kVTk×n的性质，将原有ｍ×ｎ的像素缩减成m*r*2+r*r的像素来存储<br>