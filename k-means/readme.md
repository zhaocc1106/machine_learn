# K-均值聚类算法
K-均值是发现给定数据集的k个簇的算法。簇个数k是用户给定的，每一个簇通过其质心（centroid），即簇中所有点的中心来描述。<br>
K-均值算法的工作流程是这样的。首先，随机确定k个初始点作为质心。然后将数据集中的每个点分配到一个簇中，具体来讲，为每个点找距其最近的质心，并将其分配给该质心所对应的
簇。这一步完成之后，每个簇的质心更新为该簇所有点的平均值。
上述过程的伪代码表示如下：<br>
```
创建k个点作为起始质心（经常是随机选择）
当任意一个点的簇分配结果发生改变时
	对数据集中的每个数据点
		对每个质心
			计算质心与数据点之间的距离
		将数据点分配到距其最近的簇
	对每一个簇，计算簇中所有点的均值并将均值作为质心
```

# 二分K-均值算法
为克服K-均值算法收敛于局部最小值的问题，有人提出了另一个称为二分K-均值（bisecting K-means）的算法。该算法首先将所有点作为一个簇，然后将该簇一分为二。之后选择其中一个簇继续进行划分，选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE的值。上述基于SSE的划分过程不断重复，直到得到用户指定的簇数目为止。<br>
二分K-均值算法的伪代码形式如下：<br>
```
将所有点看成一个簇
当簇数目小于k时
	对于每一个簇
		计算总误差
		在给定的簇上面进行K-均值聚类（k=2）
		计算将该簇一分为二之后的总误差
	选择使得误差最小的那个簇进行划分操作
```
另一种做法是选择SSE最大的簇进行划分，直到簇数目达到用户指定的数目为止。<br>
通过实际测试，第二种方法效果要比第一种好很多。