# 支持向量机算法

## 目录架构
* plots -- svm_scikit.py生成的plots
* SVM.py -- 使用numpy实现svm算法
* svm_scikit.py -- 使用scikit-learn的svm算法测试分类效果
* svm_tf.py -- 使用tensorflow的svm算法测试分类效果，目前只支持线性分类
* testSet.txt -- 线性可分的数据集，使用线性svm进行分类
* testSetRBF.txt -- 非线性可分的训练数据集，使用非线性svm进行训练
* testSetRBF2.txt -- 非线性可分的测试数据集，用于测试testSetRBF训练好的非线性svm

## svm算法描述
详细算法的讲解和推导见李航-《统计学习方法》第7章。<br>
下边简要介绍：
```
【SMO算法的步骤】
    SMO算法能够通过随机两个点不断地对alphas和b进行调整，直到调整到最佳值。
    * 步骤1：计算误差：
        fx = wx + b计算fx的值
        拉格朗日乘子法优化可以得出w的值为(1~n)∑i alphai*yi*xi
        fx = (1~n)∑i alphai*yi*xi * x + b => fx = (1~n)∑i alphai*yi*<xi, x> + b
        i点误差：Ei  = fXi - labelMat[i]
        j点误差：Ej  = fXj - labelMat[j]
    * 步骤2：计算上下界L和H：
        如果labelI != labelJ
            L = max(0, alphaJold - alphaIold)
                H = min(C, C + alphaJold - alphaIold)
        如果labelI == labelJ
            L = max(0, alphaJold + alphaIold - C)
            H = min(C, alphaJold + alphaIold)
    * 步骤3：计算η：
        η = 2 * xi * xj.T - xi * xi.T - xj * xj.T 
    * 步骤4：更新αj： 
        alphaJnew = alphaJold - yj(Ei - Ej) / η
    * 步骤5：根据取值范围修剪αj：
        if aj > H:
                alphaJnewClipped = H
            if aj < L:
                alphaJnewClipped = L
        else:
            alphaJnewClipped = alphaJnew
    * 步骤6：更新αi： 
        alphaInew = alphaIold + yi*yj*(alphaJold - alphaJnewClipped)
    * 步骤7：更新b1和b2： 
        b1New = bOld - Ei - yj * (alphaInew - alphaIold) * xi.T * xi - yj * (alphaJnew - alphaJold) * xj.T *xi
        b2New = bOld - Ej - yi * (alphaInew - alphaIold) * xi.T * xj - yj * (alphaJnew - alphaJold) * xj.T *xj
    * 步骤8：根据b1和b2更新b： 
        if (0 < alphas[i] and alphas[i] < C):
                b = b1
            elif (0 < alphas[j] and alphas[j] < C):
                    b = b2
            else:
                    b = (b1 + b2) / 2.0


【Platt SMO完整算法步骤】
    *在为i点寻找j点时从随机改成如下处理：
        为i点寻找j点，找到的j点的alphaJnew是变化最大的点，加大alpha的变化步长，继而加快alphas和b的调整速度
            由于alphaJnew = alphaJold - yj(Ei - Ej) / η ，并且η是常量，所以就是求Ei - Ej值最大时的j点
    *外部循环
        使用完整集合和非边界集合（0 < alpha < C）交替进行调整alphas和b


【非线性SVM】
    *核技巧
        我们已经了解到，SVM如何处理线性可分的情况，而对于非线性的情况，SVM的处理方式就是选择一个核函数。简而言之：在线性不可分的情况下，
        SVM通过某种事先选择的非线性映射（核函数）将输入变量映到一个高维特征空间，将其变成在高维空间线性可分，在这个高维空间中构造最
        优分类超平面。
        线性可分的情况下，可知最终的超平面方程为：
            fx = (1~n)∑i alphai*yi*xi * x + b
        将上述公式用内积来表示：
            fx = (1~n)∑i alphai*yi*<xi, x> + b
        对于线性不可分，我们使用一个非线性映射，将数据映射到特征空间，在特征空间中使用线性学习器，分类函数变形如下：
            fx = (1~n)∑i alphai*yi*<(xi), (x)> + b
        其中从输入空间(X)到某个特征空间(F)的映射，这意味着建立非线性学习器分为两步：
            首先使用一个非线性映射将数据变换到一个特征空间F；
            然后在特征空间使用线性学习器分类。
        如果有一种方法可以在特征空间中直接计算内积<(x_i),(x)>，就像在原始输入点的函数中一样，就有可能将两个步骤融合到一起建立一个分线性
        的学习器，这样直接计算的方法称为核函数方法。
        这里直接给出一个定义：核是一个函数k，对所有x,z∈X，满足k(x,z)=<(x_i),(x)> ，这里(·) 是从原始输入空间X到内积空间F的映射。
        简而言之：如果不是用核技术，就会先计算线性映(x_1) 和(x_2)，然后计算这它们的内积，使用了核技术之后，先把(x_1)和(x_2) 的一般表达
        式<(x_1),(x_2)>=k(<(x_1),(x_2) >) 计算出来，这里的 <·，·> 表示内积，k(·，·) 就是对应的核函数，这个表达式往往非常简单，所以
        计算非常方便。这种将内积替换成核函数的方式被称为核技巧(kernel trick)。
    *最流行的核函数径向基核函数
        K(x, y) = exp(-1 * (||x -y||^2 / 2 * σ ^ 2))，其中x 与 y是向量，σ为确定到达率或者说是函数值跌落到0的速度参数
```