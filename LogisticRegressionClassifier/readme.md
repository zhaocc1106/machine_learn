# 逻辑回归分类算法

## 基于Logistic 回归和Sigmoid 函数的分类

我们想要的函数应该是，能接受所有的输入然后预测出类别。例如，在两个类的情况下，上述函数输出0或1。或许你之前接触过具有这种性质的函数，该函数称为海维塞德阶跃函数
（Heaviside step function），或者直接称为单位阶跃函数。然而，海维塞德阶跃函数的问题在于：该函数在跳跃点上从0瞬间跳跃到1，这个瞬间跳跃过程有时很难处理。幸好，另一
个函数也有类似的性质，且数学上更易处理，这就是Sigmoid函数。Sigmoid函数具体的计算公式如下：<br>
    δ(z) = 1 / (1 + e^(-z))<br>
当x为0时，Sigmoid函数值为0.5。随着x的增大，对应的Sigmoid值将逼近于1；而随着x的减小，Sigmoid值将逼近于0。如果横坐标刻度足够大，Sigmoid函数看起来很像一个阶跃函
数。为了实现Logistic回归分类器，我们可以在每个特征上都乘以一个回归系数，然后把所有的结果值相加，将这个总和代入Sigmoid函数中，进而得到一个范围在0~1之间的数值。任
何大于0.5的数据被分入1类，小于0.5即被归入0类。所以，Logistic回归也可以被看成是一种概率估计。确定了分类器的函数形式之后，现在的问题变成了：最佳回归系数是多少?

## 基于最优化方法的最佳回归系数确定

Sigmoid函数的输入记为z，由下面公式得出：<br>
    z = w0x0 + w1x1 + w2x2 +...+wnxn <br> 
如果采用向量的写法，上述公式可以写成z=wTx，它表示将这两个数值向量对应元素相乘然后全部加起来即得到z值。其中的向量x是分类器的输入数据，向量w也就是我们要找到的最佳
参数（系数），从而使得分类器尽可能地精确。为了寻找该最佳参数，需要用到最优化理论的一些知识。

## 通过极大似然估计推导梯度函数
借用李航博士写的《统计学习方法》中6.1节内容：<br>
![reference01](http://icode.baidu.com/repos/baidu/hec-system-sw/Machine-Learning/blob/master:LogisticRegressionClassifier/images/reference01.png)<br>
![reference02](http://icode.baidu.com/repos/baidu/hec-system-sw/Machine-Learning/blob/master:LogisticRegressionClassifier/images/reference02.png)<br>
由上述书籍的详细推导，得出我们的对数似然函数如下:<br>
    L(w) = (i:1~N)∑ [yi(w・xi) - log(1+exp(w・xi))]　　　　　其中xi,yi为第i个样本。<br>
 对L(w)求梯度▽，即对所有的w求导数，能够得到：<br>
    ▽L(w) = XT ・ (Y真 - Y预测)　　　　　其中XT 为N个xi组成的矩阵的转置，“Y真”为所有yi样本组成的列向量，“Y预测”为所有预测的yi样本组成的列向量。<br>

## 梯度上升法
梯度上升法基于的思想是：要找到某函数的最大值，最好的方法是沿着该函数的梯度方向探寻。
梯度上升算法沿梯度方向移动了一步。可以看到，梯度算子总是指向函数值增长最快的方向。这里所说的是移动方向，而未提到移动量的大小。该量值称为步长，记做α。用向
量来表示的话，梯度上升算法的迭代公式如下：<br>
	w := w + a▽wf(w)<br>
该公式将一直被迭代执行，直至达到某个停止条件为止，比如迭代次数达到某个指定值或算法达到某个可以允许的误差范围。

## 梯度下降法
其实和梯度上升法本质上一样，只要把上述对数似然函数取反，当作loss函数进行梯度下降即可。

## tensorflow 的linearClassifier
如下代码块是tf.estimator.linearClassifier的实现：
```Python
if n_classes == 2: 
    head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access      
    weight_column=weight_column,      
    label_vocabulary=label_vocabulary,      
    loss_reduction=loss_reduction)else:  
    head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(  # pylint: disable=protected-access      
    n_classes, 
    weight_column=weight_column,      
    label_vocabulary=label_vocabulary,      
    loss_reduction=loss_reduction)
```
能看到，tensorflow中的linearClassifier如果是二分类问题，则使用logistic + sigmoid激活函数 + cross_entropy方式来实现。如果是大于2类的分类问题，则使用
softmax + cross_entroy的方式实现。softmax，我个人认为是李航博士写的《统计学习方法》中6.2节内容所讲的最大熵模型，无论从模型和对数似然函数的表达式来看同
softmax模型和cross_entroy基本上一致，详细推导过程见书籍详细内容。

## 拓展一下
最优化算法除了梯度下降法，还可以使用牛顿法，和拟牛顿法。牛顿法和拟牛顿法具体详细介绍见下面的连接(https://blog.csdn.net/itplus/article/details/21896453)
，或者《统计学习方法》一书中附录B。但是牛顿法和拟牛顿法都涉及到了海塞矩阵的计算或近似计算，计算量非常大，目前我比较少见到使用牛顿法和拟牛顿法的，tensorflow
中使用的是梯度下降法以及有梯度下降法衍生的加快和与优化后相关算法，例如SGD、Adagrad、Adam、Ftrl、RMSProp等。
