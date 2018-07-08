# 使用FP-growth算法来高效发现频繁项集
## 算法速度
FP-growth算法只需要对数据库进行两次扫描，而Apriori算法对于每个潜在的频繁项集都会扫描数据集判定给定模式是否频繁，因此FP-growth算法的速度要比Apriori算
法快。
## FP 树：用于编码数据集的有效方式
FP-growth算法将数据存储在一种称为FP树的紧凑数据结构中。FP代表频繁模式（FrequentPattern）。一棵FP树看上去与计算机科学中的其他树结构类似，但是它通过链
接（link）来连接相似元素，被连起来的元素项可以看成一个链表。下图给出了FP树的一个例子。<br>
![fpTree](https://github.com/zhaocc1106/machine_learn/blob/master/FP-growth/images/FP%E6%A0%91%E4%BE%8B%E5%AD%90.png)<br>
同搜索树不同的是，一个元素项可以在一棵FP树中出现多次。FP树会存储项集的出现频率，而每个项集会以路径的方式存储在树中。存在相似元素的集合会共享树的一部分。
只有当集合之间完全不同时，树才会分叉。 树节点上给出集合中的单个元素及其在序列中的出现次数，路径会给出该序列的出现次数。相似项之间的链接即节点链接
（node link），用于快速发现相似项的位置<br>。
FP-growth算法的工作流程如下。首先构建FP树，然后利用它来挖掘频繁项集。为构建FP树，需要对原始数据集扫描两遍。第一遍对所有元素项的出现次数进行计数。记住
Apriori原理，即如果某元素是不频繁的，那么包含该元素的超集也是不频繁的，所以就不需要考虑这些超集。数据库的第一遍扫描用来统计出现的频率，而第二遍扫描中
只考虑那些频繁元素。<br>
## 构建FP树
除了上图给出的FP树之外，还需要一个头指针表来指向给定类型的第一个实例。利用头指针表，可以快速访问FP树中一个给定类型的所有元素。下图给出了一个头指针表的
示意图。<br>
![headerTable](https://github.com/zhaocc1106/machine_learn/blob/master/FP-growth/images/FP%E6%A0%91%E5%8A%A0%E5%A4%B4%E6%8C%87%E9%92%88%E8%A1%A8.png)<br>
这里使用一个字典作为数据结构，来保存头指针表。除了存放指针外，头指针表还可以用来保存FP树中每类元素的总数。<br>
第一次遍历数据集会获得每个元素项的出现频率。接下来，去掉不满足最小支持度的元素项。再下一步构建FP树。在构建时，读入每个项集并将其添加到一条已经存在的路径
中。如果该路径不存在，则创建一条新路径。每个事务就是一个无序集合。假设有集合{z,x,y}和{y,z,r}，那么在FP树中，相同项会只表示一次。为了解决此问题，在将集
合添加到树之前，需要对每个集合进行排序。排序基于元素项的绝对出现频率来进行。<br>
在对事务记录过滤和排序之后，就可以构建FP树了。从空集开始，向其中不断添加频繁项集。过滤、排序后的事务依次添加到树中，如果树中已存在现有元素，则增加现有元
素的值；如果现有元素不存在，则向树添加一个分枝。例如下面例子：<br>
![](https://github.com/zhaocc1106/machine_learn/blob/master/FP-growth/images/fp%E6%A0%91%E7%94%9F%E9%95%BF%E4%BE%8B%E5%AD%90.png)
## 从一棵FP 树中挖掘频繁项集
从FP树中抽取频繁项集的三个基本步骤如下：<br>
(1) 从FP树中获得条件模式基；<br>
(2) 利用条件模式基，构建一个条件FP树；<br>
(3) 迭代重复步骤(1)步骤(2)，直到树包含一个元素项为止。<br>
首先从上一节发现的已经保存在头指针表中的单个频繁元素项开始。对于每一个元素项，获得其对应的条件模式基（conditional pattern base）。条件模式基是以所查找
元素项为结尾的路径集合。每一条路径其实都是一条前缀路径（prefix path）。简而言之，一条前缀路径是介于所查找元素项与树根节点之间的所有内容。<br>
可以想象出来，当找到某个节点的所有条件模式基（所有路径集合），使用这些数据集进行创建FP树，因为创建FP树时会将不满足minSupport条件的元素剔除，则剩下来的
元素组成headerTable，则之前节点作为前缀，后生成的headerTable作为后缀组成项集，这些都是频繁项集。