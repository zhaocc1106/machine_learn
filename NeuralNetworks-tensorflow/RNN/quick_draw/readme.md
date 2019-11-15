# 你画我猜模型

## 模型
使用到bi_rnn模型，主要是用于处理玩家画图时笔画的时序问题。

## 准备数据
使用开放的[quickdraw](https://github.com/googlecreativelab/quickdraw-dataset)数据库。
1. 已经转换为tfRecords的数据<br>
http://download.tensorflow.org/data/quickdraw_tutorial_dataset_v1.tar.gz（大约 
1GB）。<br>
解压到\tmp\quickdraw_data\目录下。
2. 原始数据<br>
使用 gsutil 下载整个数据集。请注意，原始 .ndjson 文件需要下载约 22GB 的数据。<br>
然后，使用以下命令检查 gsutil 安装是否成功以及您是否可以访问数据存储分区：<br>
``
gsutil ls -r "gs://quickdraw_dataset/full/simplified/*"
``<br>
系统会输出一长串文件，如下所示：<br>
``
gs://quickdraw_dataset/full/simplified/The Eiffel Tower.ndjson
gs://quickdraw_dataset/full/simplified/The Great Wall of China.ndjson
gs://quickdraw_dataset/full/simplified/The Mona Lisa.ndjson
gs://quickdraw_dataset/full/simplified/aircraft carrier.ndjson
...
``<br>
之后，创建一个文件夹并在其中下载数据集。<br>
``
mkdir rnn_tutorial_data
cd rnn_tutorial_data
gsutil -m cp "gs://quickdraw_dataset/full/simplified/*" .
``<br>
下载过程需要花费一段时间，且下载的数据量略超 23GB。<br>
最后运行create_dataset_for_classify.py将原始数据转换为tfRecords文件，可以添加参数指定目录。

## 训练模型
将tfRecords文件放到/tmp/quickdraw_data/目录下后，可以直接运行quick_draw_classify.py
进行训练，默认训练100w步，最后生成训练结果图片，如quick_draw_classify.png。

## 训练结果
![quick_draw_classify](https://github.com/zhaocc1106/machine_learn/blob/master/NeuralNetworks-tensorflow/RNN/quick_draw/quick_draw_classify/quick_draw_classify.png)

# 自动涂鸦模型

## 模型
使用到lstm/gru模型，主要学习每种类型涂鸦的笔画

## 准备数据
1. 下载原始数据<br>
使用 gsutil 下载整个数据集。请注意，原始 .ndjson 文件需要下载约 22GB 的数据。<br>
然后，使用以下命令检查 gsutil 安装是否成功以及您是否可以访问数据存储分区：<br>
``
gsutil ls -r "gs://quickdraw_dataset/full/simplified/*"
``<br>
系统会输出一长串文件，如下所示：<br>
``
gs://quickdraw_dataset/full/simplified/The Eiffel Tower.ndjson
gs://quickdraw_dataset/full/simplified/The Great Wall of China.ndjson
gs://quickdraw_dataset/full/simplified/The Mona Lisa.ndjson
gs://quickdraw_dataset/full/simplified/aircraft carrier.ndjson
...
``<br>
之后，创建一个文件夹并在其中下载数据集。<br>
``
mkdir rnn_tutorial_data
cd rnn_tutorial_data
gsutil -m cp "gs://quickdraw_dataset/full/simplified/*" .
``<br>
下载过程需要花费一段时间，且下载的数据量略超 23GB。<br>
2. 为每一类涂鸦生成tfRecord文件
最后运行create_dataset_for_auto_draw.py将原始数据转换为tfRecords文件，可以添加参数指定目录。

## 训练模型
将tfRecords文件放到/tmp/autodraw_data/目录下后，可以直接运行auto_draw.py，也可以指定涂鸦的类型，
默认训练校车类型涂鸦。最后生成训练结果图片，如autodraw_school_bus.png，autodraw_cloud.png，
autodraw_bicycle.png，右侧为根据左侧初始10笔画生成预测的涂鸦。

## 训练结果
![auto_draw_school_bus](https://github.com/zhaocc1106/machine_learn/blob/master/NeuralNetworks-tensorflow/RNN/quick_draw/auto_draw/autodraw_school_bus.png)
