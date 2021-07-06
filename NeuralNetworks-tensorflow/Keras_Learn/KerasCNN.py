# _*_ coding:utf-8 _*_
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
The use of keras CNN.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/9/5 19:03
"""

# common libs.
import os
import shutil
import time

# 3rd-part libs.
import numpy as np
from numpy import *
import tensorflow as tf
import matplotlib.pyplot as plt

MODEL_PATH = "/tmp/kerasCNN/"
CKPT_PATH = "/tmp/kerasCNN/checkpoint"
SAVER_PATH = "/tmp/kerasCNN/model_saver"
TRT_PATH = "/tmp/kerasCNN/trt_model"

# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# print(os.environ['LD_LIBRARY_PATH'])

def create_network(mirrored_strategy, num_class):
    """创建网络模型

    Args:
        mirrored_strategy: 分布式计算策略
        num_class: 类型的个数

    Returns:
        创建好的网络模型
    """
    with mirrored_strategy.scope():
        # 创建 keras 序贯模型
        model = tf.keras.models.Sequential()
        # 添加一个卷积层, 32个卷积核，激活函数用relu
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                                         input_shape=[28, 28, 1]
                                         ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        # 添加一个max pool层
        model.add(
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # 添加第二个卷积层
        model.add(tf.keras.layers.Conv2D(64, (5, 5)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        # 添加第二个max pool层
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # 添加flatten层
        model.add(tf.keras.layers.Flatten())
        # 添加完全连接层，1000个nn，使用relu激活函数
        model.add(tf.keras.layers.Dense(1000, activation="relu"))
        # 添加完全连接层作为输出层，分成10个类
        model.add(tf.keras.layers.Dense(num_class, activation="softmax"))
        # 使用交叉熵损失函数，使用Adam优化器
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
    return model


def train_keras_cnn(trainX, trainY, testX, testY):
    """使用keras构建CNN训练手写识别系统

    Args:
        trainX: 训练输入数据
        trainY: 训练标签数据
        testX: 测试输入数据
        testY: 测试标签数据

    Returns:
        errRate: 错误率
    """
    # 每个gpu上分配的batch大小
    batch_size_per_replica = 100
    epochs = 10
    # 手写识别有10个数字
    num_class = 10

    # 将类型label转换为二进制表示，比如1转换为(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    trainY = tf.keras.utils.to_categorical(trainY, num_class)
    testY = tf.keras.utils.to_categorical(testY, num_class)

    # 创建分布式计算策略，选择镜像分布式策略，即将batch平均分成N（gpu的个数）份。
    # 实测需要把tensorflow-gpu版本升级到tf-nightly-gpu(1.14)，使用pip install
    # tf-nightly-gpu命令。
    # 最终版本为Python(3.6.8) + tf-nightly-gpu(1.14.1-dev20190617)
    mirrored_strategy = tf.distribute.MirroredStrategy()

    # 计算所有gpu上batch size总大小
    batch_size_total = batch_size_per_replica * mirrored_strategy.num_replicas_in_sync
    print("batch_size_total: ", str(batch_size_total))

    # 创建网络模型
    model = create_network(mirrored_strategy, num_class)

    # 定义学习速率配置函数
    def learning_rate_scheduler(epoch):
        """The learning rate scheduler function.

        Args:
            epoch: Current epoch.

        Returns:
            The learning rate.
        """
        print("learning_rate_scheduler epoch:", epoch)
        return 1e-3

    # 定义model fit过程的回调
    callbacks = [
        # Model saver callback.
        tf.keras.callbacks.ModelCheckpoint(filepath=CKPT_PATH),
        # Learning rate scheduler.
        tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler,
                                                 verbose=1),
        # Tensorboard callback.
        tf.keras.callbacks.TensorBoard(log_dir=MODEL_PATH)
    ]
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)

    model.fit(trainX,  # 输入数据列表
              trainY,  # 输入标签列表
              batch_size=batch_size_total,  # 梯度更新时样本数
              epochs=epochs,  # 训练轮数
              verbose=1,  # log等级
              validation_data=(testX, testY),  # 测试数据与标签
              callbacks=callbacks  # 设置一些回调
              )
    """
    # 使用dataset 数据集作为输入，使用fit_generator训练模型
    dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess = tf.InteractiveSession()
    sess.run(iterator.initializer)
    def _input_func(sess, next_element):
        while True:
            val = sess.run(next_element)
            yield val
    steps_per_epoch = trainX.shape[0] / batch_size
    model.fit_generator(_input_func(sess, next_element),
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        verbose=1,
                        validation_data=(testX, testY))
    """
    score = model.evaluate(testX, testY, verbose=1, batch_size=100)  # 评估模型
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save(SAVER_PATH, save_format='tf')


def predict(testX):
    """预测输入图片的类型

    Args:
        testX: The test data features.
    """
    # 加载整个模型，包括 graphs 和 weights
    model = tf.keras.models.load_model(SAVER_PATH)
    summary_writer = tf.summary.create_file_writer('./orig_tensorboard')

    @tf.function
    def _pred(model, x):
        return model(x)

    beg_time = None
    for i in range(20):
        if i == 10:
            beg_time = time.time()
        tf.summary.trace_on(graph=True)
        # labels = model(testX)
        labels = _pred(model, testX)
        with summary_writer.as_default():
            tf.summary.trace_export('org', step=i)

    summary_writer.close()
    print('>>>>>>>>>>>Predict average time: {}s.'.format(round((time.time() - beg_time) / 10, 4)))

    # 显示分类结果
    n = testX.shape[0]
    nc = int(ceil(n / 4))
    f, axes = plt.subplots(nc, 4)
    for i in range(nc * 4):
        x = i // 4
        y = i % 4
        axes[x, y].axis('off')

        label = argmax(labels[i])
        confidence = max(labels[i])
        if i > n:
            break
        axes[x, y].imshow(reshape(testX[i] * 255, (28, 28)))
        axes[x, y].text(0.5, -1.0, str(label) + "\n%.3f" %
                        confidence, fontsize=14)
    # plt.show()
    plt.savefig('./pred_result')


def convert_2_trt():
    """将模型转换为tensorrt模型，tensorflow版本为2.0.4"""
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    conversion_params = trt.TrtConversionParams(
        maximum_cached_engines=10
    )
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=SAVER_PATH,
                                        input_saved_model_tags=['serve'],
                                        input_saved_model_signature_key='serving_default',
                                        conversion_params=conversion_params)
    converter.convert()

    def my_input_fn():
        inp = np.zeros(shape=(100, 28, 28, 1)).astype(np.float32)
        yield [inp]

    converter.build(input_fn=my_input_fn)
    converter.save(TRT_PATH)
    print('Convert completely!')


def trt_predict(testX):
    """使用tensorrt模型预测输入图片的类型

    Args:
        testX: The test data features.
    """
    # 加载整个模型，包括 graphs 和 weights
    model = tf.saved_model.load(TRT_PATH)

    summary_writer = tf.summary.create_file_writer('./trt_tensorboard')

    @tf.function
    def _pred(model, x):
        return model(x)

    beg_time = None
    for i in range(20):
        if i == 10:
            beg_time = time.time()
        tf.summary.trace_on(graph=True)
        # labels = model(testX)
        labels = _pred(model, testX)
        with summary_writer.as_default():
            tf.summary.trace_export('trt', step=i)

    summary_writer.close()
    print('>>>>>>>>>>>Predict average time: {}s.'.format(round((time.time() - beg_time) / 10, 4)))

    # 显示分类结果
    n = testX.shape[0]
    nc = int(ceil(n / 4))
    f, axes = plt.subplots(nc, 4)
    for i in range(nc * 4):
        x = i // 4
        y = i % 4
        axes[x, y].axis('off')

        label = argmax(labels[i])
        confidence = max(labels[i])
        if i > n:
            break
        axes[x, y].imshow(reshape(testX[i] * 255, (28, 28)))
        axes[x, y].text(0.5, -1.0, str(label) + "\n%.3f" %
                        confidence, fontsize=14)
    # plt.show()
    plt.savefig('./trt_pred_result')


if __name__ == "__main__":
    # 图片宽和高
    img_h, img_w = 28, 28

    # 加载手写识别数据集，其中X代表图片数据，Y代表label
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

    # 将图片数据转换为CNN模型输入的shape
    trainX = array(trainX).reshape(trainX.shape[0], img_h, img_w, 1)
    testX = testX.reshape(testX.shape[0], img_h, img_w, 1)

    # 转换图片数据的数据类型
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')

    # 将数据缩小到0-1范围内
    trainX = trainX / 255
    testX = testX / 255

    print('trainX shape:', trainX.shape)
    print(trainX.shape[0], 'trainX samples')
    print(testX.shape[0], 'testX samples')

    # 构建并训练分类手写识别数据的cnn模型
    train_keras_cnn(trainX, trainY, testX, testY)

    # 预测测试数据中前8个图片的类型
    predict(testX[: 8])

    # 模型转换为tensorrt模型加速推理
    convert_2_trt()

    # 使用转换后的tensorrt模型预估
    trt_predict(testX[: 8])
