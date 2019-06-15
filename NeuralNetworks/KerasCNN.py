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

# 3rd-part libs.
from numpy import *
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

MODEL_PATH = "/tmp/kerasCNN/"
SAVER_PATH = "/tmp/kerasCNN/model_saver"


def trainKerasCNN(trainX, trainY, testX, testY):
    """使用keras构建CNN训练手写识别系统

    Args:
        trainX: 训练输入数据
        trainY: 训练标签数据
        testX: 测试输入数据
        testY: 测试标签数据

    Returns:
        errRate: 错误率
    """
    batchSize = 10
    epochs = 10
    numClass = 10  # 手写识别有10个数字

    # convert class vectors to binary class matrices - this is for use in the
    # categorical_crossentropy loss below
    trainY = keras.utils.to_categorical(trainY, numClass)
    testY = keras.utils.to_categorical(testY, numClass)
    # print("trainY:\n", str(trainY), "testY:\n", str(testY))

    model = Sequential()  # 创建序贯模型
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu', input_shape=[28, 28, 1]
                     ))  # 添加一个卷积层, 32个卷积核，激活函数用relu
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # 添加一个max pool层
    model.add(Conv2D(64, (5, 5), activation="relu"))  # 添加第二个卷积层
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 添加第二个max pool层
    model.add(Flatten())  # 添加flatten层
    model.add(Dense(1000, activation="relu"))  # 添加完全连接层，1000个nn，使用relu激活函数
    model.add(Dense(numClass, activation="softmax"))  # 添加完全连接层作为输出层，分成10个类

    model.compile(loss=keras.losses.categorical_crossentropy,  # 标准交叉熵来进行分类
                  optimizer=keras.optimizers.Adam(),  # 使用Adam优化器
                  metrics=['accuracy'])  # 在训练和测试时需要评估的度量

    def learning_rate_scheduler(epoch):
        """The learning rate scheduler function.

        Args:
            epoch: Current epoch.

        Returns:
            The learning rate.
        """
        print("learning_rate_scheduler epoch:", epoch)
        return 1e-3

    callbacks = [
        # Model saver callback.
        keras.callbacks.ModelCheckpoint(filepath=SAVER_PATH),

        # Learning rate scheduler.
        keras.callbacks.LearningRateScheduler(learning_rate_scheduler,
                                              verbose=1),

        # Tensorboard callback.
        keras.callbacks.TensorBoard(log_dir=MODEL_PATH)
    ]

    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)

    model.fit(trainX,  # 输入数据列表
              trainY,  # 输入标签列表
              batch_size=batchSize,  # 梯度更新时样本数
              epochs=epochs,  # 训练轮数
              verbose=1,  # log等级
              validation_data=(testX, testY),  # 测试数据与标签
              callbacks=callbacks  # 设置一些回调
              )

    """
    # 使用dataset 数据集作为输入，使用fit_generator训练模型
    dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
    dataset = dataset.batch(batchSize)
    dataset = dataset.repeat()
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess = tf.InteractiveSession()
    sess.run(iterator.initializer)

    def _input_func(sess, next_element):
        while True:
            val = sess.run(next_element)
            yield val

    steps_per_epoch = trainX.shape[0] / batchSize
    model.fit_generator(_input_func(sess, next_element),
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        verbose=1,
                        validation_data=(testX, testY))
    """

    score = model.evaluate(testX, testY, verbose=0)  # 评估模型
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == "__main__":
    # input image dimensions
    imgX, imgY = 28, 28

    (trainX, trainY), (testX, testY) = mnist.load_data()
    # trainX, trainY = Utils.loadData("trainingDigits")
    # testX, testY = Utils.loadData("testDigits")

    # reshape the data into a 4D tensor - (sample_number, x_img_size,
    # y_img_size, num_channels)
    # because the MNIST is greyscale, we only have a single channel - RGB
    # colour images would have 3
    trainX = array(trainX).reshape(trainX.shape[0], imgX, imgY, 1)
    testX = testX.reshape(testX.shape[0], imgX, imgY, 1)
    input_shape = (imgX, imgY, 1)

    # convert the data to the right type
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX /= 255
    testX /= 255
    print('trainX shape:', trainX.shape)
    print(trainX.shape[0], 'trainX samples')
    print(testX.shape[0], 'testX samples')
    trainKerasCNN(trainX, trainY, testX, testY)
