#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""The tool to load image net data.
Include parsing image file to image data array and augmenting image data by
tensorflow.
But it's so slow, so don't use this input tool.
Use img_net_tf_records_reader.py to load image net data.
This tool is just codes to record my learning and idea.



Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/11/12 16:03
"""
# Common libs.
import os
import time
import math
import threading

# 3rd part libs.
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Show chinese normally.
plt.rcParams["font.sans-serif"] = ["SimHei"]
image_net_data_path = "../datas/image_net_origin_files/"
useless_image_array = cv2.imread(image_net_data_path + "bear/bear_14.jpg")
training_images = []
test_images = []
index_training_data = 0  # The index of training files.
index_test_data = 0  # The index of test files.
labels = {"bear": 0,
          "cat": 1,
          "bird": 2,
          "car": 3,
          "chicken": 4,
          "dog": 5,
          "flower": 6,
          "giraffe": 7,
          "person": 8,
          "tree": 9}
graph = tf.get_default_graph() # All parsing threads use the same graph.
lock = threading.Lock() # Lock when use tensorflow to augment images.


class image(object):
    """The image class"""

    def __init__(self, image_val, label):
        """The construct function of image.

        Args:
            image_val: The image value.
            label: The image label. Should be one of the global ```labels```.
        """
        self.image_val = image_val
        self.label = label


def init_img_net_data(n_thread, augmentation):
    """Initialize the image net datas. Divide datas into training data and
    test data.

    Args:
        n_thread: The quantity of threads used to parse images files.
        augmentation: If augment the image data.

    Returns:
        training_files: Training data files.
        test_files: Test data files.
    """
    files = os.listdir(image_net_data_path)

    training_files = []
    test_files = []
    for file in files:
        if os.path.isdir(image_net_data_path + file):
            image_files = os.listdir(image_net_data_path + file)
            length = len(image_files)
            print("len of " + file + ":" + str(length))
            # Use 90% data to train.
            training_size = int(length / 10 * 9)
            print("training size:" + str(training_size) + " test size:" +
                  str(length - training_size))
            training_files.extend(image_files[:training_size])
            test_files.extend(image_files[training_size:])

    np.random.shuffle(training_files)
    np.random.shuffle(test_files)

    global training_images
    global test_images
    begin_time = time.time()

    def parse_img(is_training, files):
        """Parse the image files into data lists.

        Args:
            is_training: If these files are training files.
            files: Image files list.
        """
        for file in files:
            label_name = file.split("_")[0]
            # print(image_net_data_path + label_name + "/" + file)
            image_arr = img2array(image_net_data_path + label_name + "/" +
                                  file, augmentation)
            if type(image_arr) != type(None):
                img = image(image_arr, labels[label_name])
                if is_training:
                    training_images.append(img)
                else:
                    test_images.append(img)
            else:
                continue

    # Load data from image files.
    print("Loading training images...")
    number_training_files = len(training_files)
    batch_size = math.ceil(number_training_files / n_thread)
    # Start all threads parsing training files.
    threads = []
    for index_thread in range(n_thread):
        print("Parsing thread {0}-{1}".format(index_thread * batch_size,
                                              (index_thread + 1) * batch_size))
        t = threading.Thread(target=parse_img,
                             args=(
                                 True,
                                 training_files[
                                 index_thread * batch_size:
                                 (index_thread + 1) * batch_size]))
        t.start()
        threads.append(t)
    # Wait for all threads parsing training files.
    for t in threads:
        print("wait threads parsing training files", str(t))
        t.join()

    print("Loading test images...")
    number_test_files = len(test_files)
    batch_size = math.ceil(number_test_files / n_thread)
    # Start all threads parsing test files.
    threads.clear()
    for index_thread in range(n_thread):
        print("Parsing thread {0}-{1}".format(index_thread * batch_size,
                                              (index_thread + 1) * batch_size))
        t = threading.Thread(target=parse_img,
                             args=(
                                 False,
                                 test_files[
                                 index_thread * batch_size:
                                 (index_thread + 1) * batch_size]))
        t.start()
        threads.append(t)
    # Wait for all threads parsing test files.
    for t in threads:
        print("wait threads parsing test files", str(t))
        t.join()
    print("The length of training images:")
    print(len(training_images))
    print("The length of test images:")
    print(len(test_images))
    cost_time = time.time() - begin_time
    print("Loading use {0}s".format(cost_time))


def load_image_data(test_data, mini_batch):
    """Load image datas with mini batch size.

    Args:
        test_data: bool, indicating if one should use the train or test data
        set.
        mini_batch: The mini batch size.

    Returns:
        The image datas with mini batch size.
    """
    global index_training_data
    global index_test_data
    global training_images
    global test_images
    if not test_data:
        images = training_images
        index = index_training_data
    else:
        images = test_images
        index = index_test_data

    images_size = len(images)
    images_data = []
    labels_data = []
    if (index + mini_batch) <= images_size:
        for image in training_images[index:
        index + mini_batch]:
            images_data.append(image.image_val)
            labels_data.append(image.label)
            index += mini_batch
    else:
        for image in test_images[index:]:
            images_data.append(image.image_val)
            labels_data.append(image.label)
        remain_size = mini_batch - (images_size - index)
        for image in test_images[:remain_size]:
            images_data.append(image.image_val)
            labels_data.append(image.label)
            index += remain_size

    if not test_data:
        index_training_data = index
    else:
        index_test_data = index
    print(images_data)
    print(labels_data)
    return tf.convert_to_tensor(images_data), tf.convert_to_tensor(labels_data)


def img2array(file, augmentation):
    """Convert the image to matrix.

    Args:
        file: The image file.
        augmentation: If augment the image.
init
    Returns:
        The tensor operation of the image.
    """
    img = cv2.imread(file)
    # Check if the image is invalid.
    if not type(img) is np.ndarray:
        print("Error image " + file)
        return None
    # Check if the image is rgb image.
    if len(img.shape) != 3:
        print("Not rgb image " + file)
        return None
    # Check if the image is useless.
    same = useless_image_array == img
    if type(same) == np.ndarray:
        if (useless_image_array == img).all():
            print("Useless image. " + file)
            return None
    elif type(same) == bool:
        if same:
            print("Useless image. " + file)
            return None

    global graph
    global lock
    lock.acquire()
    with graph.as_default():
        reshape = tf.cast(img, tf.float32)
        crop_img = tf.random_crop(reshape, [224, 224, 3])
        # Save crop img to file.
        # cv2.imwrite("../datas/image_net_origin_files/tmp/bear_0_crop.jpg",
        # crop_img.eval())
        if augmentation:
            h_flip_img = tf.image.random_flip_left_right(crop_img)
            v_flip_img = tf.image.random_flip_up_down(h_flip_img)
            rand_bright_img = tf.image.random_brightness(v_flip_img, max_delta=64)
            rand_cont_img = tf.image.random_contrast(rand_bright_img, lower=0.2,
                                                     upper=1.8)
            # sess = tf.InteractiveSession()
            # plt.figure(1)
            # plt.subplot(121)
            # # Opencv use bgr. Pyplot use rgb.
            # orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # plt.imshow(orig_img)
            # plt.title("Original data")
            #
            # plt.subplot(122)
            # aug_img = cv2.cvtColor(sess.run(tf.cast(rand_cont_img, tf.uint8)),
            #                        cv2.COLOR_BGR2RGB)
            # plt.imshow(aug_img)
            # plt.title("Augmented data")
            # plt.show()
            # sess.close()
            lock.release()
            return rand_cont_img
        else:
            lock.release()
            return crop_img


if __name__ == "__main__":
    # img = img2tensor("../datas/image_net_origin_files/bear/bear_14.jpg")
    init_img_net_data(2, False)
    image_training, label_training = load_image_data(False, 10)
    print("image_training shape:")
    print(image_training.shape)
    print("label_training shape:")
    print(label_training.shape)