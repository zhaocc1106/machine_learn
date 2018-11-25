#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""Parse image files and save the image data into tfRecodes.


Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/11/21 21:31
"""

# Common libs.
import os
import time

# 3rd part libs.
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFile
import tensorflow as tf

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
tf_records_dir = "../image_net_records_files/"
image_net_origin_files_dir = "../image_net_origin_files/"
useless_image_array = cv2.imread(image_net_origin_files_dir + "bear/bear_14.jpg")
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SIZE = 300


def save_data_into_tf_records(image_file, image_label, tf_records_writer):
    """Parse image file and save the image data and label into tfRecord file.

    Args:
        image_file: The image file path.
        image_label: The image label.
        tf_records_writer: The tfRecords writer.
    """
    try:
        img = Image.open(image_file)
    except OSError as e:
        print(e)
        print("Error image " + image_file)
        return
    # Unify resolution to 300 * 300.
    img = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
    # img = np.array(img)

    # Check if the image is rgb image.
    if len(img.shape) != 3 or img.shape[2] != 3:
        print("Not rgb image " + image_file)
        return
        # Check if the image is useless.
    same = useless_image_array == img
    if type(same) == np.ndarray:
        if (useless_image_array == img).all():
            print("Useless image. " + image_file)
            return
    elif type(same) == bool:
        if same:
            print("Useless image. " + image_file)
            return

    img_raw = img.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[
            image_label])),
        "raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            img_raw])),
        "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[
            img.shape[0]])),
        "width": tf.train.Feature(int64_list=tf.train.Int64List(
            value=[img.shape[1]])),
        "channel": tf.train.Feature(int64_list=tf.train.Int64List(
            value=[img.shape[2]]))
    }))
    tf_records_writer.write(example.SerializeToString())


def save_all_image_files_into_tf_records(image_dir, tf_records_dir):
    """Parse all image files and save them into tfRecords file.

    Args:
        image_dir: The image net origin files dir.
        tf_records_dir: The tfRecords files dir.
    """
    files = os.listdir(image_dir)

    training_files = []
    test_files = []
    for file in files:
        if os.path.isdir(image_dir + file):
            image_files = os.listdir(image_dir + file)
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

    # Save all training image files into train.tfRecords
    tf_records_writer = tf.python_io.TFRecordWriter(
        tf_records_dir + "train.tfRecords")
    for training_file in training_files:
        label_name = training_file.split("_")[0]
        image_label = labels[label_name]
        image_path = image_net_origin_files_dir + label_name + "/" + training_file
        save_data_into_tf_records(image_path, image_label, tf_records_writer)
    tf_records_writer.close()

    # Save all test image files into test.tfRecords
    tf_records_writer = tf.python_io.TFRecordWriter(
        tf_records_dir + "test.tfRecords")
    for test_file in test_files:
        label_name = test_file.split("_")[0]
        image_label = labels[label_name]
        image_path = image_net_origin_files_dir + label_name + "/" + test_file
        save_data_into_tf_records(image_path, image_label, tf_records_writer)
    tf_records_writer.close()


if __name__ == "__main__":
    # tf_records_writer = tf.python_io.TFRecordWriter(tf_records_dir + "train.tfRecords")
    # save_data_into_tf_records(image_net_origin_files_dir + "bird/bird_0.jpg", 1,
    #                          tf_records_writer)
    # tf_records_writer.close()
    # load_data_from_tf_records(tf_records_dir + "train.tfRecords")

    save_all_image_files_into_tf_records(image_net_origin_files_dir, tf_records_dir)