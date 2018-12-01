#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""The reader of image net data tfRecords files.


Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/11/22 20:58
"""
# Common libs.
import time

# 3rd part libs.
import numpy as np
from PIL import Image
import tensorflow as tf

tf_records_dir = "../image_net_records_files/"
IMAGE_SIZE = 224

"""
def load_data_from_tf_records(tf_records_file):
    Load data form tfRecords file.

    Args:
        tf_records_file: The tfRecords file path.

    Returns:
        The list of data.

    image_datas = []
    image_labels = []
    for serialized_example in tf.python_io.tf_record_iterator(tf_records_file):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        image_raw = example.features.feature['image_raw'].bytes_list.value
        image_label = example.features.feature['image_label'].int64_list.value
        try:
            image_data = Image.frombytes(mode="RGB", size=(IMAGE_SIZE, IMAGE_SIZE),
                                         data=image_raw[0])
        except ValueError as e:
            print(e)
            continue
        # image_data.show("bird_0.jpg")
        # print("image_data:")
        # print(np.array(image_data))
        # print("image_raw:")
        # print(image_raw)
        # print("image_label:")
        # print(image_label)
        # print("image_size:")
        # print(image_size)
        image_datas.append(image_data)
        image_labels.append(image_label)
    return image_datas, image_labels
"""


def read_img_net(file_name_queue):
    """Read and parse image net data tfRecords files.

    Args:
        file_name_queue: The file names queue.

    Returns:
        An object representing a single example, with the following fields:
        height: number of rows in the result.
        width: number of columns in the result.
        depth: number of color channels in the result.
        key: a scalar string Tensor describing the filename & record number
            for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(file_name_queue)
    print(key)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "height": tf.FixedLenFeature([], tf.int64),
            "width": tf.FixedLenFeature([], tf.int64),
            "channel": tf.FixedLenFeature([], tf.int64),
            "label": tf.FixedLenFeature([], tf.int64),
            "raw": tf.FixedLenFeature([], tf.string)
        })
    img_label = tf.cast(features["label"], tf.int32)
    print(img_label)
    img_height = tf.cast(features["height"], tf.int32)
    print(img_height)
    img_width = tf.cast(features["width"], tf.int32)
    print(img_width)
    img_channel = tf.cast(features["channel"], tf.int32)
    print(img_channel)
    img_data = tf.decode_raw(features["raw"], tf.uint8)
    img_data = tf.reshape(img_data, [img_height, img_width, img_channel])
    print(img_data)
    return key, img_data, img_label


def load_distorted_inputs(data_dir, batch_size):
    """Construct distorted input for ImageNet training using the Reader ops.

    Args:
        data_dir: Path of tfRecords files.
        batch_size: The mini batch size.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    file_name = data_dir + "train.tfRecords"
    if not tf.gfile.Exists(file_name):
        raise ValueError('Failed to find file: ' + file_name)

    # Create a queue that produces the file_name to read.
    file_name_queue = tf.train.string_input_producer([file_name])

    with tf.name_scope("data_augmentation"):
        key, img_data, img_label = read_img_net(file_name_queue)
        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        height = IMAGE_SIZE
        width = IMAGE_SIZE
        reshape = tf.cast(img_data, tf.float32)
        print(reshape)
        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(reshape, [height, width, 3])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the stddev of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])

        # If you want show the augmented image, cast the data to uint8 and do
        #  not standardize the img data.
        # distorted_image = tf.cast(distorted_image, tf.uint8)

    img_data_batch, img_label_batch = \
        tf.train.shuffle_batch([float_image, img_label],
                               batch_size=batch_size,
                               num_threads=12,
                               capacity=2000,
                               min_after_dequeue=1000)
    return img_data_batch, img_label_batch


def load_inputs(eval_data, data_dir, batch_size):
    """Get training data without augmentation or eval data .

    Args:
        eval_data: Training or eval data.
        data_dir: Path of tfRecords files.
        batch_size: The mini batch size.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    if not eval_data:
        file_name = data_dir + "train.tfRecords"
    else:
        file_name = data_dir + "test.tfRecords"
    if not tf.gfile.Exists(file_name):
        raise ValueError('Failed to find file: ' + file_name)

    # Create a queue that produces the file_name to read.
    file_name_queue = tf.train.string_input_producer([file_name])

    with tf.name_scope("inputs"):
        key, img_data, img_label = read_img_net(file_name_queue)
        # Image processing for training the network. Note the many random
        # distortions applied to the image.
        height = IMAGE_SIZE
        width = IMAGE_SIZE

        reshape = tf.cast(img_data, tf.float32)
        # reshaped_image = tf.reshape(reshape, shape=[1, 300, 300, 3])
        resized_image = tf.image.resize_image_with_crop_or_pad(reshape,
                                                               height, width)
        # resized_image = tf.reshape(resized_image, shape=[height, width, 3])
        print(resized_image)

        # Subtract off the mean and divide by the stddev of the pixels.
        float_image = tf.image.per_image_standardization(resized_image)

        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])

    if not eval_data:
        img_data_batch, img_label_batch = \
            tf.train.shuffle_batch([float_image, img_label],
                                   batch_size=batch_size,
                                   num_threads=12,
                                   capacity=2000,
                                   min_after_dequeue=1000)
    else:
        img_data_batch, img_label_batch = \
            tf.train.batch([float_image, img_label],
                           batch_size=batch_size,
                           num_threads=12)
    return img_data_batch, img_label_batch


if __name__ == "__main__":
    # tf_records_writer = tf.python_io.TFRecordWriter(tf_records_dir + "train.tfRecords")
    # save_data_into_tf_records(image_net_origin_files_dir + "bird/bird_0.jpg", 1,
    #                          tf_records_writer)
    # tf_records_writer.close()
    # load_data_from_tf_records(tf_records_dir + "train.tfRecords")

    # save_all_image_files_into_tf_records(image_net_origin_files_dir, tf_records_dir)
    begin_time = time.time()
    # train_image_datas, train_image_labels = load_data_from_tf_records(
    #     tf_records_dir + "train.tfRecords")
    # test_image_datas, test_image_labels = load_data_from_tf_records(
    #     tf_records_dir + "test.tfRecords")
    # print("train_image_datas len:")
    # print(len(train_image_datas))
    # print("train_image_labels len:")
    # print(len(train_image_labels))
    # print("test_image_datas len:")
    # print(len(test_image_datas))
    # print("test_image_labels len:")
    # print(len(test_image_labels))
    # print(train_image_datas)
    # img_data_batch, img_label_batch = load_distorted_inputs(
    #     tf_records_dir, 5)

    img_data_batch, img_label_batch = load_inputs(False, tf_records_dir, 5)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        tf.train.start_queue_runners(sess)
        for i in range(1):
            image_data, l = sess.run([img_data_batch, img_label_batch])
            print("image_data shape")
            # print(image_data)
            print(image_data.shape)
            print("label:")
            print(l)
            # Show the images.
            for num in range(image_data.shape[0]):
                image = Image.fromarray(np.reshape(image_data[num],
                                                   [IMAGE_SIZE, IMAGE_SIZE,
                                                    3]).astype(dtype=np.uint8))
                image.show()
    cost_time = time.time() - begin_time
    print("Cost {0}s".format(cost_time))
