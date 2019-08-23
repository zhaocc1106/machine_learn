#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Practice the usage of tf.data.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    19-3-19 上午9:40
"""

# 3rd libs.
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.contrib.data import CsvDataset as CsvDataSet


def one_shot_iterator_test(sess):
    """Test one shot iterator.

    Args:
        sess: session
    """
    print("one_shot_iterator_test")
    dataset = tf.data.Dataset.range(100)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    for i in range(100):
        value = sess.run(next_element)
        print(value)


def initialize_iterator_test(sess):
    """Test initialization iterator.

    Args:
        sess: session.
    """
    print("initialize_iterator_test")
    dataset = tf.data.Dataset.from_tensor_slices(
        {"a": tf.random_uniform([4]),
         "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    sess.run(iterator.initializer)
    for i in range(4):
        value = sess.run(next_element)
        print(value)


def reinitialize_iterator_test(sess):
    """Test reinitialize iterator.

    Args:
        sess: session
    """
    print("reinitialize_iterator_test")
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
    validation_dataset = tf.data.Dataset.range(50)

    # A reinitializable iterator is defined by its structure. We could use the
    # `output_types` and `output_shapes` properties of either `training_dataset`
    # or `validation_dataset` here, because they are compatible.
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)

    # Run 20 epochs in which the training dataset is traversed, followed by the
    # validation dataset.
    for _ in range(2):
        # Initialize an iterator over the training dataset.
        sess.run(training_init_op)
        print("training_dataset:")
        for _ in range(100):
            value = sess.run(next_element)
            print(value)

        # Initialize an iterator over the validation dataset.
        sess.run(validation_init_op)
        print("validation_dataset:")
        for _ in range(50):
            value = sess.run(next_element)
            print(value)


def load_npy_test(sess):
    """Test load numpy npy files data.
    Convert numpy data to tf.constant directly. It wastes ram. And may reach
    the 2GB limit of the tf.GraphDef protocol buffer.

    Args:
        sess: session.
    """
    print("load_npy_test")
    # Save numpy data.
    a = np.ones(shape=[10, 10])
    b = np.random.normal(size=[10, 10])
    d = {"a": a, "b": b}
    np.save("./test.npy", d)

    # Load numpy datas.
    data = np.load("./test.npy").item()
    print(data)
    features = data['a']
    print(features)
    labels = data['b']
    print(labels)

    # Convert numpy data to dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    print(dataset)

    # Make iterator.
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Iter.
    sess.run(iterator.initializer)
    for i in range(10):
        value = sess.run(next_element)
        print(value)


def load_npy_with_feed_test(sess):
    """Test load numpy npy files data with feeds.
    Don't waste ram.

    Args:
        sess: session.
    """
    print("load_npy_with_feed_test")
    # Save numpy data.
    a = np.ones(shape=[10, 10])
    b = np.random.normal(size=[10, 10])
    d = {"a": a, "b": b}
    np.save("./test.npy", d)

    # Load numpy datas.
    data = np.load("./test.npy").item()
    print(data)
    features = data['a']
    # print(features)
    labels = data['b']
    # print(labels)

    # Define placeholders.
    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    # Make iterator.
    dataset = tf.data.Dataset.from_tensor_slices(
        (features_placeholder, labels_placeholder))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess.run(iterator.initializer, feed_dict={
        features_placeholder: features,
        labels_placeholder: labels
    })

    # Iter.
    for i in range(10):
        value = sess.run(next_element)
        print(value)


def save_data_into_tf_records(image_file, image_label, tf_records_writer):
    """Parse image file and save the image data and label into tfRecord file.

    Args:
        image_file: The image file path.
        image_label: The image label.
        tf_records_writer: The tfRecords writer.
    """
    print("save_data_into_tf_records image_file: ", image_file)
    try:
        img = Image.open(image_file)
    except OSError as e:
        print(e)
        print("Error image " + image_file)
        return
    # Unify resolution to 300 * 300.
    img = np.array(img.resize((300, 300)))
    # img = np.array(img)

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


def _parse_image_tf_records_function(example_proto):
    """Define the parser function of image tfRecords files.

    Args:
        example_proto: The data with example proto.

    Returns:
        image data and image label.
    """
    print("_parse_image_tf_records_function")
    features = tf.parse_single_example(
        example_proto,
        features={
            "height": tf.FixedLenFeature([], tf.int64),
            "width": tf.FixedLenFeature([], tf.int64),
            "channel": tf.FixedLenFeature([], tf.int64),
            "label": tf.FixedLenFeature([], tf.int64),
            "raw": tf.FixedLenFeature([], tf.string)
        })
    img_label = tf.cast(features["label"], tf.int32)
    # print(img_label)
    img_height = tf.cast(features["height"], tf.int32)
    # print(img_height)
    img_width = tf.cast(features["width"], tf.int32)
    # print(img_width)
    img_channel = tf.cast(features["channel"], tf.int32)
    # print(img_channel)
    img_data = tf.decode_raw(features["raw"], tf.uint8)
    img_data = tf.reshape(img_data, [300, 300, 3])
    # print(img_data)
    return img_data, img_label


def load_tf_records_test(sess):
    """Test load tfRecord files.

    Args:
        sess: session.
    """
    print("load_tf_records_test")
    # Create dataset of tfRecords files.
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    # Parse dataset to tensor.
    dataset = dataset.map(_parse_image_tf_records_function)
    dataset = dataset.shuffle(buffer_size=10, seed=10)
    dataset = dataset.repeat(10)
    dataset = dataset.batch(5)

    # Create iterator.
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    training_filenames = ["./test.tfRecords"]

    # Run initializer.
    sess.run(iterator.initializer,
             feed_dict={filenames: training_filenames})

    # Iter.
    while True:
        try:
            print("next element:")
            img_data, img_label = sess.run(next_element)
            # print("img_data:")
            # print(img_data)
            print("img_label:")
            print(img_label)
        except tf.errors.OutOfRangeError:
            print("out of range error.")
            break


def load_csv_test(sess):
    """Test load csv files.

    Args:
        sess: session.
    """
    # Create CsvDataSet.
    filenames = ["census_test.csv"]
    record_defaults = [[""]] * 15
    dataset = CsvDataSet(filenames, record_defaults)
    dataset = dataset.shuffle(buffer_size=100, seed=10)
    dataset = dataset.repeat(10)
    dataset = dataset.batch(5)

    # Create iterator.
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Run initializer.
    sess.run(iterator.initializer)

    # Iter.
    while True:
        try:
            print("next element:")
            print(sess.run(next_element))
        except tf.errors.OutOfRangeError:
            print("out of range.")
            break


def load_image_files_test(sess):
    """Test load image files.

    Args:
        sess: session.
    """

    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_resized, label

    # A vector of filenames.
    filenames = tf.constant(
        ["../datas/image_net_origin_files/person/person_0.jpg",
         "../datas/image_net_origin_files/person/person_20.jpg",
         "../datas/image_net_origin_files/person/person_80.jpg",
         "../datas/image_net_origin_files/person/person_120.jpg",
         "../datas/image_net_origin_files/person/person_160.jpg"])

    # `labels[i]` is the label for the image in `filenames[i].
    labels = tf.constant([0, 1, 2, 3, 4])

    # Create dataset.
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=100, seed=10)
    dataset = dataset.repeat(10)
    dataset = dataset.batch(5)

    # Create iterator.
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Run initializer.
    sess.run(iterator.initializer)

    # Iter.
    while True:
        try:
            print("next element:")
            image_data, image_label = sess.run(next_element)
            print(image_label)
        except tf.errors.OutOfRangeError:
            print("out of range.")
            break


def load_images_with_py_func_test(sess):
    """Test load images files with py_func.

    Args:
        sess: session.
    """

    # Use a custom OpenCV function to read the image, instead of the standard
    # TensorFlow `tf.read_file()` operation.
    def _read_py_function(filename, label):
        print("_read_py_function filename: %s, label: %s" % (filename, label))
        image_decoded = cv2.imread(filename.decode())
        # image_decoded = cv2.resize(image_decoded, (28, 28))
        # print(np.shape(image_decoded))
        return image_decoded, label

    # Use standard TensorFlow operations to resize the image to a fixed shape.
    def _resize_function(image_decoded, label):
        print("_resize_function")
        image_decoded.set_shape([None, None, None])
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_resized, label

    # A vector of filenames.
    filenames = tf.constant(
        ["../datas/image_net_origin_files/person/person_0.jpg",
         "../datas/image_net_origin_files/person/person_20.jpg",
         "../datas/image_net_origin_files/person/person_80.jpg",
         "../datas/image_net_origin_files/person/person_120.jpg",
         "../datas/image_net_origin_files/person/person_160.jpg"])

    # `labels[i]` is the label for the image in `filenames[i].
    labels = tf.constant([0, 1, 2, 3, 4])

    # Create dataset.
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_func(
            _read_py_function, [filename, label], [tf.uint8, label.dtype])))
    dataset = dataset.map(_resize_function)
    dataset = dataset.shuffle(buffer_size=5, seed=10)
    dataset = dataset.repeat(10)
    dataset = dataset.batch(2)

    # Create iterator.
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Run initializer.
    sess.run(iterator.initializer)

    # Iter.
    while True:
        try:
            print("next element:")
            image_data, image_label = sess.run(next_element)
            print(image_label)
        except tf.errors.OutOfRangeError:
            print("out of range.")
            break


def load_text_test(sess):
    """ Load text file with tab split.

    Args:
        sess: The session.
    """
    batch_size = 10
    shuffle = True
    num_epochs = 5
    _COLUMNS = ['x', 'y', 'label']
    _NUM_EXAMPLE = {'train': 100, 'test': 100}
    data_file = './testSet.txt'

    assert tf.gfile.Exists(data_file), ('%s not found.' % data_file)

    def parse_text(line):
        # Split line with tab.
        columns = tf.string_split([line], "\t").values
        # String to number.
        columns = tf.string_to_number(columns)
        # Convert to dictionary.
        feature_dict = dict(
            {_COLUMNS[i]: columns[i] for i in range(len(_COLUMNS))})
        # Pop label column.
        label = feature_dict.pop("label")
        return feature_dict, label

    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(map_func=parse_text)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLE['train'], seed=10)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    # Create iterator
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Run initializer
    sess.run(iterator.initializer)

    # Iter
    while True:
        try:
            print('next element:')
            feature_columns, label = sess.run(next_element)
            print('feature_columns: ', str(feature_columns))
            print('label: ', str(label))
        except tf.errors.OutOfRangeError:
            print('out of range.')
            break


if __name__ == "__main__":
    sess = tf.InteractiveSession()
    # one_shot_iterator_test(sess) # Test one shot iterator.
    # initialize_iterator_test(sess) # Test initializable iterator.
    # reinitialize_iterator_test(sess) # Test reinitialize iterator.
    # load_npy_test(sess) # Test load npy files.
    # load_npy_with_feed_test(sess) # Test load npy files with feeds.

    # Test load tfRecords.
    # Save image info into tfRecords.
    # Use tools.image_net_downloader to download image files firstly.
    # tf_record_writer = tf.python_io.TFRecordWriter(path="./test.tfRecords")
    # save_data_into_tf_records(
    #     image_file=
    #     "../datas/image_net_origin_files/person/person_0.jpg",
    #     image_label=0,
    #     tf_records_writer=tf_record_writer
    # )
    # save_data_into_tf_records(
    #     image_file="../datas/image_net_origin_files/person/person_20.jpg",
    #     image_label=1,
    #     tf_records_writer=tf_record_writer
    # )
    # save_data_into_tf_records(
    #     image_file="../datas/image_net_origin_files/person/person_80.jpg",
    #     image_label=2,
    #     tf_records_writer=tf_record_writer
    # )
    # save_data_into_tf_records(
    #     image_file="../datas/image_net_origin_files/person/person_120.jpg",
    #     image_label=3,
    #     tf_records_writer=tf_record_writer
    # )
    # save_data_into_tf_records(
    #     image_file="../datas/image_net_origin_files/person/person_160.jpg",
    #     image_label=4,
    #     tf_records_writer=tf_record_writer
    # )
    # # Must close tfRecords writer, else will throw data loss exception.
    # tf_record_writer.close()
    # load_tf_records_test(sess)

    # Test load csv files.
    # load_csv_test(sess)

    # Test load image files.
    # load_image_files_test(sess)

    # Test load image files with py_func.
    # load_images_with_py_func_test(sess)

    # Test load pure text files.
    load_text_test(sess)