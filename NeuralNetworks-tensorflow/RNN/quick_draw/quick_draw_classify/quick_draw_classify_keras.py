#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Train a classifier with bi-rnn model implemented by keras for quick draw
dataset.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/11/29 下午4:51
"""

# common libs.
import os
import math

# 3rd-part libs.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# tf.enable_eager_execution()

print(tf.__version__)

BATCH_SIZE = 50
TRAIN_DATA_PATTERN = "/tmp/quickdraw_data/training.tfrecord-*"
EVAL_DATA_PATTERN = "/tmp/quickdraw_data/eval.tfrecord-*"
TRAIN_CLASS_FILE_PATH = "/tmp/quickdraw_data/training.tfrecord.classes"
EVAL_CLASS_FILE_PATH = "/tmp/quickdraw_data/eval.tfrecord.classes"
MODEL_DIR = "/tmp/quick_draw_classify_keras"
MODEL_SAVER_PATH = os.path.join(MODEL_DIR, "model.h5")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "weights.hdf5")


def get_classes(training):
    """Get classes from file.

    Args:
        training: If training.

    Returns:
        The classes array.
    """
    classes = []
    if training:
        file = TRAIN_CLASS_FILE_PATH
    else:
        file = EVAL_CLASS_FILE_PATH
    with tf.compat.v1.gfile.GFile(file, "r") as f:
        classes = [x for x in f]
    return classes


def get_input_generator(mode, tfrecord_pattern, batch_size):
    """Creates generator.

    Args:
        mode: "train" or "eval"
        tfrecord_pattern: path to a TF record file created using
        create_dataset_for_classify.py.
        batch_size: the batch size to output.

    Returns:
        The generator.
    """

    def _parse_tfexample_fn(example_proto):
        """Parse a single record which is expected to be a tensorflow.Example."""
        feature_to_type = {
            "ink": tf.compat.v1.VarLenFeature(dtype=tf.float32),
            "shape": tf.compat.v1.FixedLenFeature([2], dtype=tf.int64),
            "class_index": tf.compat.v1.FixedLenFeature([1], dtype=tf.int64)
        }
        parsed_features = tf.compat.v1.parse_single_example(example_proto,
                                                            feature_to_type)
        labels = parsed_features["class_index"]
        inks = tf.compat.v1.sparse_tensor_to_dense(parsed_features["ink"])
        labels = tf.reshape(labels, [])
        inks = tf.reshape(inks, [-1, 3])
        # print(labels)
        # print(inks)
        return inks, labels

    dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
    if mode == "train":
        dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.repeat()
    # Preprocess 10 files concurrently and interleaves records from each file.
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        block_length=1)
    dataset = dataset.map(_parse_tfexample_fn,
                          num_parallel_calls=10)
    dataset = dataset.prefetch(10000)
    if mode == "train":
        dataset = dataset.shuffle(buffer_size=1000000)
    # Our inputs are variable length, so pad them.
    dataset = dataset.padded_batch(
        batch_size, padded_shapes=dataset.output_shapes)
    return dataset.make_one_shot_iterator()


def model_func(cnn_len=[5, 5, 3], cnn_filters=[48, 64, 96],  # con relative.
               rnn_layer=3, units=128, rnn_type="lstm",  # rnn relative.
               batch_norm=True,
               dropout=True, dropout_rate=0.3,  # Dropout relative.
               batch_size=BATCH_SIZE):
    """Build the model with functional api.

    Args:
        cnn_len: Length of the convolution filters.
        cnn_filters: The number of the convolution filters.
        rnn_layer: The rnn layers number.
        units: The rnn units number.
        rnn_type: The rnn type. "lstm" or "gru".
        batch_norm: If batch normalization.
        dropout: If dropout.
        dropout_rate: Dropout rate.
        batch_size: The batch size.

    Returns:
        The input and output layer.
    """
    # The inks input. (x_delta, y_delta)
    inks_inp = tf.keras.layers.Input(shape=(None, 3), batch_size=batch_size)

    # Add convolution layers.
    con = inks_inp
    for filter_len, filter_num in zip(cnn_len, cnn_filters):
        con = tf.keras.layers.Conv1D(filters=filter_num,
                                     kernel_size=filter_len,
                                     padding="same",
                                     activation=None,
                                     strides=1)(con)
        if batch_norm:
            con = tf.keras.layers.BatchNormalization()(con)
        if dropout:
            con = tf.keras.layers.Dropout(dropout_rate)(con)

    # Add mask layers.
    mask = tf.keras.layers.Masking(mask_value=0)(con)

    # Add rnn layers.
    rnn = mask
    if rnn_type == "lstm":
        cell = tf.keras.layers.LSTM
    else:
        cell = tf.keras.layers.GRU

    for layer in range(rnn_layer):
        cell_layer = cell(
            units,
            return_sequences=True if layer < rnn_layer - 1 else False,
            recurrent_activation='sigmoid',
            recurrent_initializer='glorot_uniform',
            go_backwards=False,
            dropout=dropout_rate)
        rnn = tf.keras.layers.Bidirectional(
            layer=cell_layer,
            merge_mode="concat")(rnn)

    # Add full-con layers.
    logits = tf.keras.layers.Dense(units=len(get_classes(training=True)),
                                   activation="softmax")(rnn)

    return inks_inp, logits


def train(model, train_data_pattern, eval_data_pattern):
    """Train the model.

    Args:
        model: The model.
        train_data_pattern: The training data pattern.
        eval_data_pattern: The evaluation data pattern.
    """
    # Load previous weights if exist.
    if os.path.exists(CHECKPOINT_PATH):
        print("Loading weights: " + CHECKPOINT_PATH)
        model.load_weights(CHECKPOINT_PATH)

    # Create the model dir.
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # model fit callback.
    callbacks = [
        # Model saver callback.
        tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                           save_weights_only=True,
                                           save_freq=1000),
        # Tensorboard callback.
        tf.keras.callbacks.TensorBoard(log_dir=MODEL_DIR, update_freq=1000)
    ]

    # Train.
    model.fit_generator(
        generator=get_input_generator("train", train_data_pattern, BATCH_SIZE),
        validation_data=get_input_generator("eval", eval_data_pattern,
                                            BATCH_SIZE),
        validation_steps=1000,
        steps_per_epoch=20000,
        epochs=30,
        verbose=1,
        callbacks=callbacks,
    )

    # Save the full model.
    model.save(MODEL_SAVER_PATH)

    # Test.
    predict(MODEL_SAVER_PATH, eval_data_pattern)


def predict(model_saver_path, eval_data_pattern, num=30):
    """predict"""
    model = tf.keras.models.load_model(model_saver_path)
    model.summary()

    # Prepare some inks from evaluation data.
    datasets = get_input_generator("eval", eval_data_pattern, BATCH_SIZE)
    draws, labels_true = datasets.get_next()
    labels_true = labels_true.numpy()
    print(labels_true)

    # Prediction.
    labels_pred = model(draws, training=False).numpy()
    labels_ind = labels_pred.argmax(axis=-1)
    labels_prob = labels_pred.max(axis=-1)

    # Load class name.
    class_names = get_classes(False)

    # The rows and columns of plot.
    plt_columns = 5
    plt_rows = math.ceil(num / 5)

    # Plot.
    plt.figure(figsize=(10, 10))
    for i, draw in enumerate(draws):
        plot_quick_draw(draw,
                        "{0}({1:.2%})".format(class_names[labels_ind[i]],
                                              labels_prob[i]),
                        labels_true[i] == labels_ind[i],  # If pred rightly
                        plt_rows,
                        plt_columns, i + 1)
        if i >= num - 1:
            break
    plt.show()


def plot_quick_draw(inks, cls_name, right, *sub_plt_place):
    """Plot the quick drawing.

    Args:
        inks: The ink deltas array with shape(ink_num, 3). Every delta is (
        x_delta, y_delta, if_end).
        cls_name: The class name.
        right: If predict rightly.
        sub_plt_place: The subplot place.
    """
    print("inks_max: \n", str(np.max(inks[:, 0])), " ", str(np.max(inks[:,
                                                                   1])))
    print("inks_min: \n", str(np.min(inks[:, 0])), " ", str(np.min(inks[:,
                                                                   1])))

    inks_num = inks.shape[0] + 1  # The total inks number.
    plt_ink = np.zeros((inks_num, 3))

    # Convert deltas to plot inks with start point(0, 0).
    for i in range(1, inks_num):
        plt_ink[i, 0: -1] = plt_ink[i - 1, 0: -1] + inks[i - 1, 0: -1]
        plt_ink[i, -1] = inks[i - 1, -1]

    end_points = np.where(plt_ink[:, -1] == 1)[0]  # Find the end points.

    # Plot.
    print(*sub_plt_place)
    plt.subplot(*sub_plt_place)
    # print(end_points)
    for i, end_pt in enumerate(end_points):
        if i == 0:
            plt.plot(plt_ink[0: end_pt + 1, 0], plt_ink[0: end_pt + 1, 1])
        else:
            plt.plot(plt_ink[end_points[i - 1] + 1: end_pt + 1, 0],
                     plt_ink[end_points[i - 1] + 1: end_pt + 1, 1])
    plt.axis('off')
    plt.title(str(cls_name), color="g" if right else "r")


if __name__ == "__main__":
    inp, out = model_func()
    model = tf.keras.Model(inp, out)
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()
    train(model, TRAIN_DATA_PATTERN, EVAL_DATA_PATTERN)
