#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Build one rnn model to quick draw automatically.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/9/4 23:38
"""

# common libs.
import os
import shutil
import sys
import time
import argparse
import math

# 3rd-part libs.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

tf.enable_eager_execution()
print(tf.__version__)

UNITS = 512  # Number of RNN units
RNN_LAYERS = 10  # Number of RNN layers.
EPOCHS = 40  # The epoch number.
BATCH_SIZE = 50  # The batch size.
DEFAULT_TRAIN_CLASS = "cat"  # The training class.
MODEL_DIR = "/tmp/autodraw_model/" + DEFAULT_TRAIN_CLASS  # Model dir.
DATA_PATH = "/tmp/autodraw_data"  # Data dir path.
CHECKPOINT_PATH = os.path.join(MODEL_DIR,
                               "ckpt")  # Name of the checkpoint files


def load_training_data(tfrecord_pattern, batch_size):
    """Load training dataset.

    Args:
     tfrecord_pattern: path to a TF record file created using
     create_dataset_for_auto_draw.py.
     batch_size: the batch size to output.

    Returns:
        tf dataset.
    """

    def _parse_tfexample_fn(example_proto):
        """Parse a single record which is expected to be a tensorflow.Example."""
        feature_to_type = {
            "ink": tf.VarLenFeature(dtype=tf.float32),
            "shape": tf.FixedLenFeature([2], dtype=tf.int64)
        }

        parsed_features = tf.parse_single_example(example_proto,
                                                  feature_to_type)
        parsed_features["ink"] = tf.sparse_tensor_to_dense(
            parsed_features["ink"])
        return parsed_features

    dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
    # Preprocesses 10 files concurrently and interleaves records from each file.
    dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        block_length=1)
    dataset = dataset.map(_parse_tfexample_fn)
    dataset = dataset.prefetch(10000)
    dataset = dataset.shuffle(buffer_size=20000)
    # Our inputs are variable length, so pad them.
    dataset = dataset.padded_batch(
        batch_size, padded_shapes=dataset.output_shapes)
    return dataset


def model_func(rnn_layer, units, rnn_type="lstm", training=True,
               batch_size=BATCH_SIZE):
    """The auto-draw model built with keras function api.

    Args:
        rnn_layer: The layers of rnn.
        units: The rnn units.
        rnn_type: The rnn type. "gru" or "lstm".
        training: If training.
        batch_size: The batch size.

    Returns:
        The input and output.
    """
    inputs = tf.keras.Input(shape=(None, 4), batch_size=batch_size)

    x = tf.keras.layers.Dense(1024)(inputs)
    x = tf.keras.layers.Dropout(0.5)(x, training=training)

    # Add rnn layer.
    for i in range(rnn_layer):
        if rnn_type == "gru":
            # Use gru.
            if tf.test.is_gpu_available():
                gru = tf.keras.layers.CuDNNGRU(units,
                                               return_sequences=True,
                                               recurrent_initializer='glorot_uniform',
                                               stateful=True)
            else:
                gru = tf.keras.layers.GRU(units,
                                          return_sequences=True,
                                          recurrent_activation='sigmoid',
                                          recurrent_initializer='glorot_uniform',
                                          stateful=True)
            x = gru(x)
        elif rnn_type == "lstm":
            # Use lstm.
            if tf.test.is_gpu_available():
                lstm = tf.keras.layers.CuDNNLSTM(units,
                                                 return_sequences=True,
                                                 recurrent_initializer='glorot_uniform',
                                                 stateful=True)
            else:
                lstm = tf.keras.layers.LSTM(units,
                                            return_sequences=True,
                                            recurrent_activation='sigmoid',
                                            recurrent_initializer='glorot_uniform',
                                            stateful=True)
            x = lstm(x)

    x = tf.keras.layers.Dropout(0.5)(x, training=training)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.Dropout(0.5)(x, training=training)
    output = tf.keras.layers.Dense(4)(x)

    return inputs, output


def loss_func(real, pred):
    """The loss function.

    Args:
        real: The real value.
        pred: The pred value.

    Returns:
        The loss.
    """
    # resolve real data.
    # real = np.reshape(real, (-1, 3))

    # # (x_delta, y_delta)
    # delta_real = real[:, 0: 2]
    # delta_pred = pred[:, 0: 2]
    #
    # # (not_end_prob, end_prob)
    # end_flag_real = real[:, 2].astype(np.int32)
    # end_flag_pred = pred[:, 2: 4]
    #
    # loss = tf.losses.mean_squared_error(delta_real, delta_pred) + \
    #        tf.losses.sparse_softmax_cross_entropy(end_flag_real, end_flag_pred)

    return tf.losses.mean_squared_error(real, pred)


first_batch_ink = []  # The first quick draw batch of every epoch.
first_batch_shape = []  # The first quick draw batch shape of every epoch.


def train(model, quick_draw_class):
    """Train model.

    Args:
        model: The auto-draw model.
        quick_draw_class: The quick draw class. The end of tfRecord file name.
    """
    file_path = DATA_PATH + "/training.tfrecord-" \
                + quick_draw_class
    print(file_path)
    dataset = load_training_data(file_path, BATCH_SIZE)

    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    os.mkdir(MODEL_DIR)

    # Using adam optimizer with default arguments
    optimizer = tf.train.AdamOptimizer()

    epoch_losses = []
    least_loss = math.inf

    for epoch in range(EPOCHS):
        start_time = time.time()

        for batch, data in enumerate(dataset):
            if batch == 0:
                global first_batch_ink, first_batch_shape
                first_batch_ink = np.reshape(data["ink"], (BATCH_SIZE, -1, 4))
                first_batch_shape = data["shape"]

            hidden = model.reset_states()  # reset the rnn state.

            if data["ink"].shape[0] != BATCH_SIZE:
                break

            # Get ink input and target.
            ink = np.reshape(data["ink"], (BATCH_SIZE, -1, 4))
            ink_input = ink[:, 0: -1]
            ink_target = ink[:, 1:]

            with tf.GradientTape() as tape:
                # feeding the hidden state back into the model
                # This is the interesting step
                pred = model(ink_input, training=True)
                # print("pred:\n", str(pred))
                loss = loss_func(ink_target, pred)

            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {}'.format(epoch + 1,
                                                         batch,
                                                         loss))
            if batch >= 2000:
                break

        # saving (checkpoint) the model every epoch
        if (epoch + 1) % 1 == 0:
            model.save_weights(CHECKPOINT_PATH)

        print('Epoch {} Loss {}'.format(epoch + 1, loss))
        print(
            'Time taken for 1 epoch {} sec\n'.format(time.time() - start_time))

        epoch_losses.append(loss)

        # If get smaller loss, save the model and show auto drawing.
        if loss < least_loss:
            # Save to HDF5 format.
            model.save(os.path.join(MODEL_DIR, quick_draw_class + "_model.h5"))

            auto_draw(quick_draw_class, epoch + 1)
            least_loss = loss

    return epoch_losses


def plot_quick_draw(inks_, cls_name, min_len, max_len, *sub_plt_place):
    """Plot the quick drawing.

    Args:
        inks_: The ink deltas array with shape(ink_num, 4). Every delta is (
        x_delta, y_delta, end_flag, complete_flag).
        cls_name: The class name.
        min_len: The min len of this class.
        max_len: The max len of this class.
    """
    # Find the complete flag ink of total image.
    # print(inks_)
    # print(np.where(inks_[:, -1] == 1.0))
    print("plot_quick_draw cls_name: {}, min_len: {}, max_len: {}".format(
        cls_name, min_len, max_len))

    comp_inds = np.where(inks_[:, -1] == 1.0)[0].tolist()
    print(comp_inds)
    if len(comp_inds) == 0:
        inks = inks_[:, 0: -1].copy()
        print("inks_len: {}".format(len(inks)))
    else:
        # Find the complete index. Convert to range(min_len, max_len).
        ind = 0
        for ind in comp_inds:
            if ind + 1 < min_len:
                continue
            break

        inks_len = ind + 1
        if ind + 1 < min_len:
            inks_len = min_len
        if ind + 1 > max_len:
            inks_len = max_len

        print("inks_len: {}".format(inks_len))
        inks = inks_[0: inks_len, 0: -1].copy()

    # inks[-1, -1] = 1.0
    inks_num = inks.shape[0] + 1  # The total inks number.
    plt_ink = np.zeros((inks_num, 3))

    # Convert deltas to plot inks with start point(0, 0).
    for i in range(1, inks_num):
        plt_ink[i, 0: -1] = plt_ink[i - 1, 0: -1] + inks[i - 1, 0: -1]
        plt_ink[i, -1] = inks[i - 1, -1]

    # Find the end ink of every stroke.
    end_points = np.where(plt_ink[:, -1] == 1.0)[0]

    # Plot.
    plt.subplot(*sub_plt_place)
    # print(end_points)
    for i, end_pt in enumerate(end_points):
        if i == 0:
            plt.plot(plt_ink[0: end_pt + 1, 0], plt_ink[0: end_pt + 1, 1])
        else:
            plt.plot(plt_ink[end_points[i - 1] + 1: end_pt + 1, 0],
                     plt_ink[end_points[i - 1] + 1: end_pt + 1, 1])
    plt.title(str(cls_name))


def auto_draw(quick_draw_class, epoch):
    """Quick draw.

    Args:
        quick_draw_class: The quick draw class.
        epoch: Current epoch.

    """
    input, output = model_func(RNN_LAYERS, UNITS, training=False, batch_size=1)
    model = tf.keras.Model(input, output)
    model.summary()
    model.load_weights(CHECKPOINT_PATH)

    # Get the len range from csv.
    len_ranges = pd.read_csv(os.path.join(DATA_PATH, "len_ranges.csv"))
    df = len_ranges.loc[len_ranges['class_name'] == quick_draw_class]
    min_len = df['min_len'].values[0]
    max_len = df['max_len'].values[0]

    for batch_ind in range(BATCH_SIZE):
        model.reset_states()

        inks = first_batch_ink[batch_ind, 0: 10, :]
        # print(ink)
        input = inks
        input = np.expand_dims(input, 0)
        input = input.astype(np.float32)

        for i in range(max_len - 10):
            pred = model(input).numpy()
            pred = np.reshape(pred, newshape=(-1, 4))
            pred_ = np.zeros(shape=(1, 4))
            pred_[0, 0: 2] = pred[0, 0: 2]  # deltas
            if pred[0, 2] >= 0.5:
                pred_[0, 2] = 1.0  # end of stroke.
            if pred[0, 3] >= 0.5:
                pred_[0, 3] = 1.0  # complete flag of total image.

            inks = np.concatenate([inks, pred_])
            input = np.expand_dims(pred_, 0)
            input = input.astype(np.float32)

        # Plot quick draw.
        plt.figure()
        # The real quick draw.
        plot_quick_draw(np.expand_dims(first_batch_ink[batch_ind, :, :], 0)[0],
                        "real", min_len, max_len, 1, 2, 1)
        plt.axis('off')
        # The predict quick draw.
        inks = np.expand_dims(inks, 0)
        plot_quick_draw(inks[0], "predict", min_len, max_len, 1, 2, 2)
        plt.axis('off')
        img_file_name = str(epoch) + "_" + str(
            batch_ind) + "_" + quick_draw_class
        plt.savefig(os.path.join(MODEL_DIR, img_file_name))
        # plt.show()
        plt.close()


def plot_losses(losses):
    """Plot the losses of every epoch.

    Args:
        losses: The losses list.
    """
    plt.figure()
    plt.subplot()
    x = np.arange(0, len(losses), 1)
    plt.plot(x, losses, label='losses')
    plt.xlabel("epoch")
    plt.ylabel("losses")
    plt.legend()
    plt.show()


def main(argv):
    """The main function."""
    del argv

    if FLAGS.quick_draw_class != DEFAULT_TRAIN_CLASS:
        global MODEL_DIR, CHECKPOINT_PATH
        MODEL_DIR = "/tmp/autodraw_model/" + FLAGS.quick_draw_class  # Model dir.
        CHECKPOINT_PATH = os.path.join(MODEL_DIR,
                                       "ckpt")  # Name of the checkpoint files

    input, output = model_func(RNN_LAYERS, UNITS, training=True)
    model = tf.keras.Model(input, output)
    model.summary()
    epoch_loss = train(model, str(FLAGS.quick_draw_class))
    plot_losses(epoch_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--quick_draw_class",
        type=str,
        default=DEFAULT_TRAIN_CLASS,
        help="The quick draw class for training model.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
