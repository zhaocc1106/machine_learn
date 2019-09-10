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
import time
import shutil

# 3rd-part libs.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()

UNITS = 1024  # Number of RNN units
EPOCHS = 100  # The epoch number.
BATCH_SIZE = 8  # The batch size.
MODEL_DIR = "/tmp/autodraw_model"  # Model dir.
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

    dataset = tf.data.TFRecordDataset(tfrecord_pattern)
    dataset = dataset.map(_parse_tfexample_fn)
    dataset = dataset.prefetch(10000)
    dataset = dataset.shuffle(buffer_size=20000)
    # Our inputs are variable length, so pad them.
    dataset = dataset.padded_batch(
        batch_size, padded_shapes=dataset.output_shapes)
    return dataset


def plot_quick_draw(inks, cls_name, *sub_plt_place):
    """Plot the quick drawing.

    Args:
        inks: The ink deltas array with shape(ink_num, 3). Every delta is (
        x_delta, y_delta, if_end).
        cls_name: The class name.
    """
    inks_num = inks.shape[0] + 1  # The total inks number.
    plt_ink = np.zeros((inks_num, 3))

    # Convert deltas to plot inks with start point(0, 0).
    for i in range(1, inks_num):
        plt_ink[i, 0: -1] = plt_ink[i - 1, 0: -1] + inks[i - 1, 0: -1]
        plt_ink[i, -1] = inks[i - 1, -1]

    # Find the end points.
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


class Model(tf.keras.Model):
    def __init__(self, units):
        super(Model, self).__init__()
        self.units = units

        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(self.units,
                                                return_sequences=True,
                                                recurrent_initializer='glorot_uniform',
                                                stateful=True)
        else:
            self.gru = tf.keras.layers.GRU(self.units,
                                           return_sequences=True,
                                           recurrent_activation='sigmoid',
                                           recurrent_initializer='glorot_uniform',
                                           stateful=True)

        self.fc0 = tf.keras.layers.Dense(4096)
        self.fc1 = tf.keras.layers.Dense(4)

    def call(self, inks):
        # output at every time step
        # output shape == (batch_size, seq_length, hidden_size)
        output = self.gru(inks)
        # print(output.shape)

        # Add dnn.
        # The prediction shape is (batch_size * seq_length, 4). [x_delta,
        # y_delta, not_end_prob, end_prob]
        output = self.fc0(output)
        delta_pred = tf.reshape(self.fc1(output), (-1, 4))
        # print(delta_pred.shape)

        # states will be used to pass at every step to the model while training
        return delta_pred


def loss_func(real, pred):
    """The loss function.

    Args:
        real: The real value.
        pred: The pred value.

    Returns:
        The loss.
    """
    # resolve real data.
    real = np.reshape(real, (-1, 3))

    # (x_delta, y_delta)
    delta_real = real[:, 0: 2]
    delta_pred = pred[:, 0: 2]

    # (not_end_prob, end_prob)
    end_flag_real = real[:, 2].astype(np.int32)
    end_flag_pred = pred[:, 2: 4]

    loss = tf.losses.mean_squared_error(delta_real, delta_pred) + \
           tf.losses.sparse_softmax_cross_entropy(end_flag_real, end_flag_pred)

    return loss


first_ink = []  # The first quick draw ink. Only train one quick draw at now.


def train(model, dataset):
    """Train model.

    Args:
        model: The auto-draw model.
        dataset: The dataset generator.
    """
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)

    # Using adam optimizer with default arguments
    optimizer = tf.train.AdamOptimizer()

    for epoch in range(EPOCHS):
        start_time = time.time()

        for batch, data in enumerate(dataset):

            # Only train one quick draw at now.
            if batch == 0:
                ink = np.reshape(data["ink"], (BATCH_SIZE, -1, 3))
                ink = ink[0, :, :]
                ink = np.broadcast_to(ink, (BATCH_SIZE, ink.shape[0],
                                            ink.shape[1]))
                global first_ink
                first_ink = ink

            hidden = model.reset_states()

            if data["ink"].shape[0] != BATCH_SIZE:
                break

            # Get ink input and target.
            ink = np.reshape(data["ink"], (BATCH_SIZE, -1, 3))
            ink_input = first_ink[:, 0: -1]
            ink_target = first_ink[:, 1:]

            with tf.GradientTape() as tape:
                # feeding the hidden state back into the model
                # This is the interesting step
                pred = model(ink_input)
                # print("pred:\n", str(pred))
                loss = loss_func(ink_target, pred)

            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {}'.format(epoch + 1,
                                                         batch,
                                                         loss))

            # Only train 10000 step at now.
            if batch >= 10000:
                break

        # saving (checkpoint) the model every epoch
        if (epoch + 1) % 1 == 0:
            model.save_weights(CHECKPOINT_PATH)

        print('Epoch {} Loss {}'.format(epoch + 1, loss))
        print(
            'Time taken for 1 epoch {} sec\n'.format(time.time() - start_time))

        # Only train 10000 step at now.
        break


def auto_draw(ink_num):
    """Quick draw.

    Args:
        ink_num: The ink count.
    """
    model = Model(UNITS)
    model.build(tf.TensorShape([1, None, 3]))
    model.summary()
    model.load_weights(CHECKPOINT_PATH)

    model.reset_states()

    # First ink.
    inks = np.expand_dims(first_ink[0, 0, :], 0)
    # print(inks)
    ink_num = first_ink.shape[1]
    # print(ink_num)
    input = inks
    input = np.expand_dims(input, 0)
    input = input.astype(np.float32)

    for i in range(ink_num):
        pred = model(input).numpy()

        # Convert (x_delta, y_delta, no_end_prob, end_prob) to (x_delta,
        # y_delta, end_flag)
        pred_ = np.zeros(shape=(1, 3))
        pred_[0, 0: 2] = pred[0, 0: 2]
        if pred[0, 2] < pred[0, 3]:
            pred_[0, 2] = 1.0

        inks = np.concatenate([inks, pred_])
        input = np.expand_dims(pred_, 0)
        input = input.astype(np.float32)

    # Plot quick draw.
    plt.figure()
    plot_quick_draw(np.expand_dims(first_ink[0, :, :], 0)[0], "real_butterfly",
                    1, 2, 1)
    plt.axis('off')
    inks = np.expand_dims(inks, 0)
    plot_quick_draw(inks[0], "pred_butterfly", 1, 2, 2)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    for i in range(10):
        dataset = load_training_data(
            "/tmp/autodraw_data/training.tfrecord-butterfly", BATCH_SIZE)

        """
           dataset = dataset.make_one_shot_iterator()
           features = dataset.next()
           for i in range(8):
               np_ink = features["ink"].numpy()[i, :]
               np_ink = np.reshape(np_ink, (-1, 3))
               plot_quick_draw(np_ink, "airplane")
        """

        model = Model(UNITS)
        model.build(tf.TensorShape([BATCH_SIZE, None, 3]))
        model.summary()
        train(model, dataset)

        auto_draw(50)
