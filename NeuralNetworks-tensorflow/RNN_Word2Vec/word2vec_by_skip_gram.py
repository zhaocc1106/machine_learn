#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""The word to vector algorithm realized by skip-gram.


Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/12/9 16:15
"""

# Common libs.
import collections
import math

# 3rd-part libs.
import numpy as np
import tensorflow as tf
import tools.words_downloader_and_reader as words_downloader_and_reader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data_index = 0  # The data index in text data.
data = []  # The number list of every word.
count = {}  # The dictionary of ```(word, count)```.
dicionary = {}  # The dictionary of ```(word, number)```.
reverse_dictionary = {}  # The dictionary of ```(number, word)```.


def generate_batch(data, batch_size, num_skips, skip_window):
    """Generate batches.

    :param data: The text data.
    :param batch_size: The batch size.
    :param num_skips: The number of samples for every words.
    :param skip_window: The farthest distance that can be reached for every
    words.
    :return:
        batch: The batch words data.
        label: The near words for every words data in batch.
    """
    global data_index
    # Make sure the batch has the total samples for every word.
    assert batch_size % num_skips == 0
    # The max number of samples for every word is 2 * skip_window.
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    """
    The above params meanings is as follow:
    <--------------span---------------->
                   target<--skip_window->
    "The    quick   brown   fox    jumped    over    the    lazy    dog."
    
    The word "brown" samples can be "brown->The", "brown->quick", 
    "brown->fox", "brown->jumped". The max num_skips is 4.
    """
    # Get the initial span buffer data.
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # print(buffer)
    # Get num_skips
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_void = [target]
        # print(targets_to_void)

        for j in range(num_skips):
            # Find the target in span buffer.
            while target in targets_to_void:
                target = np.random.randint(0, span)
            targets_to_void.append(target)
            # print(targets_to_void)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        # Buffer shift left one step.
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


class Network(object):
    """The skip-gram neuron network"""

    def __init__(self, vocabulary_size, batch_size, embedding_size,
                 valid_size, valid_window, num_sampled):
        """The construct function of network.

        :param vocabulary_size: The total vocabulary size.
        :param batch_size: The mini batch size.
        :param embedding_size: The embedding size.
        :param valid_size: The validation data size.
        :param valid_window: The validation data is from valid_window
        with highest frequency.
        :param num_sampled: The negative noise sample size.
        """
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.valid_size = valid_size
        self.num_sampled = num_sampled
        self.valid_samples = np.random.choice(valid_window, valid_size,
                                              replace=False)

        self.train_inputs = tf.placeholder(tf.int32,
                                           shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32,
                                           shape=[self.batch_size, 1])
        self.eta = tf.placeholder(tf.float32)
        valid_dataset = tf.constant(self.valid_samples, dtype=tf.int32)

        # Inference the train operation and train loss.
        self.train_op, self.train_loss = self.__inference()

        # Normalize the embeddings for calc the cosine similarity.
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), axis=1,
                                     keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            self.normalized_embeddings, valid_dataset)
        # Mat multiply result is just cosine similarity because the L2
        # norm of vectors is 1.
        # print(valid_embeddings)
        # print(self.normalized_embeddings)
        self.similarity = tf.matmul(valid_embeddings,
                                    self.normalized_embeddings,
                                    transpose_b=True)

    def __inference(self):
        """Inference the train operation and train loss.

        :return:
            train_op: The train operation.
            train_loss: The train loss.
        """
        # Some operation cannot calc in gpu.
        with tf.device("/cpu:0"):
            self.embeddings = tf.Variable(tf.random_uniform([
                self.vocabulary_size,
                self.embedding_size], -1, 1))
            embeded = tf.nn.embedding_lookup(self.embeddings,
                                             self.train_inputs)

            nce_weights = tf.Variable(tf.truncated_normal(
                shape=[self.vocabulary_size, self.embedding_size],
                stddev=1.0 / math.sqrt(self.embedding_size)))
            nce_biases = tf.Variable(tf.zeros(shape=[self.vocabulary_size]))

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=self.train_labels,
                                             inputs=embeded,
                                             num_sampled=self.num_sampled,
                                             num_classes=self.vocabulary_size))
        optimizer = tf.train.GradientDescentOptimizer(self.eta).minimize(loss)
        return optimizer, loss

    def SGD(self, num_skips, skip_window, eta, epochs, epoch_train_size):
        """Training and test AlexNet CNN using stochastic gradient descent.

        :param num_skips: The number of samples for every words.
        :param skip_window: The farthest distance that can be reached for
        every words.
        :param eta: The learning rate.
        :param epochs: Thr train epochs.
        :param epoch_train_size: The train size of every epoch.

        :return
            final_embeddings: The final normalized embeddings used to calc
            the similarity.

        """
        assert epochs > 0, "epochs wrong."
        assert epoch_train_size > 0, "epoch_train_size wrong."
        # Calc the steps of every epoch.
        every_epoch_steps = epoch_train_size
        # Calc the total steps of total epochs.
        steps = int(epochs * every_epoch_steps)
        print(every_epoch_steps)
        print(steps)

        global data, reverse_dictionary
        session = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        init.run(session=session)
        print("Initialized")
        losses = 0.0
        for step in range(steps):
            batch_inputs, batch_labels = generate_batch(data,
                                                        self.batch_size,
                                                        num_skips,
                                                        skip_window)
            _, train_loss = session.run([self.train_op, self.train_loss],
                                        feed_dict={
                                            self.train_inputs: batch_inputs,
                                            self.train_labels: batch_labels,
                                            self.eta: eta
                                        })
            losses += train_loss

            if (step > 0 and step % 1000 == 0) or step == (steps - 1):
                losses = losses / 1000
                print("Train {0} steps with losses:{1}".format(step,
                                                               losses))
                losses = 0.0

            # Show the words with top-k similarity for every valid words.
            if (step > 0 and step % every_epoch_steps == 0) or step == (steps
                                                                        - 1):
                sim = self.similarity.eval()
                for i in range(self.valid_size):
                    valid_word = reverse_dictionary[self.valid_samples[i]]
                    top_k = 8
                    # 0 is # it self.
                    # print(sim.shape)
                    nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                    log_str = "Nearest to {0}:".format(valid_word)
                    for j in nearest:
                        log_str = "%s, %s" % (log_str,
                                              reverse_dictionary[j])
                    print(log_str)
            final_embeddings = self.normalized_embeddings.eval()
        return final_embeddings


def plot_with_labels(low_dim_embeds, labels):
    """Plot the words vectors with 2-dimensional space.

    :param low_dim_embeds: The lower dimension embeddings.
    :param labels: The labels corresponding with embeddings.
    """
    print("plot_with_labels")
    assert low_dim_embeds.shape[0] >= len(labels), "More labels than " \
                                                   "embeddings."
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embeds[i, :]  # Get the coordinate.
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     ha="right",
                     va="bottom")
    plt.title("Words vector")
    plt.show()


if __name__ == "__main__":
    words_downloader_and_reader.maybe_download("text8.zip")
    words = words_downloader_and_reader.read_data("text8.zip")
    print("words len:\n", len(words))
    vocabulary_size = 50000
    global count, data, dicionary, reverse_dictionary
    count, data, dicionary, reverse_dictionary = \
        words_downloader_and_reader.build_dataset(words, vocabulary_size)
    print("count[:10]:\n", str(count[:10]))
    print("data[:10]:\n", str(data[:10]))
    print([reverse_dictionary[index] for index in data[:10]])
    # print("dictionary:\n", str(dicionary))
    # print("reverse_dictionary:\n", str(reverse_dictionary))
    del words
    # batch, labels = generate_batch(data, 12, 4, 2)
    # for i in range(12):
    #     print(str(reverse_dictionary[batch[i]]) + "->" + str(
    #         reverse_dictionary[labels[i, 0]]))

    network = Network(vocabulary_size=vocabulary_size,
                      batch_size=128,
                      embedding_size=128,
                      valid_size=16,
                      valid_window=100,
                      num_sampled=64)

    final_embeddings = network.SGD(num_skips=2,
                                   skip_window=1,
                                   eta=1.0,
                                   epochs=10,
                                   epoch_train_size=10000)
    # print(final_embeddings)
    if not final_embeddings is None:
        tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
        plot_only = 100
        # PCA dimension reduction.
        low_dim_embeds = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embeds, labels)

