#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""The PTB(Penn Three Bank) language model realized by LSTM(Long short term
memory).

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/12/21 9:23
"""

# common libs
import time

# 3rd-part libs
import tensorflow as tf
import numpy as np
import RNN.RNN_ptb_model.reader as reader


class PTBInput(object):
    """The PTB model data input class.

    The input_data and targets format is as the following example:
    raw_data_len: 20
    batch_size: 4
    batch_len: 5 (batch_len = raw_data_len // batch_size)
    num_step: 2
    epoch_size: 2 (epoch_size = (batch_len - 1) // num_step)

         ▏←     batch len    →▕
    ▁    *    *    *    *    *    *
    ↑
         *    *    *    *    *    *
   batch
   size  *    *    *    *    *    *

    ↓    *    *    *    *    *    *
    ▔
         <-------------------->
              <------------------->
          ↑    | ↑
          The  |The
          input|target
          of   |of
          one  |one
          step |step

    Attributes:
        batch_size: The inputs batch size.
        num_steps: The number of unrolls.
        epoch_size: The epoch size.
        input_data: The input data of every batch.
        targets: The targets corresponding with input_data.
    """

    def __init__(self, config, data, name=None):
        """Inits PTBInput with config, data and name

        Attributes:
            config: The config.
            data: The raw data.
            name: The input name.
        """
        self.batch_size = batch_size =  config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size - 1) // num_steps)
        self.input_data, self.targets = reader.ptb_producer(data, batch_size,
                                                            num_steps, name)


class PTBModel(object):
    """"The PTB model.

    Attributes:
        __input: The input model.
        __initial_state: The initial state.
        __cost: The cost.
        __final_state: The final state.
        __lr: The learning rate.
        __train_op: The training operation.
    """

    def __init__(self, is_training, config, input_):
        """Inits PTBModel with is_training, config, and input.

        Attributes:
            is_training: If is training.
            config: The config.
            input_: The input model.
        """
        self.__input = input_
        batch_size = config.batch_size
        num_steps = config.num_steps
        hidden_size = config.hidden_size # The hidden layer size.
        vocab_size = config.vocab_size
        # Define lstm cell layer.
        def lstm_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size,
                                                 forget_bias=0.0,
                                                 state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            # Define dropout layer.
            def attn_cell():
                return tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell(),
                                                      output_keep_prob=config.keep_prob)
        else:
            attn_cell = lstm_cell
        # Define multiple lstm cell.
        cell = tf.nn.rnn_cell.MultiRNNCell(cells=
                                            [attn_cell() for _ in range(
                                                config.num_layers)],
                                            state_is_tuple=True)

        self.__initial_state = cell.zero_state(batch_size=batch_size,
                                               dtype=tf.float32)

        # Define embedding layer. Theses operation don't work well in gpu.
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(name="embedding",
                                        shape=[vocab_size, hidden_size],
                                        dtype=tf.float32)
            embed = tf.nn.embedding_lookup(embedding, input_.input_data)
        if is_training and config.keep_prob < 1:
            embed = tf.nn.dropout(embed, keep_prob=config.keep_prob)

        # Define outputs layer.
        outputs = []
        state = self.__initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    # steps use the same variables.
                    tf.get_variable_scope().reuse_variables()
                (cell_out, state) = cell(inputs=embed[:, time_step, :],
                                         state=state)
                outputs.append(cell_out)
        # print(len(outputs))
        # print(outputs[0])

        # Concatenate the outputs.
        concat_out = tf.reshape(tensor=tf.concat(outputs, 1), shape=[-1, hidden_size])

        # Define the logits.
        softmax_w = tf.get_variable(shape=[hidden_size, vocab_size],
                                    dtype=tf.float32,
                                    name="softmax_weights")
        softmax_b = tf.get_variable(shape=[vocab_size],
                                    dtype=tf.float32,
                                    name="softmax_bias")
        logits = tf.matmul(concat_out, softmax_w) + softmax_b

        # Define the cost.
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            logits=[logits],
            targets=[tf.reshape(input_.targets, [-1])],
            weights=[tf.ones([batch_size * num_steps], dtype=tf.float32)]
        )
        self.__cost = cost = tf.reduce_sum(loss) / batch_size
        # tf.summary.scalar(self.__cost)
        self.__final_state = state

        if not is_training:
            return

        # Define learning rate.
        self.__lr = tf.Variable(0.0, trainable=False)
        # tf.summary.scalar(self.__lr)
        train_vals = tf.trainable_variables()
        print("train_vals:")
        print(train_vals)
        # Use gradient clipping to normalize the variables.
        grads, _ = tf.clip_by_global_norm(t_list=tf.gradients(cost,
                                                              train_vals),
                                          clip_norm=config.max_grad_norm)

        # Define the optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.__lr)
        self.__train_op = optimizer.apply_gradients(grads_and_vars=zip(grads,
                                                              train_vals),
                                  global_step=tf.train.get_or_create_global_step())

        # Define the update operation of learning rate.
        self.__new_lr = tf.placeholder(dtype=tf.float32, shape=[])
        self.__lr_update = tf.assign(self.__lr, self.__new_lr)

    def assign_lr(self, session, new_lr):
        """Update learning rate.

        Args:
            session: Current session.
            new_lr: The new learning rate.
        """
        session.run([self.__lr_update], feed_dict={self.__new_lr: new_lr})

    @property
    def input(self):
        return self.__input

    @property
    def initial_state(self):
        return self.__initial_state

    @property
    def cost(self):
        return self.__cost

    @property
    def final_state(self):
        return self.__final_state

    @property
    def lr(self):
        return self.__lr

    @property
    def train_op(self):
        return self.__train_op


class SmallConfig(object):
    """The small model config.

    Attributes:
        init_scale: The weights initial value.
        learning_rate: The initial learning rate.
        max_grad_norm: The max clip norm param of clip_by_global_norm.
        num_layers: The number of hidden layers.
        hidden_size: The size of hidden layer.
        max_epoch: The max epoch using the initial learning rate.
        max_max_epoch: The max training epochs.
        keep_prob: The keep prob param of dropout.
        lr_decay: The decay speed.
        batch_size: The batch size.
        num_steps: The number of unrolls.
        vocab_size: The vocabulary size.
    """
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    hidden_size = 100
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    num_steps = 20
    vocab_size = 10000


class MediumConfig(object):
    """The medium model config.

    Attributes:
        init_scale: The weights initial value.
        learning_rate: The initial learning rate.
        max_grad_norm: The max clip norm param of clip_by_global_norm.
        num_layers: The number of hidden layers.
        hidden_size: The size of hidden layer.
        max_epoch: The max epoch using the initial learning rate.
        max_max_epoch: The max training epochs.
        keep_prob: The keep prob param of dropout.
        lr_decay: The decay speed.
        batch_size: The batch size.
        num_steps: The number of unrolls.
        vocab_size: The vocabulary size.
    """
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    num_steps = 35
    vocab_size = 10000


class LargeConfig(object):
    """The large model config.

    Attributes:
        init_scale: The weights initial value.
        learning_rate: The initial learning rate.
        max_grad_norm: The max clip norm param of clip_by_global_norm.
        num_layers: The number of hidden layers.
        hidden_size: The size of hidden layer.
        max_epoch: The max epoch using the initial learning rate.
        max_max_epoch: The max training epochs.
        keep_prob: The keep prob param of dropout.
        lr_decay: The decay speed.
        batch_size: The batch size.
        num_steps: The number of unrolls.
        vocab_size: The vocabulary size.
    """
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    num_steps = 35
    vocab_size = 10000


class TestConfig(object):
    """The test model config just for test the total model.

    Attributes:
        init_scale: The weights initial value.
        learning_rate: The initial learning rate.
        max_grad_norm: The max clip norm param of clip_by_global_norm.
        num_layers: The number of hidden layers.
        hidden_size: The size of hidden layer.
        max_epoch: The max epoch using the initial learning rate.
        max_max_epoch: The max training epochs.
        keep_prob: The keep prob param of dropout.
        lr_decay: The decay speed.
        batch_size: The batch size.
        num_steps: The number of unrolls.
        vocab_size: The vocabulary size.
    """
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    num_steps = 2
    vocab_size = 10000


def run_epoch(session, model, train_op=None, verbose=False):
    """Run epoch.

    Args:
        session: The session.
        model: The model.
        train_op: The train operation.
        verbose: If log is verbose.

    Returns:
        perplexity: The perplexity.
    """
    begin_time = time.time()
    costs = 0.0
    iters = 0
    # Calc the initial state.
    state = session.run(model.initial_state)

    # Define the fetches.
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state
    }
    if train_op is not None:
        fetches["eval_op"] = train_op

    print("epoch_size:%d" % model.input.epoch_size)
    step = model.input.epoch_size // 10
    # Define the feed_dict.
    for epoch in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in  enumerate(model.initial_state):
            # print("i:", str(i))
            # print("c:", str(c))
            # print("h:", str(h))
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if epoch % step == 0 or epoch == model.input.epoch_size:
            print("Small epoch {0}, perplexity:{1}, speed:{2} word per second."
                  .format(epoch,
                          np.exp(costs / iters),
                          iters * model.input.batch_size /
                          (time.time() - begin_time)))

    return np.exp(costs / iters)

if __name__ == "__main__":
    raw_data = reader.ptb_raw_data("./simple-examples/data")
    train_data, valid_data, test_data, vocabulary_size = raw_data
    print("len of train_data:\n", str(len(train_data)))
    print("len of valid_data:\n", str(len(valid_data)))
    print("len of test_data:\n", str(len(test_data)))
    print("vocabulary_size:\n", str(vocabulary_size))
    config = MediumConfig()
    eval_config = MediumConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(minval=-config.init_scale,
                                                    maxval=config.init_scale,
                                                    dtype=tf.float32)

        # Define the train model.
        with tf.name_scope("Train"):
            train_input = PTBInput(config=MediumConfig, data=train_data,
                                 name="TrainingInput")
            with tf.variable_scope(name_or_scope="Model", reuse=None,
                                   initializer=initializer):
                train_model = PTBModel(is_training=True, config=MediumConfig,
                                     input_=train_input)
        # Define the validate model.
        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data,
                                 name="ValidInput")
            with tf.variable_scope(name_or_scope="Model", reuse=True,
                                   initializer=initializer):
                valid_model = PTBModel(is_training=False, config=config,
                                     input_=valid_input)
        # Define the test model.
        with tf.name_scope("Test"):
            test_input = PTBInput(config=eval_config, data=test_data,
                                 name="TestInput")
            with tf.variable_scope(name_or_scope="Model", reuse=True,
                                   initializer=initializer):
                test_model = PTBModel(is_training=False, config=eval_config,
                                     input_=test_input)

        # train_writer = tf.summary.FileWriter("./")
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir = "../../out/model_saver/RNN_ptb_model") as sess:
            # tf.global_variables_initializer().run(session=sess)
            tf.train.start_queue_runners(sess=sess)
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0)
                print("lr_decay:")
                print(lr_decay)
                print("new learning:")
                print(config.learning_rate * lr_decay)
                train_model.assign_lr(sess, config.learning_rate * lr_decay)
                print("Big epoch {0}, learning rate:{1}"
                      .format(i + 1,  sess.run(train_model.lr)))
                train_perplexity = run_epoch(session=sess,
                                             model=train_model,
                                             train_op=train_model.train_op,
                                             verbose=True)

                print("Big epoch{0}, train_perplexity:{1}".format(i + 1,
                                                                  train_perplexity))

                valid_perplexity = run_epoch(session=sess,
                                             model=valid_model,
                                             train_op=None,
                                             verbose=True)
                print("Big epoch{0}, valid_perplexity:{1}".format(i + 1,
                                                                  valid_perplexity))

            test_perplexity = run_epoch(session=sess,
                                        model=test_model,
                                        train_op=None,
                                        verbose=True)
            print("test_perplexity:{0}".format(test_perplexity))
