#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Train a value network(Deep Q-Learning network) to play grid world game.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    19-1-24 上午9:17
"""

# Common libs.
import os
import time
import random

# 3rd-part libs.
import numpy as np
import tensorflow as tf
import reinforcement_learning.grid_world as grid_world
import matplotlib.pyplot as plt


class Config(object):
    """Global configs.

    Attributes:
        actions_num: The probable action numbers.
        batch_size: The batch size.
        update_freq: The steps to update vars.
        discount_fac: The discount factor.
        num_episodes: The total episodes number.
        h_size: The hidden layer size.
        tvars_update_eta: The learning rate of target vars updater.
        path: The model saver path.
        update_freq: The update frequency.
        start_random: The probability of random start action.
        end_random: The probability of random end action.
        annealing_step: The total drop steps from start_random to end_random.
        pre_train_steps: The previous training steps of actually selected
        action.
        max_epi_actions: The max actions of every episodes.
        load_model: If load previous model.
    """
    actions_num = 4
    batch_size = 32
    discount_fac = 0.99
    num_episodes = 10000
    h_size = 512
    tvars_update_eta = 0.001
    path = "../model_saver/reinforcement_learning/value_net_model/"
    update_freq = 4
    p_start_random = 1.0
    p_end_random = 0.1
    annealing_step = 10000
    pre_train_steps = 10000
    max_epi_actions = 50
    load_model = False


class DeepQNetwork(object):
    """The deep Q network(value network)
    """

    def __init__(self, h_size):
        """Initialize the deep Q network. Infer the graph of tensorflow.

        Args:
            h_size: The hidden layer size.
        """
        self.scalar_input = tf.placeholder(dtype=tf.float32,
                                           shape=[None, 84 * 84 * 3],
                                           name="scalar_input")
        self.__infer(h_size)

    def __infer(self, h_size):
        """Infer the tensorflow graph of the network.

        Args:
            h_size: The hidden layer size.

        Returns:

        """
        self.image_in = tf.reshape(self.scalar_input, shape=[-1, 84, 84, 3])
        # Define first convolution layer.
        conv1 = tf.contrib.layers.conv2d(
            inputs=self.image_in,
            num_outputs=32,
            kernel_size=[8, 8],
            stride=[4, 4],
            padding="VALID",
            biases_initializer=None  # None biases.
        )
        # Define the second convolution layer.
        conv2 = tf.contrib.layers.conv2d(
            inputs=conv1,
            num_outputs=64,
            kernel_size=[4, 4],
            stride=[2, 2],
            padding="VALID",
            biases_initializer=None,  # None biases.
        )
        # Define the third convolution layer.
        conv3 = tf.contrib.layers.conv2d(
            inputs=conv2,
            num_outputs=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding="VALID",
            biases_initializer=None,  # None biases.
        )
        # Define the forth convolution layer.
        conv4 = tf.contrib.layers.conv2d(
            inputs=conv3,
            num_outputs=h_size,
            kernel_size=[7, 7],
            stride=[1, 1],
            padding="VALID",
            biases_initializer=None,  # None biases.
        )
        # Split the conv4 to static environment value and dynamic value by
        # selecting action.
        stream_dy, stream_st = tf.split(conv4, num_or_size_splits=2, axis=3)
        stream_dy = tf.layers.flatten(stream_dy)
        stream_st = tf.layers.flatten(stream_st)
        stream_dy_w = tf.Variable(
            tf.random_normal(shape=[h_size // 2, Config.actions_num]))
        stream_st_w = tf.Variable(tf.random_normal(shape=[h_size // 2, 1]))
        advantage = tf.matmul(stream_dy, stream_dy_w)
        value = tf.matmul(stream_st, stream_st_w)

        # Calc the q out.
        self.q_out = value + tf.subtract(advantage, tf.reduce_mean(advantage,
                                                                   reduction_indices=1,
                                                                   keep_dims=True))
        # Select action with max q out.
        self.predict = tf.argmax(self.q_out, axis=1)
        # Define the target q.
        self.target_q = tf.placeholder(shape=None, dtype=tf.float32,
                                       name="target_q")
        # Define agent action.
        self.agent_action = tf.placeholder(shape=None, dtype=tf.int32,
                                           name="agent_action")
        actions_onehot = tf.one_hot(self.agent_action, Config.actions_num,
                                    dtype=tf.float32)
        # Calc predicted q.
        pred_q = tf.reduce_sum(tf.multiply(self.q_out, actions_onehot),
                               axis=1)
        # print("pred_q shape:", str(tf.shape(pred_q)))

        # Optimize the loss between target q and predicted q.
        self.sq_cost = tf.reduce_mean(tf.square(self.target_q - pred_q))
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.update_model = trainer.minimize(self.sq_cost)


class ExperienceBuffer(object):
    """Train the value network with experience relay policy. Create
    experience buffer to save the experience samples of agent.
    """

    def __init__(self, buffer_size=50000):
        """Initialize the experience buffer.

        Args:
            buffer_size: The buffer size.
        """
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        """Add one experience into buffer.

        Args:
            experience: The experience.
        """
        if len(self.buffer) + len(experience) > self.buffer_size:
            self.buffer[0: len(self.buffer) + len(experience) -
                           self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        """Get training samples.

        Args:
            size: The sample size.
        """
        return np.reshape(np.array(random.sample(self.buffer, size)),
                          newshape=[size, 5])


def update_target_graph(vars, eta):
    """Construct the graph of updater that update target DQN vars by main DQN
    vars with a very low learning eta.

    Args:
        vars: All tf vars. The first half are main DQN vars. The second half
        are target DQN vars.
        eta: learning eta.

    Returns:
        op_holder: The update operations holder.
    """
    len_vars = len(vars)
    op_holder = []
    for index, var in enumerate(vars[0: len_vars // 2]):
        """
        For example, update b to a with eta.
        b_new = b_old + (a - b_old) * eta.
        """
        # print("var:", str(var))
        # print("vars[index + len_vars // 2]:", str(vars[index + len_vars // 2]))
        op_holder.append(vars[index + len_vars // 2].assign(
            vars[index + len_vars // 2].value() + eta * (
                    var.value() - vars[index + len_vars // 2].value()
            )))
    return op_holder


def update_target(op_holder, sess):
    """Update target DQN vars.

    Args:
        op_holder: The update operations holder.
        sess: The session.
    """
    for op in op_holder:
        sess.run(op)


def flatten_state(states):
    return np.reshape(states, 21168)  # 84 * 84 * 3


def train():
    """Train deep q network.

    Returns:
    """
    game_env = grid_world.GameEnv(size_x=5, size_y=5)
    main_qn = DeepQNetwork(Config.h_size)
    target_qn = DeepQNetwork(Config.h_size)
    init = tf.global_variables_initializer()
    trainable_vars = tf.trainable_variables()
    print("trainable_vars len", len(trainable_vars))
    target_updater = update_target_graph(trainable_vars,
                                         Config.tvars_update_eta)

    experience_buffer = ExperienceBuffer()
    # Define current probability of random action.
    current_p = Config.p_start_random
    # Calc the drop step from p_start_random to p_end_random.
    step_drop = (Config.p_start_random - Config.p_end_random) / \
                Config.annealing_step
    # Define r_list to save all rewards of episodes.
    r_list = []
    # Save current step.
    total_steps = 0

    # Define model saver.
    saver = tf.train.Saver()
    if not os.path.exists(Config.path):
        os.mkdir(Config.path)

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter("./", sess.graph)
        # writer.close()

        if Config.load_model == True:
            print("Loading model...")
            ckpt = tf.train.get_checkpoint_state(Config.path)
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        sess.run(init)  # Initialize all train vars.
        update_target(target_updater, sess)  # Update target_qn.

        for i in range(Config.num_episodes):
            # Save experience of current episode.
            episode_buffer = ExperienceBuffer()
            s = game_env.reset()
            s = flatten_state(s)
            d = False  # The done state.
            r_all = 0.0  # Save total rewards of current episode.

            for j in range(Config.max_epi_actions):
                # print("total_steps:", total_steps)
                # print("current_p:", current_p)
                if total_steps < Config.pre_train_steps or random.random() < \
                        current_p:
                    action = np.random.randint(0, 4)
                    # print("random action ", action)
                else:
                    action = sess.run([main_qn.predict],
                                      feed_dict={main_qn.scalar_input: [s]})[0]
                    # print("predict action ", action)
                s_new, r, d = game_env.step(action)
                s_new = flatten_state(s_new)
                episode_buffer.add(
                    np.reshape(np.array([s, action, r, s_new, d]),
                               newshape=[1, 5]))
                total_steps += 1
                r_all += r
                s = s_new

                # If total steps is bigger than pre_train_steps, start training
                # the networks.
                # print("total_steps:", total_steps)
                if total_steps > Config.pre_train_steps:
                    if current_p > Config.p_end_random:
                        current_p = current_p - step_drop
                    if total_steps % Config.update_freq == 0:
                        train_batch = experience_buffer.sample(
                            Config.batch_size)
                        # Use main q network to select the action.
                        predict_action = sess.run([main_qn.predict],
                                                  feed_dict={
                                                      main_qn.scalar_input:
                                                          np.vstack(train_batch[
                                                                    :, 3])
                                                  })[0]
                        # print(predict_action)
                        # Use target q network to calc the q_out.
                        q_out = sess.run([target_qn.q_out],
                                         feed_dict={
                                             target_qn.scalar_input:
                                                 np.vstack(train_batch[:, 3])
                                         })[0]
                        # print(np.shape(q_out))
                        # Use the predict action of main q network to select the
                        # q out.
                        double_q = q_out[range(len(train_batch)),
                                         predict_action]
                        # print(np.shape(double_q))
                        # Calc the target q_out with discounted factor.
                        target_q = train_batch[:,
                                   2] + Config.discount_fac * double_q
                        # print(np.shape(target_q))
                        # Update the main q network.
                        cost, _ = sess.run([main_qn.sq_cost,
                                          main_qn.update_model],
                                 feed_dict={
                                     main_qn.scalar_input:
                                         np.vstack(train_batch[:, 0]),
                                     main_qn.agent_action:
                                         train_batch[:, 1],
                                     main_qn.target_q: target_q
                                 })
                        # print("total_steps: {0}, cost:{1}".format(
                        #     total_steps, cost))
                        # Let target q network learns the main q network with a
                        # very low learning rate.
                        update_target(target_updater, sess)
                if d == True:
                    break
            # Add episode buffer into total experience buffer.
            experience_buffer.add(episode_buffer.buffer)
            # Save the total episode reward into r_list.
            r_list.append(r_all)
            # Show rewards per 25 episodes.
            if (i + 1) % 25 == 0:
                print("r_list len", str(len(r_list)))
                print("r_list[-25:]:", r_list[-25:])
                print("Episode {0}, average rewards of last 25 episodes"
                      " is {1}".format(i + 1, np.mean(r_list[-25:])))
            # Save the model every 1000 episodes.
            if (i + 1) > 0 and (i + 1) % 1000 == 0:
                saver.save(sess, Config.path + "/model-" + str(i + 1) + ".ckpt")
                print("Model checkpoint{0} saved".format(i + 1))
        saver.save(sess, Config.path + "/model-" + str(Config.num_episodes)
                   + ".ckpt")
        return r_list


def plot_rewards(reward_list):
    """Plot the reward change of all episodes.

    Args:
        reward_list: The reward list.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(0, len(reward_list), 1)
    ax.plot(x, reward_list, label='reward')
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title("The best reward:{0}".format(np.max(reward_list)))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    r_list = train()
    plot_rewards(r_list)