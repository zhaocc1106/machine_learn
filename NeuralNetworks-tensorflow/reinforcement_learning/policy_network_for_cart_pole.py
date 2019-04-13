#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Train a policy network to play cart pole game.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    19-1-14 上午9:11
"""

# Common libs.
import time

# 3rd-part libs.
import gym
import tensorflow as tf
import numpy as np

env = gym.make("CartPole-v0")


class Config(object):
    """The policy network config.

    Attributes:
        hidden_size: The hidden layer size.
        batch_size: The batch size.
        learning_rate: The learning rate.
        D: The dimension of reward.
        gamma: The attenuation coefficient of discount reward.
    """

    hidden_size = 50
    batch_size = 25
    learning_rate = 0.1
    D = 4
    gamma = 0.99


def random_policy():
    """Use random policy to train CartPole.

    Returns:
        mean_reward_sum: The mean reward sum of 10 episodes.
    """
    env.reset()
    reward_sum = 0.0
    mean_reward_sum = 0.0
    random_episodes = 0
    while random_episodes < 10:
        env.render()
        observation, reward, done, info = env.step(np.random.randint(0, 2))
        print("observation:", str(observation))
        print("reward:", str(reward))
        print("done:", str(done))
        print("info:", str(info))
        reward_sum += reward

        if done:
            random_episodes += 1
            print("Reward for this episodes:", str(reward_sum))
            mean_reward_sum += reward_sum
            reward_sum = 0.0
            env.reset()
    mean_reward_sum /= 10
    print("The mean reward_sum of 10 episodes:", str(mean_reward_sum))
    return mean_reward_sum


class PolicyNetwork(object):

    def __init__(self):
        """Init the policy network and infer the tensorflow graph of the
        network."""
        self.__inference()

    def __inference(self):
        """Infer the tensorflow graph of the network."""
        # Define the placeholder of cart pole observation.
        self.__observation = tf.placeholder(dtype=tf.float32,
                                            shape=[None, Config.D],
                                            name="observation")
        # Define the hidden layer.
        w1 = tf.get_variable(name="w1",
                             shape=[Config.D, Config.hidden_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(self.__observation, w1))  # No biases.
        w2 = tf.get_variable(name="w2",
                             shape=[Config.hidden_size, 1],
                             initializer=tf.contrib.layers.xavier_initializer())

        # Define the optimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.learning_rate)
        self.__w1_grads = tf.placeholder(dtype=tf.float32, name="w1_grads")
        self.__w2_grads = tf.placeholder(dtype=tf.float32, name="w2_grads")

        # Define the action 1 probability.
        self.__probability = tf.nn.sigmoid(tf.matmul(layer1, w2))

        # Define that the input_y is (1 - Action)
        self.__input_y = tf.placeholder(dtype=tf.float32, shape=[None, 1],
                                        name="input_y")
        print(self.__input_y)

        # Define that the advantages is the discounted reward of one action.
        self.__advantages = tf.placeholder(dtype=tf.float32,
                                           name="reward_signal")
        print(self.__advantages)

        # Define that the log_like is the log of action[0, 1] probability.
        log_like = tf.log(self.__input_y * (self.__input_y - self.__probability)
                          + (1 - self.__input_y) * (self.__input_y +
                                                    self.__probability))
        print(log_like)

        # Define the loss.
        loss = -tf.reduce_mean(log_like * self.__advantages)

        # Define update_grads.
        self.__train_vars = tf.trainable_variables()
        self.__update_grads = optimizer.apply_gradients(grads_and_vars=zip([
            self.__w1_grads, self.__w2_grads],
            self.__train_vars))  # Minimize the loss.

        # Define the new grads.
        self.__new_grads = tf.gradients(ys=loss, xs=self.__train_vars)

    def __discounted_rewards(self, reward):
        """Calculate the discounted rewards for every action.

        Args:
            reward: The rewards of the complete actions.

        Returns:
            discounted_r: The discounted rewards of every action.
        """
        discounted_r = np.zeros_like(reward)
        running_add = 0.0  # The reward of final action is 0.
        for i in reversed(range(reward.size)):
            """
            For example, if there are total of 10 actions.
            discounted_r10 = 0 * gamma + 0
            discounted_r9 = discounted_r10 * gamma + r9
            discounted_r8 = discounted_r9 * gamma + r8 = 
            discounted_r10 * gamma² + discounted_r9 * gamma + discounted_r8
            ...
            
            So, discounted_r1 = r1 + r2 * gamma + r3 * gamma² + r4 * gamma³ + ...
            """
            running_add = running_add * Config.gamma + reward[0]
            discounted_r[i] = running_add
        return discounted_r

    def train(self, total_episodes, target_reward):
        """Train the policy network taking the target_reward as a goal.

        Args:
            total_episodes: The total episodes to train policy network.
            target_reward: The target reward.
        """
        xs = []  # The list of episodes observations.
        ys = []  # The list of input_y(1 - Action).
        rs = []  # The list of episodes rewards.
        reward_sum = 0.0
        episode_num = 1

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            rendering = False
            observation = env.reset()

            # Initialize the grad_buffer.
            grad_buffer = sess.run(self.__train_vars)
            for (i, grad) in enumerate(grad_buffer):
                grad_buffer[i] = grad * 0.0

            while episode_num <= total_episodes:
                # When mean reward of batch > 100, show the rendering.
                if reward_sum / Config.batch_size > 190 or rendering == True:
                    env.render()
                    rendering = True

                x = np.reshape(observation, newshape=[1, Config.D])
                probability = sess.run(fetches=self.__probability,
                                       feed_dict={self.__observation: x})
                action = 1 if np.random.uniform() < probability else 0
                y = 1 - action
                xs.append(x)
                ys.append(y)

                # Run one step of cat pole.
                observation, reward, done, info = env.step(action)
                reward_sum += reward
                rs.append(reward_sum)

                # If done.
                if done:
                    episode_num += 1
                    # Reshape the xs ys rs to vertical.
                    epx = np.vstack(xs)
                    epy = np.vstack(ys)
                    epr = np.vstack(rs)
                    xs, ys, rs = [], [], []

                    # Calc the discounted reward for this episode.
                    # print("epr:", str(epr))
                    discounted_reward = self.__discounted_rewards(epr)
                    # print("discounted_reward:", str(discounted_reward))

                    # Standardize the discounted reward.
                    discounted_reward -= np.mean(discounted_reward)
                    discounted_reward /= np.std(discounted_reward)

                    new_grads = sess.run(fetches=self.__new_grads,
                                         feed_dict={self.__observation: epx,
                                                    self.__input_y: epy,
                                                    self.__advantages:
                                                        discounted_reward})
                    for i, grad in enumerate(new_grads):
                        grad_buffer[i] += grad  # Calc the total grad  of batch.

                    if episode_num % Config.batch_size == 0:
                        # Update variables every batch.
                        sess.run(fetches=self.__update_grads,
                                 feed_dict={self.__w1_grads: grad_buffer[0],
                                            self.__w2_grads: grad_buffer[1]})

                        for (i, grad) in enumerate(grad_buffer):
                            grad_buffer[i] = grad * 0.0

                        print("Episode {0}, average rewards: {1}".format(
                            episode_num, reward_sum / Config.batch_size))

                        if reward_sum / Config.batch_size >= target_reward:
                            print("Train end with %d episode" % episode_num)
                            break
                        reward_sum = 0.0

                    # When done, should reset environment.
                    observation = env.reset()


if __name__ == "__main__":
    # random_policy()
    network = PolicyNetwork()
    network.train(10000, 200)
