# -*-coding:utf-8 -*-
import numpy as np
import tensorflow as tf


# import itertools,time
# import sys, os
# from collections import OrderedDict
# from copy import deepcopy
# from time import time
# import matplotlib.pyplot as plt
# import pickle
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def log_dir_init(fan_in, fan_out, topics=50):
    return tf.log((1.0 / topics) * tf.ones([fan_in, fan_out]))


tf.reset_default_graph()


class VAE(object):
    """
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, sess, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        # 激活函数 softplus 可看作relu的平滑版
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        print('Learning Rate:', self.learning_rate)

        # tf Graph input
        # 模型的输入（数据）
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        self.keep_prob = tf.placeholder(tf.float32)

        # 隐藏层维度（主题个数）
        self.h_dim = int(network_architecture["n_z"])
        self.a = 1 * np.ones((1, self.h_dim)).astype(np.float32)
        self.mu2 = tf.constant((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        self.var2 = tf.constant((((1.0 / self.a) * (1 - (2.0 / self.h_dim))).T +
                                 (1.0 / (self.h_dim * self.h_dim)) * np.sum(1.0 / self.a, 1)).T)

        self._create_network()
        self._create_loss_optimizer()
        if sess!=None:
            self.sess = sess
        # init = tf.initialize_all_variables()
        #
        # self.sess = tf.InteractiveSession()
        # self.sess.run(init)

    def _create_network(self):
        self.network_weights = self._initialize_weights(**self.network_architecture)
        # 编码端(从x到z,但不是直接到z，而是获得z分布的均值和方差，然后获得z)
        # 通过网络获得z(隐藏主题向量)的均值和方差的对数值(方差的对数比方差更好用线性网络拟合，不需要额外的激活函数)
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(self.network_weights["weights_recog"],
                                      self.network_weights["biases_recog"])

        n_z = self.network_architecture["n_z"]
        # 从标准正态分布采样
        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)
        # 通过参数变化获得z的分布的采样结果
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        # 方差
        self.sigma = tf.exp(self.z_log_sigma_sq)

        # 解码端(从z到x，主题到文档（大小为词库大小）)
        self.x_reconstr_mean = \
            self._generator_network(self.z, self.network_weights["weights_gener"])

        print(self.x_reconstr_mean)

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.get_variable('h1', [n_input, n_hidden_recog_1]),
            'h2': tf.get_variable('h2', [n_hidden_recog_1, n_hidden_recog_2]),
            'out_mean': tf.get_variable('out_mean', [n_hidden_recog_2, n_z]),
            'out_log_sigma': tf.get_variable('out_log_sigma', [n_hidden_recog_2, n_z])}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h2': tf.Variable(xavier_init(n_z, n_hidden_gener_1))}

        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network)
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        # dropout层
        layer_do = tf.nn.dropout(layer_2, self.keep_prob)

        z_mean = tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, weights['out_mean']),
                                                     biases['out_mean']))
        z_log_sigma_sq = \
            tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, weights['out_log_sigma']),
                                                biases['out_log_sigma']))

        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, z, weights):
        self.layer_do_0 = tf.nn.dropout(tf.nn.softmax(z), self.keep_prob)
        # 文档的主题概率分布，主题的词概率分布，二者相乘，求得文档中词的概率
        x_reconstr_mean = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.add(
            tf.matmul(self.layer_do_0, weights['h2']), 0.0)))
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        self.x_reconstr_mean += 1e-10

        # 有点类似于交叉熵损失
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(self.x_reconstr_mean), 1)  # /tf.reduce_sum(self.x,1)

        # 分布的损失（强制逼近标准正态）
        latent_loss = 0.5 * (tf.reduce_sum(tf.div(self.sigma, self.var2), 1) + \
                             tf.reduce_sum(tf.multiply(tf.div((self.mu2 - self.z_mean), self.var2),
                                                       (self.mu2 - self.z_mean)), 1) - self.h_dim + \
                             tf.reduce_sum(tf.log(self.var2), 1) - tf.reduce_sum(self.z_log_sigma_sq, 1))

        self.cost = tf.reduce_mean(reconstr_loss) + tf.reduce_mean(latent_loss)  # average over batch

        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.99).minimize(self.cost)

    def partial_fit(self, X):
        # 获得优化器，损失，以及解码层网络参数（主题下的词的概率分布）
        opt, cost, emb = self.sess.run((self.optimizer, self.cost, self.network_weights['weights_gener']['h2']),
                                       feed_dict={self.x: X, self.keep_prob: .4})
        return cost, emb

    def test(self, X):
        """Test the model and return the lowerbound on the log-likelihood.
        对数似然的边分下界为损失函数，损失最终需要平摊到每个词上
        """
        cost = self.sess.run((self.cost), feed_dict={self.x: np.expand_dims(X, axis=0), self.keep_prob: 1.0})
        return cost

    def topic_prop(self, X):
        """heta_ is the topic proportion vector. Apply softmax transformation to it before use.
        获得文档的主题
        """
        theta_ = self.sess.run((self.z), feed_dict={self.x: np.expand_dims(X, axis=0), self.keep_prob: 1.0})
        return theta_

    def test1(self, X, sess):
        """Test the model and return the lowerbound on the log-likelihood.
        对数似然的边分下界为损失函数，损失最终需要平摊到每个词上
        """
        cost = sess.run((self.cost), feed_dict={self.x: np.expand_dims(X, axis=0), self.keep_prob: 1.0})
        return cost

    def topic_prop1(self, X, sess):
        """heta_ is the topic proportion vector. Apply softmax transformation to it before use.
        获得文档的主题
        """
        theta_ = sess.run((self.z), feed_dict={self.x: np.expand_dims(X, axis=0), self.keep_prob: 1.0})
        return theta_
