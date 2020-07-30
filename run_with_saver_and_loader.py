#!/usr/bin/python
# -*-coding:utf-8 -*-
import numpy as np
import tensorflow as tf
# import itertools, time
# import sys, os
# from collections import OrderedDict
# from copy import deepcopy
# from time import time
# import matplotlib.pyplot as plt
import pickle
import sys, getopt
from models import prodlda_, nvlda

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

'''-----------Data--------------'''


def onehot(data, min_length):
    # Count number of occurrences of each value in array of non-negative ints.
    # 实际就是bag of words 表达的数据样本
    return np.bincount(data, minlength=min_length)


# 一、加载数据部分（数据为文档集，每篇文档中的词已经由其在词库中的对应位置表达）
# 训练数据
dataset_tr = 'data/20news_clean/train.txt.npy'
data_tr = np.load(dataset_tr, allow_pickle=True, encoding="latin1")
# 测试数据
dataset_te = 'data/20news_clean/test.txt.npy'
data_te = np.load(dataset_te, allow_pickle=True, encoding="latin1")
# 词库
vocab = 'data/20news_clean/vocab.pkl'
vocab = pickle.load(open(vocab, 'r'))
# print(vocab)
vocab_size = len(vocab)
print("the number of words in vocab is ", vocab_size)
# --------------convert to one-hot representation------------------
# 转变数据为词袋表达，每个样本长度即为词库大小，每个位置对应该词在该样本中出现的次数
print('Converting data to one-hot representation')
data_tr = np.array([onehot(doc.astype('int'), vocab_size) for doc in data_tr if np.sum(doc) != 0])
data_te = np.array([onehot(doc.astype('int'), vocab_size) for doc in data_te if np.sum(doc) != 0])
# --------------print the data dimentions--------------------------
print('Data Loaded')
print('Dim Training Data', data_tr.shape)
print('Dim Test Data', data_te.shape)
'''-----------------------------'''

'''--------------Global Params---------------'''
# 设置全局参数
n_samples_tr = data_tr.shape[0]
n_samples_te = data_te.shape[0]
docs_tr = data_tr
docs_te = data_te
batch_size = 200
learning_rate = 0.002
network_architecture = \
    dict(n_hidden_recog_1=100,  # 1st layer encoder neurons
         n_hidden_recog_2=100,  # 2nd layer encoder neurons
         n_hidden_gener_1=data_tr.shape[1],  # 1st layer decoder neurons, 即词库大小
         n_input=data_tr.shape[1],  # MNIST data input (img shape: 28*28), 即词库大小
         n_z=50)  # dimensionality of latent space, 即主题个数

'''-----------------------------'''

'''--------------Netowrk Architecture and settings---------------'''


def make_network(layer1=100, layer2=100, num_topics=50, bs=200, eta=0.002):
    tf.reset_default_graph()
    network_architecture = \
        dict(n_hidden_recog_1=layer1,  # 1st layer encoder neurons
             n_hidden_recog_2=layer2,  # 2nd layer encoder neurons
             n_hidden_gener_1=data_tr.shape[1],  # 1st layer decoder neurons
             n_input=data_tr.shape[1],  # MNIST data input (img shape: 28*28)
             n_z=num_topics)  # dimensionality of latent space
    batch_size = bs
    learning_rate = eta
    return network_architecture, batch_size, learning_rate


'''--------------Methods--------------'''


def create_minibatch(data):
    # 获得batch数据
    rng = np.random.RandomState(10)

    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        ixs = rng.randint(data.shape[0], size=batch_size)
        yield data[ixs]


def train(model_path, network_architecture, minibatches, type='prodlda', learning_rate=0.001,
          batch_size=200, training_epochs=100, display_step=5):
    tf.reset_default_graph()
    vae = ''
    if type == 'prodlda':
        vae = prodlda_.VAE(network_architecture,
                           learning_rate=learning_rate,
                           batch_size=batch_size)
    elif type == 'nvlda':
        vae = nvlda.VAE(network_architecture,
                        learning_rate=learning_rate,
                        batch_size=batch_size)

    with tf.Session(config=tf_config) as sess:

        tf.summary.FileWriter('graph', sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        emb = 0
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples_tr / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = minibatches.next()
                # Fit training using batch data
                cost, emb = vae.partial_fit(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples_tr * batch_size

                if np.isnan(avg_cost):
                    print(epoch, i, np.sum(batch_xs, 1).astype(np.int), batch_xs.shape)
                    print(
                        'Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                    # return vae,emb
                    sys.exit()

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), \
                      "cost=", "{:.9f}".format(avg_cost))
            #     path = saver.save(sess, model_path)  #
            #     print("Save model checkpoint to {}\n".format(path))
            # if epoch == training_epochs-1:
            #     path = saver.save(sess, model_path)  #
            #     print("Save model checkpoint to {}\n".format(path))
        path = saver.save(sess, model_path)  #
        print("Save model checkpoint to {}\n".format(path))
        return vae, emb


def test(model_path, network_architecture, type='prodlda', learning_rate=0.001,
         batch_size=200):
    tf.reset_default_graph()
    vae =""
    if type == 'prodlda':
        vae = prodlda_.VAE(network_architecture,
                           learning_rate=learning_rate,
                           batch_size=batch_size)
    elif type == 'nvlda':
        vae = nvlda.VAE(network_architecture,
                        learning_rate=learning_rate,
                        batch_size=batch_size)
    # saver = tf.train.import_meta_graph(model_path + '.meta')  # 加载图结构
    # vae = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量

    with tf.Session(config=tf_config) as sess:
        saver = tf.train.Saver()
        print(model_path)
        saver.restore(sess, model_path)
        saver.restore(sess, tf.train.latest_checkpoint("checkpoint/"))

        calcPerp1(vae,sess)
        # calcTopic1(vae,sess)


    # graph = tf.get_default_graph()
    # sess = tf.Session(config=tf_config)
    # with graph.as_default():
    #     vae = ""
    #     if type == 'prodlda':
    #         vae = prodlda_.VAE(network_architecture,
    #                            learning_rate=learning_rate,
    #                            batch_size=batch_size)
    #     elif type == 'nvlda':
    #         vae = nvlda.VAE(network_architecture,
    #                         learning_rate=learning_rate,
    #                         batch_size=batch_size)
    #     saver = tf.train.Saver()
    #     saver.restore(sess, model_path)
    #     # cost = []
    #     # for doc in docs_te:
    #     #     doc = doc.astype('float32')
    #     #     n_d = np.sum(doc)
    #     #     c = sess.run((vae.cost), feed_dict={vae.x: np.expand_dims(doc, axis=0), vae.keep_prob: 1.0})
    #     #     cost.append(c / n_d)
    #     # print('The approximated perplexity is: ', (np.exp(np.mean(np.array(cost)))))
    #
    #     emb = sess.run(vae.network_weights['weights_gener']['h2'])
    #     print_top_words(emb, zip(*sorted(vocab.items(), key=lambda x: x[1]))[0])
    #     calcPerp(vae)
    #     calcTopic(vae)
    #
    # with tf.Session() as sess:
    #     saver = tf.train.import_meta_graph(model_path + ".meta")
    #     saver.restore(sess, model_path)
    #     vae = tf.get_default_graph()
    #     # ave.get_tensor_by_name("topic_word")
    #     # calcPerp(ave)
    #     # calcTopic(ave)
    #     vae_z= vae.get_tensor_by_name("z:0")
    #     cost = []
    #     for doc in docs_te:
    #         doc = doc.astype('float32')
    #         topic = sess.run((vae_z), feed_dict={vae.x: np.expand_dims(doc, axis=0), vae.keep_prob: 1.0})
    #     print('The approximated perplexity is: ', (np.exp(np.mean(np.array(cost)))))


def print_top_words(beta, feature_names, n_top_words=10):
    # 打印每个主题下前top个词
    print('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        print(" ".join([feature_names[j]
                        for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
    print('---------------End of Topics------------------')


def calcPerp(model):
    # count = 0
    cost = []
    for doc in docs_te:
        doc = doc.astype('float32')
        n_d = np.sum(doc)
        # print(n_d)
        c = model.test(doc)
        # if count ==0:
        #     theta_ = model.topic_prop(doc)
        #     print(theta_)
        # count+=1
        cost.append(c / n_d)
    print('The approximated perplexity is: ', (np.exp(np.mean(np.array(cost)))))


def calcTopic(model):
    # count = 0
    cost = []
    for doc in docs_te:
        doc = doc.astype('float32')
        theta_ = model.topic_prop(doc)
        print('The topic distribution is: ', theta_)
        break

def calcPerp1(model,sess):
    # count = 0
    cost = []
    for doc in docs_te:
        doc = doc.astype('float32')
        n_d = np.sum(doc)
        # print(n_d)
        c = model.test1(doc,sess)
        # if count ==0:
        #     theta_ = model.topic_prop(doc)
        #     print(theta_)
        # count+=1
        cost.append(c / n_d)
    print('The approximated perplexity is: ', (np.exp(np.mean(np.array(cost)))))

def calcTopic1(model,sess):
    # count = 0
    cost = []
    for doc in docs_te:
        doc = doc.astype('float32')
        theta_ = model.topic_prop1(doc,sess)
        print('The topic distribution is: ', theta_)
        break

def main(argv):
    """
    -m prodlda or nvlda
    -f 100   # hidden layer size of encoder1(编码器端第一层隐藏层维度)
    -s 100   # hidden layer size of encoder2(编码器端第二层隐藏层维度)
    -t 50    # number of topics(主题数)
    -b 200   # batch size(batch大小)
    -e 80    # number of epochs to train(epoch大小)
    -r 0.002 # learning rate(学习率)
    """
    m = ''
    f = ''
    s = ''
    t = ''
    b = ''
    r = ''
    e = ''
    train_or_test = ''
    model_path = ''
    try:
        opts, args = getopt.getopt(argv, "hpnm:f:s:t:b:r:,e: o:q:",
                                   ["default=", "model=", "layer1=", "layer2=", "num_topics=", "batch_size=",
                                    "learning_rate=", "training_epochs", "train_or_test", "path="])
        print(r"获取参数成功: ", opts)
    except getopt.GetoptError:
        print(
            'CUDA_VISIBLE_DEVICES=0 python run.py -m <model> -f <#units> -s <#units> -t <#topics> -b <batch_size> -r <learning_rate [0,1] -e <training_epochs>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'CUDA_VISIBLE_DEVICES=0 python run.py -m <model> -f <#units> -s <#units> -t <#topics> -b <batch_size> -r <learning_rate [0,1]> -e <training_epochs>')
            sys.exit()
        elif opt == '-p':
            print('Running with the Default settings for prodLDA...')
            print('CUDA_VISIBLE_DEVICES=0 python run.py -m prodlda -f 100 -s 100 -t 50 -b 200 -r 0.002 -e 100')
            m = 'prodlda'
            f = 100
            s = 100
            t = 50
            b = 200
            r = 0.002
            e = 100
        elif opt == '-n':
            print('Running with the Default settings for NVLDA...')
            print('CUDA_VISIBLE_DEVICES=0 python run.py -m nvlda -f 100 -s 100 -t 50 -b 200 -r 0.005 -e 300')
            m = 'nvlda'
            f = 100
            s = 100
            t = 50
            b = 200
            r = 0.01
            e = 300
        elif opt == "-m":
            m = arg
        elif opt == "-f":
            f = int(arg)
        elif opt == "-s":
            s = int(arg)
        elif opt == "-t":
            t = int(arg)
        elif opt == "-b":
            b = int(arg)
        elif opt == "-r":
            r = float(arg)
        elif opt == "-e":
            e = int(arg)
        elif opt == "-o":
            train_or_test = arg
        elif opt == "-q":
            model_path = arg

    print(train_or_test)
    print(model_path)
    # 设置网络参数、学习率、batch_size等
    network_architecture, batch_size, learning_rate = make_network(f, s, t, b, r)
    print(network_architecture)

    if train_or_test == "train":
        print("start training...")

        # print(opts)
        # 一、训练
        # 1.获得batch数据
        minibatches = create_minibatch(docs_tr.astype('float32'))
        # 2.训练模型
        vae, emb = train(model_path, network_architecture, minibatches, m, training_epochs=e, batch_size=batch_size,
                         learning_rate=learning_rate)
        print_top_words(emb, zip(*sorted(vocab.items(), key=lambda x: x[1]))[0])
        print_top_words(emb, zip(*sorted(vocab.items(), key=lambda x: x[1]))[0])
        calcPerp(vae)
        calcPerp(vae)
        calcTopic(vae)
        calcTopic(vae)
    elif train_or_test == "test":
        # 二、测试
        print("start testing...")
        test(model_path, network_architecture, m, learning_rate=learning_rate, batch_size=batch_size)

        # print(zip(*sorted(vocab.items(), key=lambda x: x[1])))


if __name__ == "__main__":
    main(sys.argv[1:])
