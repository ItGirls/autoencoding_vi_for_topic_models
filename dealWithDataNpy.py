#!/usr/local/bin/python3
# -*-coding:utf-8 -*-
"""
@Date  : 2020/7/28 下午7:01

@Author : zhutingting

@Desc : ==============================================

Blowing in the wind. ===

# ======================================================

@Project : autoencoding_vi_for_topic_models

@FileName: dealWithDataNpy.py

@Software: PyCharm
"""
import pickle

import numpy as np

from run import onehot

dataset_tr = 'data/20news_clean/test.txt.npy'
vocab = 'data/20news_clean/vocab.pkl'
vocab = pickle.load(open(vocab, 'r'))
# print(vocab)
vocab_size = len(vocab)
if __name__ == "__main__":

    arr = np.load(dataset_tr,allow_pickle=True,encoding="latin1")
    print(arr[0])
    print(len(arr[0]))
    print(type(arr))
    print(arr)
    data_tr = np.array([onehot(doc.astype('int'), vocab_size) for doc in arr if np.sum(doc) != 0])
    print(data_tr[0])

    # print(arr.size)