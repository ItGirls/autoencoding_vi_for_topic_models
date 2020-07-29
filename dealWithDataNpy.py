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
import numpy as np

dataset_tr = 'data/20news_clean/train.txt.npy'

if __name__ == "__main__":

    arr = np.load(dataset_tr,allow_pickle=True,encoding="latin1")
    print(arr[0])
    print(len(arr[0]))
    print(type(arr))
    print(arr)
    # print(arr.size)