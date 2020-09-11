# -*- encoding: utf-8 -*-
"""
@File    : interation.py.py
@Time    : 2020-06-22 14:46
@Author  : knight
"""

import numpy as np
import preprocess
import tensorflow as tf
from keras.layers import *


# embed_map = Input(shape=(200 + 12, 200))
# print(embed_map)
def auto_interacting(embed_map, d=8, n_attention_head=2):
    """
    实现单层 AutoInt Interacting Layer
    @param embed_map: 输入的embedding feature map, (?, n_feats, n_dim)
    @param d: Q,K,V映射后的维度
    @param n_attention_head: multi-head attention的个数
    """
    assert len(embed_map.shape) == 3, "Input embedding feature map must be 3-D tensor."

    k = embed_map.shape[-1]

    # 存储多个self-attention的结果
    attention_heads = []
    W_Q = []
    W_K = []
    W_V = []

    # 1.构建多个attention
    for i in range(n_attention_head):
        W_Q.append(tf.Variable(tf.random.truncated_normal(shape=(int(k), d)), name="query_" + str(i)))  # k, d
        #         print(np.shape(W_Q))
        W_K.append(tf.Variable(tf.random.truncated_normal(shape=(int(k), d)), name="key_" + str(i)))  # k, d
        W_V.append(tf.Variable(tf.random.truncated_normal(shape=(int(k), d)), name="value_" + str(i)))  # k, d
    #         W_V.append(tf.Variable(tf.random.truncated_normal(shape=(int(k), d)), name="value_"+str(i)))
    #         W_Q.append(name='WQ', shape=(int(k), d), initializer='glorot_uniform',trainable=True)
    #     keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    # print(W_Q)

    for i in range(n_attention_head):
        # 映射到d维空间
        embed_q = tf.matmul(embed_map, W_Q[i])  # ?, 39, d
        embed_k = tf.matmul(embed_map, W_K[i])  # ?, 39, d
        embed_v = tf.matmul(embed_map, W_V[i])  # ?, 39, d

        # 计算attention
        energy = tf.matmul(embed_q, tf.transpose(embed_k, [0, 2, 1]))  # ?, 39, 39
        attention = tf.nn.softmax(energy)  # ?, 39, 39

        attention_output = tf.matmul(attention, embed_v)  # ?, 39, d
        attention_heads.append(attention_output)

    # 2.concat multi head
    multi_attention_output = Concatenate(axis=-1)(attention_heads)  # ?, 39, n_attention_head*d
    #     print(multi_attention_output)

    # 3.ResNet
    w_res = tf.Variable(tf.random.truncated_normal(shape=(k, d * n_attention_head)),
                        name="w_res_" + str(i))  # k, d*n_attention_head
    output = Activation("relu")(multi_attention_output + tf.matmul(embed_map, w_res))  # ?, 39, d*n_attention_head)

    return output


def build_interaction(x0, n_layers):
    xl = x0
    for i in range(n_layers):
        xl = auto_interacting(xl, d=8, n_attention_head=2)

    return xl


# data_path = '/Users/knight/Desktop/GodClassDetection_0704/trainset/train'
# # metrics = preprocess.get_metrics(data_path)
# metrics = preprocess.get_metrics_up()
#
# # 参数设置
# EMBEDDING_DIM = 200
# MAX_SEQUENCE_LENGTH = 200
# MAX_JACCARD_LENGTH = 30
# INC_BATCH_SIZE = 80000
#
# BASE_DIR = ''
# W2V_MODEL_DIR = '/Users/knight/Desktop/GodClassDetection_0704/embedding_model/new_model6.bin'
# TRAIN_SET_DIR = '/Users/knight/Desktop/GodClassDetection_0704/trainset'  # 直接改成自己的路径
#
# tokenizer = preprocess.get_tokenizer(TRAIN_SET_DIR)
# all_word_index = tokenizer.word_index
# embedding_matrix = preprocess.get_embedding_matrix(all_word_index, W2V_MODEL_DIR, dim=EMBEDDING_DIM)
# mn_datas = preprocess.get_data(TRAIN_SET_DIR + '/train', maxlen=MAX_SEQUENCE_LENGTH, tokenizer=tokenizer,
#                                embedding_matrix=embedding_matrix)
#
# embed_map = np.concatenate((metrics, mn_datas), axis=1)
# # print(embed_map.shape)
# # print(embed_map)
#
#
# # 构建3层interacting layer
# autoint_layer = build_interaction(embed_map, 3)
# print(autoint_layer.shape)  # (1620, 212, 16)
# # print(autoint_layer[0])
# #
# # autoint_layer = Flatten()(autoint_layer)
# # print(autoint_layer.shape)
# # print(autoint_layer)
# # output_layer = Dense(1, activation="sigmoid")(autoint_layer)
# # print(output_layer)
