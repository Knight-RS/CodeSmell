# -*- encoding: utf-8 -*-
"""
@File    : attention.py.py
@Time    : 2020-06-11 15:31
@Author  : knight
"""

import preprocess
import os
import numpy as np
from keras import Model
import keras.backend as K
from keras.layers import *
from matplotlib import pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# In[*]
def generate_data():
    TRAIN_SET_DIR = '/Users/knight/Desktop/GodClassDetection/trainset/train1'  # 直接改成自己的路径
    train_x = preprocess.get_metrics(TRAIN_SET_DIR)
    print('train_x[0]:', train_x[0])
    train_y = preprocess.get_labels(TRAIN_SET_DIR)
    print('train_y[0]:', train_y[0])

    TEST_SET_DIR = '/Users/knight/Desktop/GodClassDetection/trainset/train2'  # 直接改成自己的路径
    test_x = preprocess.get_metrics(TEST_SET_DIR)
    test_y = preprocess.get_labels(TEST_SET_DIR)

    print('-------------train datasize:', len(train_x))
    print('train_x:\n', train_x)
    print('train_y:\n', train_y)
    print()
    print('-------------test datasize:', len(test_x))
    print('test_x:\n', test_x)
    print('test_y:\n', test_y)

    return train_x, train_y, test_x, test_y


# In[*]
def build_model(model_type, merge_type, dim):
    K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = Input(shape=(12,), dtype='float32')

    embd = Embedding(100000, dim)(inputs)
    embd = BatchNormalization()(embd)
    Dropout(0.25)
    if model_type != 'basic':
        if model_type == 'att1':
            hidden = attention_block_1(embd, feature_cnt, dim)
        elif model_type == 'att2':
            hidden = attention_block_2(embd, feature_cnt, dim)
        elif model_type == 'att3':
            hidden = attention_block_3(embd, feature_cnt, dim)

        if merge_type == 'avg':
            # hidden = GlobalAveragePooling1D()(hidden)
            hidden = Lambda(lambda x: K.mean(x, axis=1))(hidden)
        else:
            hidden = Flatten()(hidden)
    else:
        hidden = Flatten()(embd)
    outputs = Dense(1, activation='softmax')(hidden)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


# In[*]
def attention_block_1(inputs, feature_cnt, dim):
    h_block = int(feature_cnt * dim / 32 / 2)
    hidden = Flatten()(inputs)
    while (h_block >= 1):
        h_dim = h_block * 32
        hidden = Dense(h_dim, activation='selu', use_bias=True)(hidden)
        h_block = int(h_block / 2)
    attention = Dense(feature_cnt, activation='softmax', name='attention')(hidden)
    attention = RepeatVector(dim)(attention)
    attention = Permute((2, 1))(attention)

    attention_out = Multiply()([attention, inputs])
    return attention_out


# In[*]
def attention_block_2(inputs, feature_cnt, dim):
    a = Permute((2, 1))(inputs)
    a = Reshape((dim, feature_cnt))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(feature_cnt, activation='softmax')(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='attention')(a)
    a = RepeatVector(dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    attention_out = Multiply()([inputs, a_probs])
    return attention_out


# In[*]
def attention_block_3(inputs, feature_cnt, dim):
    a = Flatten()(inputs)
    a = Dense(feature_cnt * dim, activation='softmax')(a)
    a = Reshape((feature_cnt, dim,))(a)
    a = Lambda(lambda x: K.sum(x, axis=2), name='attention')(a)
    a = RepeatVector(dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    attention_out = Multiply()([inputs, a_probs])
    return attention_out


# In[*]
def train(train_x, train_y, test_x, test_y, model_type, epochs, merge_type, dim):
    model = build_model(model_type, merge_type, dim)
    model.fit(train_x, train_y
              , batch_size=64
              , epochs=epochs
              , validation_data=(test_x, test_y)
              )
    if model_type != 'basic':
        att = get_layer_output(model, 'attention', test_x)
        print(att)
        plt.bar(np.arange(12), att)
        print(np.sum(att))


# In[*]
def get_layer_output(model, layer_name, inputs):
    layer = Model(inputs=model.input,
                  outputs=model.get_layer(layer_name).output)
    layer_out = layer.predict(inputs)
    #    print(layer_out[0])
    return np.mean(layer_out, axis=0)


dim = 200  # 嵌入维度
epochs = 1  # 训练轮数
feature_cnt = 12

train_x, train_y, test_x, test_y = generate_data()
print('fsfdvdfbdfgwefsdf')
model_types = ['basic', 'att1', 'att2', 'att3']
model_type = model_types[3]
merge_types = ['avg', 'concat']
merge_type = merge_types[0]
train(train_x, train_y, test_x, test_y, model_type, epochs, merge_type, dim)
print('1111111111111111111111')
