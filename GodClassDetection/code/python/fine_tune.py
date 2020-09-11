# -*- encoding: utf-8 -*-
"""
@File    : fine_tune.py.py
@Time    : 2020-05-14 20:23
@Author  : knight
"""

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import time

import keras
import matplotlib.pyplot as plt
import preprocess  # 引入自定义的preprocess.py,所以preprocess.py不是主函数
import interaction1
from interaction1 import InteractingLayer

from keras.layers import Input, concatenate, Flatten, Dense, Dropout, LSTM, Masking
from keras.models import Model, load_model
from keras.layers import *
from tensorflow.keras.utils import plot_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

localtime = time.asctime(time.localtime(time.time()))
print("begin time", localtime)
print(keras.__version__)

# 加载预训练的模型
model_path = '/Users/knight/Desktop/GodClassDetection/trained_model/pre_training_model.h5'
# model_path = '/Users/knight/Desktop/GodClassDetection/trained_model/pre_training_lstm_att_model.h5'

interaction = InteractingLayer()
model = load_model(model_path, custom_objects={'InteractingLayer': interaction})
# model.summary()

EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 200
MAX_JACCARD_LENGTH = 30
INC_BATCH_SIZE = 80000

BASE_DIR = ''
# W2V_MODEL_DIR = '/Users/knight/Desktop/GodClassDetection/embedding_model/new_model6.bin'
W2V_MODEL_DIR = '/Users/knight/Desktop/GodClassDetection/embedding_model/new_model6_nltk.bin'
TRAIN_SET_DIR = '/Users/knight/Desktop/GodClassDetection/trainset'  # 直接改成自己的路径
FULL_MN_DIR = TRAIN_SET_DIR

tokenizer = preprocess.get_tokenizer(FULL_MN_DIR)
all_word_index = tokenizer.word_index
embedding_matrix = preprocess.get_embedding_matrix(all_word_index, W2V_MODEL_DIR, dim=EMBEDDING_DIM)

acc_list = []
loss_list = []

print("11111111111111111")
x_train, y_train = preprocess.get_xy_train(TRAIN_SET_DIR + '/finetune', tokenizer=tokenizer,
                                           mn_maxlen=MAX_SEQUENCE_LENGTH,
                                           embedding_matrix=embedding_matrix)

print('Fine tune model.')

# 微调
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, nb_epoch=10, batch_size=5, verbose=1)
score = model.evaluate(x_train, y_train, verbose=0)

# 模型保存JSON文件
model_json = model.to_json()
with open(r'/Users/knight/Desktop/GodClassDetection/trained_model/fine_tune_model.json', 'w') as file:
    # with open(r'/Users/knight/Desktop/GodClassDetection-master-mao-new/trained_model/fine_tune_gru_model.json', 'w') as file:
    # with open(r'/Users/knight/Desktop/GodClassDetection-master-mao-new/trained_model/fine_tune_lstm_att_model.json', 'w') as file:
    file.write(model_json)

# 保存微调后的模型
model.save('/Users/knight/Desktop/GodClassDetection/trained_model/fine_tune_model.h5')
# model.save('/Users/knight/Desktop/GodClassDetection-master-mao-new/trained_model/fine_tune_gru_model.h5')
# model.save('/Users/knight/Desktop/GodClassDetection-master-mao-new/trained_model/fine_tune_lstm_att_model.h5')

model.summary()

# 绘制训练损失图像
hist_dict = hist.history
print(hist_dict.keys())
loss = hist.history['loss']
# val_loss = hist.history['binary_crossentropy']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'b', label='Training loss')  # b代表蓝色实线
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制训练精度图像

plt.clf()

acc = hist.history['acc']

plt.plot(epochs, acc, 'b', label='Training acc')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print('train mse:', score[0])
print('train accuracy:', score[1])
loss_list.append(score[0])
acc_list.append(score[1])

localtime = time.asctime(time.localtime(time.time()))
print("endding time", localtime)

avg_accuracy = sum(acc_list) / len(acc_list)
avg_loss = sum(loss_list) / len(loss_list)

print("average accuracy: ", avg_accuracy)
print("average loss: ", avg_loss)
