"""
输入有交互的预训练
"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os
import time

import keras
import matplotlib.pyplot as plt
import preprocess  # 引入自定义的preprocess.py,所以preprocess.py不是主函数
import interaction1

from keras.layers import Input,concatenate,Flatten,Dense,Dropout,LSTM, Masking
from keras.models import Model
from keras.layers import *
from tensorflow.keras.utils import plot_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

localtime = time.asctime(time.localtime(time.time()))
print("begin time", localtime)

'''
分类器训练过程：
一.输入1：textual input：attributes and methods
1.类中文本信息经预处理(embedding)之后，把文本信息转换成数值信息，将以词向量（输入形式为50*200矩阵）的形式输入数据屏蔽层(masking,mask_value=0)然后进入LSTM中
2.LSTM 层激活函数为sigmoid 函数,输出维度为2,并对该层权重做均匀分布(uniform)初始化
二.输入2：numerical input (code metrics)
1.将从类中提取出的12个度量值输入全连接层(dense),分类器可以在有监督的情况下对输入进行迭代训练,逐步调整出与训练集标签最匹配的参数组合
2.Dense层激活函数为tanh 函数,输出维度为12,并对本层权重做均匀分布(uniform)初始化
三.合并两部分
1.两部分数据会经由融合层(merge)以向量拼接(concatenate,axis=−1)的形式合并
2.再由一层全连接层（Dense）映射到最终的Sigmoid 输出层,此间的全连接层激活函数为tanh 函数,输出维度为4,权重初始化为全零矩阵(zero)
3.输出层（Output）激活函数为sigmoid 函数,输出维度为1.选取的模型损失函数(loss function)为binary_crossentropy 函数,优化器(optimizer)为adam 自适应方法,
  迭代次数(epoch)为10 次,批尺寸(batch_size)为5.
keras各层简介：
1.Dense层：常用的全连接层，所实现的运算是output = activation(dot(input, kernel)+bias)。其中activation是逐元素计算的激活函数，
  kernel是本层的权值矩阵，bias为偏置向量，只有当use_bias=True才会添加。
2.Masking层：keras.layers.core.Masking(mask_value=0.0)，使用给定的值对输入的序列信号进行“屏蔽”，用以定位需要跳过的时间步
  对于输入张量的时间步，即输入张量的第1维度（维度从0开始算，见例子），如果输入张量在该时间步上都等于mask_value，则该时间步将在模型接下来的所有层（只要支持masking）被跳过（屏蔽）。
  例子：考虑输入数据x是一个形如(samples,timesteps,features)的张量，现将其送入LSTM层。因为你缺少时间步为3和5的信号，所以你希望将其掩盖。
        这时候应该：赋值x[:,3,:] = 0.，x[:,5,:] = 0.
        在LSTM层之前插入mask_value=0.的Masking层
3.Merge层：keras.layers.Concatenate(axis=-1)该层接收一个列表的同shape张量，并返回它们的按照给定轴相接构成的向量。axis=-1 按列拼接 axis=0 按行拼接
'''



# 参数设置
EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 50
MAX_JACCARD_LENGTH = 30
INC_BATCH_SIZE = 80000

BASE_DIR = ''
# W2V_MODEL_DIR = '/Users/knight/Desktop/GodClassDetection-master-mao-new/embedding_model/new_model6.bin'
W2V_MODEL_DIR = '/Users/knight/Desktop/GodClassDetection/embedding_model/new_model6_nltk.bin'
TRAIN_SET_DIR = '/Users/knight/Desktop/GodClassDetection/trainset'  # 直接改成自己的路径


tokenizer = preprocess.get_tokenizer(TRAIN_SET_DIR)
all_word_index = tokenizer.word_index
embedding_matrix = preprocess.get_embedding_matrix(all_word_index, W2V_MODEL_DIR, dim=EMBEDDING_DIM)

epochs = 50
acc_list = []
loss_list = []



x_train, y_train = preprocess.get_xy_train(TRAIN_SET_DIR + '/data', tokenizer=tokenizer, mn_maxlen=MAX_SEQUENCE_LENGTH,
# x_train, y_train = preprocess.get_xy_train(TRAIN_SET_DIR + '/finetune', tokenizer=tokenizer, mn_maxlen=MAX_SEQUENCE_LENGTH,
                                           embedding_matrix=embedding_matrix)
# print("x_train:", x_train[0])
# print("y_train:", y_train[0])


# 模型定义输入，矩阵，格式
print('Training model......')

# 输入1：文本信息
method_a = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), name='method_a')  # 200*200
# 输入2：度量值信息输入
metric_a = Input(shape=(12,), name='metric_a')
# 输入3：交互信息
att_a = Input(shape=(MAX_SEQUENCE_LENGTH + 12, EMBEDDING_DIM), name='interaction_a')  # 212*200
# att_input = Input(shape=(200 + 12, 200))

# print("method_a:", method_a)
# print("method_a_shape:", method_a.shape)
# print("metric_a:", metric_a)
# print("metric_a_shape:", metric_a.shape)
# print("interaction_a:", att_a)
# print("interaction_a_shape:", att_a.shape)


# 处理文本信息：LSTM
masking_layer = Masking(mask_value=0, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
# LSTM层 输出维度为2，激活函数为sigmoid，并对该层权重做均匀分布(uniform)初始化
lstm_share = LSTM(output_dim=2, activation='sigmoid', init='uniform')
embedding_a = masking_layer(method_a)  # 嵌入层
lstm_a = lstm_share(embedding_a)

# 处理度量值信息：直接Dense
dense_share2 = Dense(12, activation='tanh', init='uniform')  # 全连接层
mtrdense_a = dense_share2(metric_a)

# 处理交互信息：
# 构建3层interacting layer
for _ in range(3):
    att_output = interaction1.InteractingLayer(8, 2, True)(att_a)
# print(att_output.shape)
# print(att_output)

att_output = Conv1D(128, 1, padding="same", activation='tanh')(att_output)
# print(att_output)
att_output = Conv1D(128, 1, activation='tanh')(att_output)
# print(att_output)
att_output = Conv1D(128, 1, activation='tanh')(att_output)
# print(att_output)
att_output = Flatten()(att_output)
# print(att_output)

interaction_a = Dense(128, activation='tanh')(att_output)
# print(interaction_a)


# print("lstm_a:", lstm_a)
# print("lstm_a_shape:", lstm_a.shape)
# print("mtrdense_a:", mtrdense_a)
# print("mtrdense_a_shape:", mtrdense_a.shape)
# print("interaction_share:", interaction_a)
# print("interaction_share_shape:", interaction_a.shape)

# 合并三个输入

m_j_merged_a = concatenate([lstm_a, mtrdense_a, interaction_a], axis=-1)
# 再过一层Dense层，激活函数为tanh 函数,输出维度为4,权重初始化为全零矩阵(zero)
dense1_a = Dense(4, activation='tanh', init='zero')(m_j_merged_a)
# 按照0.6的概率将神经网络中的数据暂时丢弃，正常取值是在0.2-0.5之间
total_dropout = Dropout(0.6)(dense1_a)
total_output = Dense(1, activation='sigmoid', name='output')(total_dropout)  # 输出层
model_final = Model(inputs=[method_a, metric_a, att_a], output=total_output)

plot_model(model_final, "GodClass.png")

# 优化
sgd = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
model_final.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model_final.fit(x_train, y_train, nb_epoch=epochs, batch_size=5, verbose=1)
score = model_final.evaluate(x_train, y_train, verbose=0)
model_final.summary()


# print("lstm_a:", lstm_a)
# print("mtrdense_a_shape:", mtrdense_a)
# print("lstm_a:", lstm_a.shape)
# print("mtrdense_a_shape:", mtrdense_a.shape)
# print("interaction_a:", interaction_a)
# print("interaction_a_shape:", interaction_a.shape)



# 模型保存JSON文件
model_json = model_final.to_json()
with open(r'/Users/knight/Desktop/GodClassDetection/trained_model/pre_training_model_int_50.json', 'w') as file:
    file.write(model_json)

# 保存预训练的模型
model_final.save('/Users/knight/Desktop/GodClassDetection/trained_model/pre_training_model_int_50.h5')


# 绘制训练损失图像
hist_dict = hist.history
print(hist_dict.keys())
loss = hist.history['loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'b', label='Training loss')  # b代表蓝色实线
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
