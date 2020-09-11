# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import time

import keras
import matplotlib.pyplot as plt
import preprocess  # 引入自定义的preprocess.py,所以preprocess.py不是主函数
from keras.layers import Dense, Dropout, Input
from keras.layers import LSTM, Masking
from keras.models import Model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

localtime = time.asctime(time.localtime(time.time()))
print("begin time", localtime)

EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 50
MAX_JACCARD_LENGTH = 30
INC_BATCH_SIZE = 80000

BASE_DIR = ''
W2V_MODEL_DIR = '/Users/knight/Desktop/GodClassDetection-master-mao-new/embedding_model/new_model6.bin'
TRAIN_SET_DIR = '/Users/knight/Desktop/GodClassDetection-master-mao-new/trainset'  # 直接改成自己的路径

FULL_MN_DIR = TRAIN_SET_DIR
# TRAINED_MODEL_WEIGHT_DIR = '/Users/knight/Desktop/GodClassDetection-master-mao-new/trained_model/' + '/pre_training_weights'
# TRAINED_MODEL_JSON_DIR = '/Users/knight/Desktop/GodClassDetection-master-mao-new/trained_model/' + '/pre_training_json'
# TRAINED_MODEL_LOG_DIR = '/Users/knight/Desktop/GodClassDetection-master-mao-new/trained_model/' + '/pre_training_log'
# TRAINED_MODEL_FIG_DIR = '/Users/knight/Desktop/GodClassDetection-master-mao-new/trained_model/' + '/pre_training_fig'

tokenizer = preprocess.get_tokenizer(FULL_MN_DIR)
all_word_index = tokenizer.word_index
embedding_matrix = preprocess.get_embedding_matrix(all_word_index, W2V_MODEL_DIR, dim=EMBEDDING_DIM)

acc_list = []
loss_list = []

test_path = TRAIN_SET_DIR + '/test'

# trained_weight_path = TRAINED_MODEL_WEIGHT_DIR + '.h5'
# trained_json_path = TRAINED_MODEL_JSON_DIR + '.json'
# trained_model_log = TRAINED_MODEL_LOG_DIR + '.txt'
# trained_moedl_fig = TRAINED_MODEL_FIG_DIR + '.png'
# if os.path.exists(trained_weight_path):  # 如果存在.h5的文件，就修改名字，加上str(i)
#     for i in range(10):
#         trained_weight_path = TRAINED_MODEL_WEIGHT_DIR + '_' + str(i) + '.h5'
#         trained_json_path = TRAINED_MODEL_JSON_DIR + '_' + str(i) + '.json'
#         trained_model_log = TRAINED_MODEL_LOG_DIR + '_' + str(i) + '.txt'
#         trained_moedl_fig = TRAINED_MODEL_FIG_DIR + '_' + str(i) + '.png'
#         if os.path.exists(trained_weight_path) == False:
#             break

# # project = 'manual-ant-1.8.2'  # D:\项目\codesmell\GodClassDetection-master\trainset\test
x_train, y_train = preprocess.get_xy_train(TRAIN_SET_DIR + '/train', tokenizer=tokenizer, mn_maxlen=MAX_SEQUENCE_LENGTH,
# x_train, y_train = preprocess.get_xy_train(TRAIN_SET_DIR + '/finetune', tokenizer=tokenizer, mn_maxlen=MAX_SEQUENCE_LENGTH,
                                           embedding_matrix=embedding_matrix)

print('Training model.')


# 模型定义输入，矩阵，格式
method_a = Input(shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), name='method_a')  # 50*200
metric_a = Input(shape=(12,), name='metric_a')

masking_layer = Masking(mask_value=0, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
lstm_share = LSTM(output_dim=2, activation='sigmoid', init='uniform')  # LSTM的输出层，激活函数，初始化
# gru_share = GRU(output_dim=2, activation='sigmoid', init='uniform')  # LSTM的输出层，激活函数，初始化


embedding_a = masking_layer(method_a)  # 嵌入层
lstm_a = lstm_share(embedding_a)
# gru_a = gru_share(embedding_a)
dense_share2 = Dense(12, activation='tanh', init='uniform')  # 全连接层
mtrdense_a = dense_share2(metric_a)

# m_j_merged_a = keras.layers.concatenate([lstm_a, mtrdense_a], axis=-1)
m_j_merged_a = keras.layers.concatenate([gru_a, mtrdense_a], axis=-1)
dense1_a = Dense(4, activation='tanh', init='zero')(m_j_merged_a)
total_dropout = Dropout(0.6)(dense1_a)
total_output = Dense(1, activation='sigmoid', name='output')(total_dropout)  # 输出层
model_final = Model(inputs=[method_a, metric_a], output=total_output)

# 优化
sgd = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
model_final.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model_final.fit(x_train, y_train, nb_epoch=10, batch_size=5, verbose=1)
score = model_final.evaluate(x_train, y_train, verbose=0)


# 模型保存JSON文件
model_json = model_final.to_json()
with open(r'/Users/knight/Desktop/GodClassDetection-master-mao-new/trained_model/pre_training_gru_model.json', 'w') as file:
    file.write(model_json)

# 保存预训练的模型
model_final.save('/Users/knight/Desktop/GodClassDetection-master-mao-new/trained_model/pre_training_gru_model.h5')
# model_final.save('/Users/knight/Desktop/GodClassDetection-master-mao-new/trained_model/pre_training_fine_tune_model.h5')
model_final.summary()

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

# json_string = model_final.to_json()
# splited_path = os.path.split(trained_json_path)
#
# if not os.path.exists(splited_path[0]):  # 如果路径不存在，就创建该路径
#     os.makedirs(splited_path[0])
# with open(trained_json_path, 'w') as f:  # 写入文件
#     f.write(json_string)
# model_final.save_weights(trained_weight_path)  # 保存参数

localtime = time.asctime(time.localtime(time.time()))
print("endding time", localtime)

avg_accuracy = sum(acc_list) / len(acc_list)
avg_loss = sum(loss_list) / len(loss_list)

print("average accuracy: ", avg_accuracy)
print("average loss: ", avg_loss)
