# -*- encoding: utf-8 -*-
"""
@File    : fine_tune.py.py
@Time    : 2020-05-14 20:23
@Author  : knight
"""

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import time

import preprocess2  # 引入自定义的preprocess.py,所以preprocess.py不是主函数
from keras.models import load_model
from sklearn import metrics

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

localtime = time.asctime(time.localtime(time.time()))
print("begin time", localtime)

# # 从JSON文件中加载模型
# with open(r'/Users/knight/Desktop/GodClassDetection-master-mao-new/trained_model/pre_training_model.json', 'r') as file:
#     model_json = file.read()
#
# # 加载模型
# new_model = model_from_json(model_json)
# new_model.load_weights('model.json.h5')

# 加载预训练的模型
model_path = '/Users/knight/Desktop/GodClassDetection/trained_model/pre_training_model.h5'

# model_path = '/Users/knight/Desktop/GodClassDetection-master-mao-new/trained_model/pre_training_lstm_att_model.h5'
model = load_model(model_path)
# model.summary()

EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 50
MAX_JACCARD_LENGTH = 30
INC_BATCH_SIZE = 80000

BASE_DIR = ''
# W2V_MODEL_DIR = '/Users/knight/Desktop/GodClassDetection/embedding_model/new_model6.bin'
W2V_MODEL_DIR = '/Users/knight/Desktop/GodClassDetection/embedding_model/new_model6_nltk.bin'
TRAIN_SET_DIR = '/Users/knight/Desktop/GodClassDetection/trainset'  # 直接改成自己的路径
FULL_MN_DIR = TRAIN_SET_DIR

tokenizer = preprocess2.get_tokenizer(FULL_MN_DIR)
all_word_index = tokenizer.word_index
embedding_matrix = preprocess2.get_embedding_matrix(all_word_index, W2V_MODEL_DIR, dim=EMBEDDING_DIM)

acc_list = []
loss_list = []
projects = []
x_train = []
y_train = []
x_test = []
y_test = []
f1_scores = []
data_path = '/Users/knight/Desktop/GodClassDetection/trainset/test'
for project in sorted(os.listdir(data_path)):
    projects.append(project)
    path = data_path + "/" + project
    print("path:", path)
    x_train, y_train = preprocess2.get_xy_train1(TRAIN_SET_DIR + '/finetune1', project, mn_maxlen=MAX_SEQUENCE_LENGTH,
                                                 tokenizer=tokenizer,
                                                 # x_train, y_train = preprocess2.get_xy_train1(data_path, project, mn_maxlen=MAX_SEQUENCE_LENGTH, tokenizer=tokenizer,
                                                 embedding_matrix=embedding_matrix)

    x_test, y_test = preprocess2.get_xy_test1(data_path, project, mn_maxlen=MAX_SEQUENCE_LENGTH, tokenizer=tokenizer,
                                              embedding_matrix=embedding_matrix)
    # x_train.append(x_tr)
    # y_train.append(y_tr)
    # x_test.append(x_te)
    # y_test.append(y_te)

    print('Fine tune model.')

    # 微调
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    # 训练集度量值
    hist = model.fit(x_train, y_train, nb_epoch=10, batch_size=5, verbose=2)  # 1为输出进度条记录，2为每个epoch输出一行记录
    score = model.evaluate(x_train, y_train, verbose=0)

    print("训练集的一些参数：！！！！！！！！！！！！")
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

    # 验证集度量值
    print("\n验证集的一些参数：！！！！！！！！！！！！")
    print("project:", project)
    score1 = model.evaluate(x_test, y_test, verbose=0)
    # print model.metrics_names
    print('Test score:', score1[0])
    print('Test accuracy:', score1[1])
    # input()
    predict_posibility = model.predict(x_test)

    predict_label = []
    godclass_indices = []
    for i, item in enumerate(predict_posibility):
        if item <= 0.5:
            predict_label.append(0)
        else:
            godclass_indices.append(i)
            predict_label.append(1)

    labeld_godclasses = []
    for i, item in enumerate(y_test):
        if item == 1:
            labeld_godclasses.append(i)
    print(predict_label)
    precision = metrics.precision_score(y_test, predict_label)
    recall = metrics.recall_score(y_test, predict_label)
    f1_score = metrics.f1_score(y_test, predict_label)

    print('test precision:', precision)
    print('test recall', recall)
    print('test f1 score:', f1_score)
    f1_scores.append(f1_score)
    print("\n")

avg_f1 = sum(f1_scores) / len(f1_scores)
print("average f1: ", avg_f1)
localtime = time.asctime(time.localtime(time.time()))
print("end time", localtime)

# # # 模型保存JSON文件
# # model_json = model.to_json()
# # with open(r'/Users/knight/Desktop/GodClassDetection/trained_model/fine_tune_model.json', 'w') as file:
# #     file.write(model_json)
# #
# # # 保存微调后的模型
# # model.save('/Users/knight/Desktop/GodClassDetection/trained_model/fine_tune_model.h5')
# #
# # model.summary()