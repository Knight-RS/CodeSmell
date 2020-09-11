# -*- encoding: utf-8 -*-
"""
@File    : fine_tune.py.py
@Time    : 2020-05-14 20:23
@Author  : knight
"""

"""
输入有交互的微调
"""

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import time

import preprocess4
from keras.models import load_model
from sklearn import metrics
from interaction1 import InteractingLayer
import csv
import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

localtime = time.asctime(time.localtime(time.time()))
print("begin time", localtime)


# 加载预训练的模型
model_path = '/home/rsj/GodClassDetection/trained_model/pre_training_model_int_10.h5'

interaction = InteractingLayer()
model = load_model(model_path, custom_objects={'InteractingLayer': interaction})
# model.summary()

EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 50
MAX_JACCARD_LENGTH = 30
INC_BATCH_SIZE = 80000

BASE_DIR = ''
# W2V_MODEL_DIR = '/Users/knight/Desktop/GodClassDetection/embedding_model/new_model6.bin'
W2V_MODEL_DIR = '/home/rsj/GodClassDetection/embedding_model/new_model6_nltk.bin'
TRAIN_SET_DIR = '/home/rsj/GodClassDetection/trainset'  # 直接改成自己的路径
FULL_MN_DIR = TRAIN_SET_DIR

tokenizer = preprocess4.get_tokenizer(FULL_MN_DIR)
all_word_index = tokenizer.word_index
embedding_matrix = preprocess4.get_embedding_matrix(all_word_index, W2V_MODEL_DIR, dim=EMBEDDING_DIM)

epochs = 10
haveInt = 'haveInt'
W2V = 'new_model6_nltk'
proportion = '1:3'

acc_list = []
loss_list = []
projects = []
x_train = []
y_train = []
x_test = []
y_test = []
f1_scores = []

total_y_pre = []
total_y_test = []

f = open(
    '/home/rsj/GodClassDetection/result/' + 'result_' + str(epochs) + '_' + haveInt + '_' + W2V + '_' + proportion + '.csv',
    'w')
writer = csv.writer(f)
writer.writerow(['Precision', 'Recall', 'F1', 'MCC', 'AUC'])


# 评价标准
def eval(y_pre, y_test):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y_pre)):
        total_y_pre.append(y_pre[i])
        total_y_test.append(y_test[i])
        if y_pre[i] >= 0.5:
            if y_test[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if y_test[i] == 1:
                fn += 1
            else:
                tn += 1

    print("tp : ", tp)
    print("tn : ", tn)
    print("fp : ", fp)
    print("fn : ", fn)
    P = tp * 1.0 / (tp + fp)
    R = tp * 1.0 / (tp + fn)
    print("Precision : ", P)
    print("Recall : ", R)
    print("F1 : ", 2 * P * R / (P + R))
    F = 2 * P * R / (P + R)

    a = tp + fp
    b = tp + fn
    c = tn + fp
    d = tn + fn
    print("MCC : ", (tp * tn - fp * fn) / ((a * b * c * d) ** 0.5))
    print("AUC : ", metrics.roc_auc_score(y_test, y_pre))
    M = (tp * tn - fp * fn) / ((a * b * c * d) ** 0.5)
    A = metrics.roc_auc_score(y_test, y_pre)

    writer.writerow([P, R, F, M, A])

    return 2 * P * R / (P + R)


data_path = '/home/rsj/GodClassDetection/trainset/test'
for project in sorted(os.listdir(data_path)):
    projects.append(project)
    path = data_path + "/" + project
    print("path:", path)
    x_train, y_train = preprocess4.get_xy_train(TRAIN_SET_DIR + '/finetune', project, mn_maxlen=MAX_SEQUENCE_LENGTH,
                                                 tokenizer=tokenizer,
                                                 # x_train, y_train = preprocess2.get_xy_train1(data_path, project, mn_maxlen=MAX_SEQUENCE_LENGTH, tokenizer=tokenizer,
                                                 embedding_matrix=embedding_matrix)

    x_test, y_test = preprocess4.get_xy_test(data_path, project, mn_maxlen=MAX_SEQUENCE_LENGTH, tokenizer=tokenizer,
                                              embedding_matrix=embedding_matrix)
    # x_train.append(x_tr)
    # y_train.append(y_tr)
    # x_test.append(x_te)
    # y_test.append(y_te)

    metric = preprocess4.get_metrics_up(project, TRAIN_SET_DIR + '/finetune')
    metric1 = preprocess4.get_metrics_up1(project, data_path)

    # print(len(metric))
    # print(metric.shape)
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    c5 = []
    c6 = []

    for i in range(len(metric)):
        c1.append(metric[i][0:3])
        c2.append(metric[i][3:6])
        c3.append(metric[i][6:12])

    for i in range(len(metric1)):
        c4.append(metric1[i][0:3])
        c5.append(metric1[i][3:6])
        c6.append(metric1[i][6:12])


    # print("c4:", len(c4))
    # print("c5:", len(c5))
    # print("c6:", len(c6))
    # exit()


    x_train.append(np.asarray(c1))
    x_train.append(np.asarray(c2))
    x_train.append(np.asarray(c3))

    x_test.append(np.asarray(c4))
    x_test.append(np.asarray(c5))
    x_test.append(np.asarray(c6))


    # print("x_train:")
    # print(len(x_train))
    # for i in range(len(x_train)):
    #     print(x_train[i].shape)
    # 
    # print("x_test:")
    # print(len(x_test))

    # for i in range(len(x_test)):
    #     print(x_test[i].shape)

    x_train_t = []
    x_train_t.append(x_train[0])
    x_train_t.append(x_train[1])
    x_train_t.append(x_train[3])

    print("x_train_t:")
    print(len(x_train_t))
    for i in range(len(x_train_t)):
        print(x_train_t[i].shape)

    x_test_t = []
    x_test_t.append(x_test[0])
    x_test_t.append(x_test[1])
    x_test_t.append(x_test[3])

    print("x_test_t:")
    print(len(x_test_t))
    for i in range(len(x_test_t)):
        print(x_test_t[i].shape)

    print('Fine tune model.')

    # 微调
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    # 训练集度量值
    hist = model.fit(x_train_t, y_train, nb_epoch=epochs, batch_size=5, verbose=2)  # 1为输出进度条记录，2为每个epoch输出一行记录
    score = model.evaluate(x_train_t, y_train, verbose=0)

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
    score1 = model.evaluate(x_test_t, y_test, verbose=0)
    # print model.metrics_names
    print('Test score:', score1[0])
    print('Test accuracy:', score1[1])
    # input()
    predict_posibility = model.predict(x_test_t)

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
    # print(predict_label)
    eval(predict_label, y_test)
    # precision = metrics.precision_score(y_test, predict_label)
    # recall = metrics.recall_score(y_test, predict_label)
    f1_score = metrics.f1_score(y_test, predict_label)

    # print('test precision:', precision)
    # print('test recall', recall)
    # print('test f1 score:', f1_score)
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