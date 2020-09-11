from __future__ import division

import os
import time

import preprocess4
from keras.models import load_model
from sklearn import metrics
from interaction1 import InteractingLayer
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

localtime = time.asctime(time.localtime(time.time()))
print("begin time", localtime)

date_id = '0528'
k = 0
BASE_DIR = ''
# W2V_MODEL_DIR = '/Users/knight/Desktop/GodClassDetection/embedding_model/new_model6.bin'
W2V_MODEL_DIR = 'E:/GodClassDetection/embedding_model/new_model6_nltk.bin'

TRAIN_SET_DIR = 'E:/GodClassDetection/trainset'  # 直接改成自己的路径

FULL_MN_DIR = TRAIN_SET_DIR

# TRAINED_MODEL_JSON_DIR = 'D:/项目/codesmell/GodClassDetection-master/trained_model/' + 'trained_model_' + '/trained_model_json.json'
# TRAINED_MODEL_WEIGHT_DIR = 'D:/项目/codesmell/GodClassDetection-master/trained_model/' + 'trained_model_' + '/trained_model_weights.h5'

MAX_SEQUENCE_LENGTH = 50
MAX_JACCARD_LENGTH = 30
EMBEDDING_DIM = 200

f1_scores = []
tokenizer = preprocess4.get_tokenizer(FULL_MN_DIR)
all_word_index = tokenizer.word_index
embedding_matrix = preprocess4.get_embedding_matrix(all_word_index, W2V_MODEL_DIR, dim=EMBEDDING_DIM)
# test_path = TRAIN_SET_DIR + '/finetune/'
test_path = TRAIN_SET_DIR + '/test/'
initial_line = {'manual-FreeMind_integration': 374, 'manual-ant-1.8.2': 946, 'manual-derby-10.9.1.0': 1986,
                'manual-hadoop-1.1.2': 1770, 'manual-jhotdraw-7.5.1': 436, 'manual-struts-2.2.1': 926}


# 加载预训练的模型
model_path = 'E:/GodClassDetection/trained_model/pre_training_model_int_10.h5'
interaction = InteractingLayer()
model = load_model(model_path, custom_objects={'InteractingLayer': interaction})


total_y_pre = []
total_y_test = []

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

    a = tp + fp
    b = tp + fn
    c = tn + fp
    d = tn + fn
    print("MCC : ", (tp * tn - fp * fn) / ((a * b * c * d) ** 0.5))
    print("AUC : ", metrics.roc_auc_score(y_test, y_pre))


    return 2 * P * R / (P + R)


# model = load_model(model_path)
# model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

for project in sorted(os.listdir(test_path)):
    print(project)
    x_val, y_val = preprocess4.get_xy_test(test_path, project, mn_maxlen=MAX_SEQUENCE_LENGTH, tokenizer=tokenizer,
                                              embedding_matrix=embedding_matrix)

    metric = preprocess4.get_metrics_up1(project, test_path)
    c1 = []
    c2 = []
    c3 = []
    for i in range(len(metric)):
        c1.append(metric[i][0:3])
        c2.append(metric[i][3:6])
        c3.append(metric[i][6:12])

    x_val.append(np.asarray(c1))
    x_val.append(np.asarray(c2))
    x_val.append(np.asarray(c3))
    predict_posibility = model.predict(x_val)

    predict_label = []
    godclass_indices = []
    for i, item in enumerate(predict_posibility):
        if item <= 0.5:
            predict_label.append(0)
        else:
            godclass_indices.append(i)
            predict_label.append(1)

    labeld_godclasses = []
    for i, item in enumerate(y_val):
        if item == 1:
            labeld_godclasses.append(i)
    eval(predict_label, y_val)
    # precision = metrics.precision_score(y_val, predict_label)
    # recall = metrics.recall_score(y_val, predict_label)
    # f1_score = metrics.f1_score(y_val, predict_label)

    # print('test precision:', precision)
    # print('test recall', recall)
    # print('test f1 score:', f1_score)
    # f1_scores.append(f1_score)
    print("\n")

# input()
# avg_f1 = sum(f1_scores) / len(f1_scores)
# print("average f1: ", avg_f1)
localtime = time.asctime(time.localtime(time.time()))
print("end time", localtime)
