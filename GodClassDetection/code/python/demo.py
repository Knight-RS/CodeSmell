# # import tensorflow as tf
# # import keras
# # import preprocess
# #
# # # print(tf.version)
# # # print(tf.__version__)
# # print(keras.__version__)
# # print(tf.keras.__version__)
# #
# # data_path = '/Users/knight/Desktop/GodClassDetection_0704/trainset/train'
# # metrics = preprocess.get_metrics_up(data_path)
# # # print(metrics)
# # # print(metrics.shape)
# #
# #
# # # W_Q = []
# # # W_K = []
# # # W_V = []
# # # W_K.append(tf.Variable(tf.random.truncated_normal(shape=(8, 6)), name="key_"))  # k, d
# # # W_Q.add_weight(name='WQ', shape=(8, 6), initializer='glorot_uniform', trainable=True)
# # #
# # # print(W_K)
# # # print(W_Q)
#
# # x = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
# # print(len(x))
# # kf = KFold(n_splits=12)
# #
# # # for train_index, test_index in kf:
# # #     print(train_index, test_index)
# # x_train = []
# # y_train = []
# # tmp = []
# # tmp1 = []
# # for train, test in kf.split(x):
# #     print(train, test)
# #     # y_train.append(x[test[0]])
# #     # print(train)
# #     # print(x[train[0]])
# #     for i in range(len(train)):
# #         tmp.append(x[train[i]])
# #     x_train.append(tmp)
# #     tmp = []
# #     for j in range(len(test)):
# #         tmp1.append(x[test[j]])
# #     y_train.append(tmp1)
# #     tmp1 = []
# #
# # print(len(x_train))
# # print(x_train)
# # print(y_train)
#
#
# # kf = KFold(n_splits=12)
#
# # tmp = []
# # for train, test in kf.split(x_k):
# #     # print(train, test)
# #     for i in range(len(train)):
# #         tmp.append(x_k[train[i]])
# #     x_train.append(tmp)
# #     tmp = []
# #     for j in range(len(test)):
# #         tmp.append(x_k[test[j]])
# #     x_test.append(tmp)
# #     tmp = []
# # print(len(x_train))
# # print(len(x_test))
# #
# # for train, test in kf.split(y_k):
# #     # print(train, test)
# #     for i in range(len(train)):
# #         tmp.append(y_k[train[i]])
# #     y_train.append(tmp)
# #     tmp = []
# #     for j in range(len(test)):
# #         tmp.append(y_k[test[j]])
# #     y_test.append(tmp)
# #     tmp = []
# # print(len(y_train))
# # print(len(y_test))
# # print(len(y_test[0]))
# # print(y_test)
# # print(x_train)
# # print(x_test)
#
# x_k = [
#     [12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1],
#     [11, 10, 9,   8,  7,  6,  5,  4,  3,  2,  1,  12],
#     [10, 9,  8,   7,  6,  5,  4,  3,  2,  1,  12, 11],
#     [9,  8,  7,   6,  5,  4,  3,  2,  1,  12, 11, 10],
#     [8,  7,  6,   5,  4,  3,  2,  1,  12, 11, 10, 9],
#     [7,  6,  5,   4,  3,  2,  1,  12, 11, 10, 9,  8],
#     [6,  5,  4,   3,  2,  1,  12, 11, 10, 9,  8,  7],
#     [5,  4,  3,   2,  1,  12, 11, 10, 9,  8,  7,  6],
#     [4,  3,  2,   1,  12, 11, 10, 9,  8,  7,  6,  5],
#     [3,  2,  1,   12, 11, 10, 9,  8,  7,  6,  5,  4],
#     [2,  1,  12,  11, 10, 9,  8,  7,  6,  5,  4,  3],
#     [1,  12, 11,  10, 9,  8,  7,  6,  5,  4,  3,  2]
# ]
# y_k = [0,  1,  0,  0, 1, 1, 0, 0, 0, 1, 1, 1]
# x_train = []
# y_train = []
# x_test = []
# y_test = []
# idx = 0
# for i in range(12):
#     for j in range(12):
#         if j == idx:
#             x_test.append(x_k[j])
#             y_test.append(y_k[j])
#         else:
#             x_train.append(x_k[j])
#             y_train.append(y_k[j])
#     idx = idx + 1
#     print("x_train:", len(x_train))
#     print("y_train:", len(y_train))
#     print("x_test", len(x_test))
#     print("y_test:", len(y_test))
#     print("x_train:", x_train)
#     print("y_train:", y_train)
#     print("x_test", x_test)
#     print("y_test:", y_test)
#     print("\n")
#     x_train = []
#     y_train = []
#     x_test = []
#     y_test = []


import csv
import time
localtime = time.asctime(time.localtime(time.time()))
f = open('/Users/knight/Desktop/GodClassDetection/result/' + 'result_' + localtime + '.csv', 'w')
p = 1
writer = csv.writer(f)
writer.writerow(['10001','ZhangSan',p])
# writer.writerow(p))