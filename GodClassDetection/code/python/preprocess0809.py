import collections
import os
# import interaction

import tensorflow as tf
import keras.preprocessing.sequence as s
import numpy as np
from gensim import models
from keras.preprocessing.text import Tokenizer


# import gensim


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def get_data(path, maxlen, tokenizer, embedding_matrix):
    mn_datas = []
    train_projects = []
    for project in sorted(os.listdir(path)):
        train_projects.append(project)
        mn_path = path + "/" + project + "/mn_train.txt"  # D:\卜依凡\学习\论文-检测God Class\数据集\trainset_0707\manual-freeplane
        # print("mn_path:", mn_path)
        with open(mn_path) as f:
            for line in f:
                data_mn_list = line.strip('\n').strip('\r').strip().split('.')
                identifiers = []
                tp_sequs = tokenizer.texts_to_sequences(data_mn_list)
                for tp_sequ in tp_sequs:
                    if len(tp_sequ):
                        embeddings = []
                        for tp in tp_sequ:
                            embedding = embedding_matrix[tp]
                            embeddings.append(embedding)
                        identifier_embedding = sum(embeddings) / len(embeddings)
                        identifiers.append(identifier_embedding)
                mn_datas.append(identifiers)
    mn_datas = s.pad_sequences(mn_datas, maxlen=maxlen, dtype='float32')
    print('training ', train_projects)
    return mn_datas


def get_labels(path):  # 01标签
    labels = []
    for project in sorted(os.listdir(path)):
        lb_path = path + "/" + project + "/lb_train.txt"  # D:\卜依凡\学习\论文-检测God Class\数据集\trainset_0707\manual-freeplane
        # print("lb_path:", lb_path)
        with open(lb_path) as f:
            for line in f:
                labels.append(int(line))

    labels = np.asarray(labels)

    nb_labels_ONE = 0
    nb_labels_ZERO = 0
    for i in labels:
        if (i == 0):
            nb_labels_ZERO = nb_labels_ZERO + 1
        if (i == 1):
            nb_labels_ONE = nb_labels_ONE + 1
    print('nb_labels_ONE: ', nb_labels_ONE)
    print('nb_labels_ZERO: ', nb_labels_ZERO)
    return labels


def get_metrics(path):  # 12个度量
    metrics_datas = []
    for project in sorted(os.listdir(path)):
        mt_path = path + "/" + project + "/mt_train.txt"  # D:\卜依凡\学习\论文-检测God Class\数据集\trainset_0707\manual-freeplane
        # print("mt_path:", mt_path)
        with open(mt_path) as f:
            for line in f:
                metrics = line.split()
                metrics_data = []
                for metric in metrics:
                    metrics_data.append(float(metric))
                metrics_datas.append(metrics_data)
    # print(np.asarray(metrics_datas))
    # print(np.asarray(metrics_datas).ndim)
    return np.asarray(metrics_datas)


# # 度量值增加维度
# def get_metrics_up(path):
#     metrics = get_metrics(path)
#     metrics = metrics.reshape(metrics.shape[0], metrics.shape[1], 1)
#     metrics1 = metrics
#     for i in range(199):
#         metrics1 = np.concatenate((metrics1, metrics), axis=-1)
#
#     metrics1 = metrics1.astype(np.float32)  # 交互那一块运算时候需要数据类型相同，mn_datas经过处理之后的数据类型是float32，metrics之前是float64
#
#     return metrics1

# 度量值增加维度
def get_metrics_up():
    data_path = 'E:/GodClassDetection/trainset/data'
    # data_path = '/Users/knight/Desktop/GodClassDetection/trainset/train'
    # data_path = '/Users/knight/Desktop/GodClassDetection_0704/trainset/finetune'
    metrics = get_metrics(data_path)
    metrics = metrics.reshape(metrics.shape[0], metrics.shape[1], 1)
    metrics1 = metrics
    for i in range(199):
        metrics1 = np.concatenate((metrics1, metrics), axis=-1)

    metrics1 = metrics1.astype(np.float32)  # 交互那一块运算时候需要数据类型相同，mn_datas经过处理之后的数据类型是float32，metrics之前是float64
    # print(metrics1.type)

    return metrics1


def get_interaction(mn_datas):
    metrics = get_metrics_up()
    # embed_map = tf.concat([metrics, mn_datas], axis=1)
    print(metrics.shape)
    print(mn_datas.shape)
    embed_map = np.concatenate((metrics, mn_datas), axis=1)

    # print(embed_map.type)
    # print(embed_map)

    return embed_map


# 无交互的输入
def get_xy_train1(path, mn_maxlen, tokenizer, embedding_matrix):
    mn_datas = get_data(path, mn_maxlen, tokenizer, embedding_matrix)
    metrics_datas = get_metrics(path)
    labels = get_labels(path)

    print('Shape of name tensor:', mn_datas.shape)
    print('Shape of metrics tensor:', metrics_datas.shape)
    print('Shape of label tensor:', labels.shape)

    np.random.seed(0)
    indices = np.arange(mn_datas.shape[0])
    np.random.shuffle(indices)

    mn_datas = np.asarray(mn_datas)[indices]
    metrics_datas = np.asarray(metrics_datas)[indices]
    labels = np.asarray(labels)[indices]
    # print("mn_datas:", mn_datas[0])
    # print(mn_datas.shape)
    # print("metrics_datas:", metrics_datas)
    # print(metrics_datas.shape)

    x_train = []
    x_train.append(mn_datas)
    x_train.append(metrics_datas)
    y_train = labels
    return x_train, y_train


# 有交互的输入
def get_xy_train(path, mn_maxlen, tokenizer, embedding_matrix):
    mn_datas = get_data(path, mn_maxlen, tokenizer, embedding_matrix)
    metrics_datas = get_metrics(path)
    interaction_datas = get_interaction(mn_datas)
    labels = get_labels(path)

    print('Shape of name tensor:', mn_datas.shape)
    print('Shape of metrics tensor:', metrics_datas.shape)
    print('Shape of interaction_datas tensor:', interaction_datas.shape)
    print('Shape of label tensor:', labels.shape)

    np.random.seed(0)
    indices = np.arange(mn_datas.shape[0])
    np.random.shuffle(indices)

    mn_datas = np.asarray(mn_datas)[indices]
    metrics_datas = np.asarray(metrics_datas)[indices]
    interaction_datas = np.asarray(interaction_datas)[indices]
    labels = np.asarray(labels)[indices]
    # print("mn_datas:", mn_datas[0])
    # print(mn_datas.shape)
    # print("metrics_datas:", metrics_datas)
    # print(metrics_datas.shape)

    x_train = []
    x_train.append(mn_datas)
    x_train.append(metrics_datas)
    x_train.append(interaction_datas)
    x_train.append(mn_datas)
    # x_train.append(metrics_datas)
    y_train = labels
    return x_train, y_train


def get_xy_test(path, test_project, maxlen, tokenizer, embedding_matrix):
    labels = []
    mn_datas = []
    metrics_datas = []
    lb_path = path + "/" + test_project + "/lb_train.txt"
    with open(lb_path) as f:
        for line in f:
            labels.append(int(line))

    mn_path = path + "/" + test_project + "/mn_train.txt"
    with open(mn_path) as f:
        for line in f:
            data_mn_list = line.strip('\n').strip('\r').strip().split('.')
            identifiers = []
            tp_sequs = tokenizer.texts_to_sequences(data_mn_list)
            for tp_sequ in tp_sequs:
                if len(tp_sequ):
                    embeddings = []
                    for tp in tp_sequ:
                        embedding = embedding_matrix[tp]
                        embeddings.append(embedding)
                    identifier_embedding = sum(embeddings) / len(embeddings)
                    identifiers.append(identifier_embedding)
            mn_datas.append(identifiers)

    mt_path = path + "/" + test_project + "/mt_train.txt"
    with open(mt_path) as f:
        for line in f:
            metrics = line.split()
            metrics_data = []
            for metric in metrics:
                metrics_data.append(float(metric))
            metrics_datas.append(metrics_data)

    labels = np.asarray(labels)

    nb_labels_ONE = 0
    nb_labels_ZERO = 0
    for i in labels:
        if (i == 0):
            nb_labels_ZERO = nb_labels_ZERO + 1
        if (i == 1):
            nb_labels_ONE = nb_labels_ONE + 1
    print('nb_labels_ONE: ', nb_labels_ONE)
    print('nb_labels_ZERO: ', nb_labels_ZERO)

    mn_datas = s.pad_sequences(mn_datas, maxlen=maxlen, dtype='float32')
    metrics_datas = np.asarray(metrics_datas)

    x_val = []
    x_val.append(mn_datas)
    x_val.append(metrics_datas)
    y_val = labels
    return x_val, y_val


def get_test_data(path, maxlen, tokenizer):
    texts_first = []
    texts_second = []
    for test_index in sorted(os.listdir(path)):
        test_class_path = path + test_index + '/'
        mn_path = test_class_path + 'mn_train.txt'
        # print('in ' + mn_path)
        with open(mn_path) as f:
            for line in f:
                identifiers = line.split('.')
                identifier0 = identifiers[0]
                identifier1 = identifiers[1]
                words0 = identifier0.split()
                words1 = identifier1.strip('\b').split()
                words0 = " ".join(words0)
                words1 = " ".join(words1)
                texts_first.append(words0)
                texts_second.append(words1)
    sequences_first = tokenizer.texts_to_sequences(texts_first)
    sequences_second = tokenizer.texts_to_sequences(texts_second)
    data1 = s.pad_sequences(sequences_first, maxlen=maxlen)
    data2 = s.pad_sequences(sequences_second, maxlen=maxlen)
    return data1, data2


def get_x_pre(mn_path, jc_path, mn_maxlen, jc_maxlen, tokenizer):
    data1, data2 = get_data(mn_path, mn_maxlen, tokenizer)
    jaccard1, jaccard2 = get_jaccard(jc_path, jc_maxlen)
    x_pre = []
    x_pre.append(data1)
    x_pre.append(data2)
    x_pre.append(jaccard1)
    x_pre.append(jaccard2)
    return x_pre


def get_embedding_matrix(all_word_index, model_path, dim):
    # print('Preparing embedding matrix.')
    # print("Model_Path==" + model_path)  # Model_Path==D:/项目/codesmell/GodClassDetection-master/embedding_model/new_model20180517/new_model20180517.bin
    embedding_matrix = np.zeros((len(all_word_index) + 1, dim))
    # print("Matrix:")
    # print(embedding_matrix.shape)

    w2v_model = models.word2vec.Word2Vec.load(model_path)
    # print(w2v_model)

    for word, i in all_word_index.items():
        try:
            embedding_vector = w2v_model[word]
        except KeyError:
            continue
        embedding_matrix[i] = embedding_vector  # word_index to word_embedding_vector ,<20000(nb_words)
    # print(embedding_matrix)

    return embedding_matrix


def get_tokenizer(path):  # 标识符
    texts = []
    for sett in sorted(os.listdir(path)):
        # print("Sett=:"+sett)
        if sett == 'temp':
            continue
        for project in sorted(os.listdir(path + '/' + sett)):
            full_path = path + '/' + sett + "/" + project + '/full_mn/mn_full.txt'
            print(full_path)
            # print("Path="+path+",Sett="+sett+",project="+project+",Get Full Path:"+full_path)  #../../trainset/train/AoI30/full_mn/full_mn/mn_full.txt,可以输出路径，说明first_train关于路径部分没有问题
            f = open(full_path)
            for line in f:
                texts.append(line)
            f.close()

    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(texts)
    return tokenizer
