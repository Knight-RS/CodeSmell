# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:01:30 2017

@author: xzf0724
"""


import nltk
from gensim.models import word2vec

vec_dim = 200




def moreTrain():
    # file = open("/Users/knight/Desktop/GodClassDetection-master-mao-new/word2vec训练结果/all_mn.txt", 'r', encoding='UTF-8-sig')
    # sentences = []
    # for line in file:
    #     temp = line.replace('\n', '').split('\t')
    #     item = ''.join(temp)  # 转成str类型
    #     item = nltk.word_tokenize(item)  # nltk.word_tokenize()操作对象要是str
    #     sentences.append(item)
    # #     print(item)
    # file.close()


    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # sentences = word2vec.Text8Corpus(u"allcode1.txt")
    sentences = word2vec.Text8Corpus(u"/Users/knight/Desktop/GodClassDetection-master-mao-new/word2vec训练结果/all_mn.txt")
    model = word2vec.Word2Vec(sentences, min_count=0, window=10, size=vec_dim, workers=3, iter=10)
    # 模型保存与载入
    model.save('/Users/knight/Desktop/GodClassDetection_0704/embedding_model/new_model6.bin')
    # new_model.save('new_model')

if __name__ == "__main__":
    moreTrain()