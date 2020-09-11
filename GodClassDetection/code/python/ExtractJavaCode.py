# -*- encoding: utf-8 -*-
"""
@File    : ExtractJavaCode.py
@Time    : 2019/9/11 20:31
@Author  : knight
"""

import os

project_path = "C:/Users/AAAA/Desktop/large class/Database/word2vec语料库/mn"
project_path1 = "C:/Users/AAAA/Desktop/large class/Database/word2vec语料库/mn1"
project_path2= "C:/Users/AAAA/Desktop/large class/Database/word2vec语料库/mn2"
project_path3 = "C:/Users/AAAA/Desktop/large class/Database/word2vec语料库/mn3"
output_path = "C:/Users/AAAA/Desktop/large class/Database/word2vec语料库/allcode.txt"
output_path1 = "C:/Users/AAAA/Desktop/large class/Database/word2vec语料库/allcode1.txt"
output_path2 = "C:/Users/AAAA/Desktop/large class/Database/word2vec语料库/allcode2.txt"
output_path3 = "C:/Users/AAAA/Desktop/large class/Database/word2vec语料库/allcode3.txt"

def findAll(dirs):
    for a_dir in os.listdir(dirs):
        print(a_dir)
        new_dir = dirs+'/'+a_dir
        if a_dir.endswith('.txt'):
            read_file = open(new_dir,'rb')
            write_file = open(output_path3,'ab+')
            data = read_file.read()
            # print(data)
            write_file.write(data)
            read_file.close()
            write_file.close()

        if os.path.isdir(new_dir):
            findAll(new_dir)
            os.chdir(new_dir)


if __name__ == "__main__":
    findAll(project_path3)

