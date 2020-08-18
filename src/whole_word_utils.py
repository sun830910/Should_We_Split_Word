# -*- coding: utf-8 -*-

"""
Created on 2020-08-17 20:47
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""


import pandas as pd
import numpy as np
import jieba

from tensorflow.keras.preprocessing import sequence


class PreProcesser(object):
    def __init__(self):
        self.token_dict = None
        self.corpus_df = None
        self.max_length = 100

        self.get_all_words_dict()
        self.padding_tokens()  # 将输入转换成tokens形式

    def read_corpus(self, split=False):
        """
        读训练集数据
        :return:
        """
        neg_path = '../data/neg.xls'
        pos_path = '../data/pos.xls'

        neg_df = pd.read_excel(neg_path, header=None, index=None)
        pos_df = pd.read_excel(pos_path, header=None, index=None)
        neg_df['label'] = 0
        pos_df['label'] = 1
        print('负面评论数量: {} 条'.format(str(len(neg_df))))
        print('正面评论数量: {} 条'.format(str(len(pos_df))))

        total_df = pd.concat([neg_df, pos_df], ignore_index=True)
        if split == True:
            total_df['words'] = total_df.apply(lambda total_df: self.split_sentence(total_df[0]), axis=1)  # 分词
        else:
            total_df['words'] = total_df[0]

        self.corpus_df = total_df
        return total_df

    def split_sentence(self, sentence):
        return list(jieba.cut(str(sentence)))

    def read_sum(self, split=False):
        sum_path = '../data/sum.xls'
        comments = pd.read_excel(sum_path)
        comments = comments[comments['rateContent'].notnull()]
        if split == True:
            comments['words'] = comments.apply(lambda row: self.split_sentence(row['rateContent']), axis=1)  # 分词
        else:
            comments['words'] = comments['rateContent']
        return comments


    def get_all_words_dict(self):
        """
        构建单词token字典
        :return:
        """
        corpus_data = self.read_corpus(split=True)
        sum_data = self.read_sum(split=True)
        total_words = pd.concat([corpus_data['words'], sum_data['words']], ignore_index=True)
        word_cnt_dict = dict()
        for words in total_words:
            for word in words:
                if word not in word_cnt_dict:
                    word_cnt_dict.setdefault(word, 1)
                else:
                    word_cnt_dict[word] += 1
        sorted_dict = sorted(word_cnt_dict.items(), key=lambda x: x[1], reverse=True)  # 将字典按词出现次数从大到小排序

        word2idx_dict = dict()
        for idx in range(len(sorted_dict)):
            word2idx_dict.setdefault(sorted_dict[idx][0], idx)
        self.token_dict = word2idx_dict

    def sentence2token(self, splited):
        """
        将句子转换成tokens表示
        :param splited:
        :return:
        """
        result = []
        for word in splited:
            if word in self.token_dict:
                result.append(self.token_dict.get(word))  # 这边对未出现在字典中的词选择放弃
        return result

    def padding_tokens(self):
        """
        用于深度学习模型，将DataFrame中的words转换为tokens形式
        :return:
        """
        self.corpus_df['words'] = self.corpus_df.apply(lambda row: self.sentence2token(row['words']), axis=1)
        self.corpus_df['words'] = list(sequence.pad_sequences(self.corpus_df['words'], maxlen=self.max_length))

    def get_trainingSet(self):
        train_x = np.array(list(self.corpus_df['words']))[::2]
        train_y = np.array(list(self.corpus_df['label']))[::2]
        return train_x, train_y

    def get_validSet(self):
        valid_x = np.array(list(self.corpus_df['words']))[1::2]
        valid_y = np.array(list(self.corpus_df['label']))[1::2]
        return valid_x, valid_y

    def get_allSet(self):
        x = np.array(list(self.corpus_df['words']))
        y = np.array(list(self.corpus_df['label']))
        return x, y

if __name__ == '__main__':
    test = PreProcesser()
    train, label = test.get_trainingSet()
