# -*- coding: utf-8 -*-

"""
Created on 2020-08-18 17:18
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import sequence


class one_hot_loader(object):
    def __init__(self):
        self.max_length = 200

        self.df = self.read_corpus()
        self.token_dict = self.get_one_hot(self.df)
        self.build_tokens()

    def read_corpus(self):
        pos = pd.read_excel('../data/pos.xls', header=None, index=None)
        pos['labels'] = 1
        neg = pd.read_excel('../data/neg.xls', header=None, index=None)
        neg['labels'] = 0
        return pd.concat([pos,neg], ignore_index=True)

    def get_one_hot(self, df):
        contents = ''.join(df[0])
        word_dict = dict()
        for word in contents:
            if word not in word_dict:
                word_dict.setdefault(word, 1)
            else:
                word_dict[word] += 1

        word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)

        min_cnt = 20
        more = []
        for word in word_dict:
            if word[1] > min_cnt:
                more.append(word[0])

        result = dict()
        for more_idx in range(len(more)):
            result.setdefault(more[more_idx], more_idx)
        return result

    def sentence2token(self, sentence):
        result = []
        for word in sentence:
            if word in self.token_dict:
                result.append(self.token_dict.get(word))
        return result

    def build_tokens(self):
        self.df['tokens'] = self.df.apply(lambda row: self.sentence2token(row[0]), axis=1)
        self.df['tokens'] = list(sequence.pad_sequences(self.df['tokens'], maxlen=self.max_length))

    def get_trainingSet(self):
        train_x = np.array(list(self.df['tokens']))[::2]
        train_y = np.array(list(self.df['labels']))[::2]
        return train_x, train_y

    def get_validSet(self):
        valid_x = np.array(list(self.df['tokens']))[1::2]
        valid_y = np.array(list(self.df['labels']))[1::2]
        return valid_x, valid_y

    def get_allSet(self):
        x = np.array(list(self.df['tokens']))
        y = np.array(list(self.df['labels']))
        return x, y



if __name__ == '__main__':
    test = one_hot_loader()
