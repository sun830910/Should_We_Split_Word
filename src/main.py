# -*- coding: utf-8 -*-

"""
Created on 2020-08-18 17:10
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

from whole_word_utils import PreProcesser
from one_hot_utils import one_hot_loader
from LSTM import LSTMModel

if __name__ == '__main__':
    # 按照分词后去训练
    whole_word = PreProcesser()
    train_x, train_y = whole_word.get_trainingSet()
    test_x, test_y = whole_word.get_validSet()

    lstm_clf1 = LSTMModel(len(whole_word.token_dict), 100)
    lstm_clf1.train(train_x, train_y, 1)
    lstm_clf1.test(test_x, test_y)

    #  按照one-hot去训练
    one_hot = one_hot_loader()
    train_x, train_y = one_hot.get_trainingSet()
    test_x, test_y = one_hot.get_validSet()

    lstm_clf2 = LSTMModel(len(one_hot.token_dict), 200)
    lstm_clf2.train(train_x, train_y, 1)
    lstm_clf2.test(test_x, test_y)


