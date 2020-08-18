# -*- coding: utf-8 -*-

"""
Created on 2020-08-18 14:57
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""


import os
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.models import load_model

from utils import PreProcesser


class LSTMModel(object):
    def __init__(self, word_dict_length, seq_length):
        self.word_dict_length = word_dict_length
        self.seq_length = seq_length

        self.model = None

    def build_model(self):
        model_input = keras.layers.Input((self.seq_length))
        embedding_layer = Embedding(self.word_dict_length+1, 256)(model_input)
        lstm_layer1 = LSTM(128)(embedding_layer)
        dropout_layer = Dropout(0.5)(lstm_layer1)
        dense_layer = Dense(1)(dropout_layer)
        model_output = Activation('sigmoid')(dense_layer)

        model = keras.models.Model(inputs=model_input, outputs=model_output)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model

    def train(self, train_x, train_y, epochs):
        if self.model is None:
            self.model = self.build_model()

        for _ in range(epochs):
            self.model.fit(train_x, train_y, batch_size=16)
        self.model.save('../result/lstm/lstm_model.h5')

    def test(self, test_x, test_y):
        model_path = '../result/lstm/lstm_model.h5'
        if self.model is None and os.path.exists(model_path):
            self.model = load_model(model_path)
        pred_y = self.model.predict(test_x)
        pred_label = [1 if y >= 0.5 else 0 for y in pred_y]

        assert len(pred_label) == len(test_y)
        cnt = 0
        for y_idx in range(len(pred_label)):
            if pred_label[y_idx] == test_y[y_idx]:
               cnt += 1

        print('当前测试集准确率为: {} '.format(str(cnt/len(test_y))))


if __name__ == '__main__':
    test = PreProcesser()
    test.padding_tokens()  # 将输入转换成tokens形式
    train_x, train_y = test.get_trainingSet()
    test_x, test_y = test.get_validSet()

    lstm_clf = LSTMModel(len(test.token_dict), 50)
    # lstm_clf.train(train_x, train_y, 1)
    lstm_clf.test(test_x, test_y)
