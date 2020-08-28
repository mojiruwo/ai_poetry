# -*- coding: utf-8 -*-
import settings
import math
import numpy as np
from datadict import tokenizer
import tensorflow as tf

class PoetryDataGenerator:
    """
    数据处理器
    """

    def __init__(self, data, random = False):
        self.data = data
        self.data_len = len(data)
        self.batch_size = settings.BATCH_SIZE
        self.step = int(math.floor(self.data_len / self.batch_size))
        self.random = random

    def sequence_padding(self,data, length = None, padding = None):
        if length is None:
            length = max(map(len,data))

        if padding is None:
            padding = tokenizer.token_to_id('[PAD]')

        outputs = []
        for line in data:
            padding_length = length - len(line)
            if padding_length > 0:
                outputs.append(np.concatenate([line, [padding] * padding_length]))
            else:
                outputs.append(line[:length])

        return np.array(outputs)

    def __iter__(self):
        total = self.data_len
        if self.random:
            np.random.shuffle(self.data)

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size , total)
            batch_data = []
            for single_data in self.data[start:end]:
                batch_data.append(tokenizer.encode(single_data))
            batch_data = self.sequence_padding(batch_data)
            yield batch_data[:, :-1], tf.one_hot(batch_data[:, 1:], tokenizer.vocab_size)

            del batch_data


    def for_fit(self):
        while True:
            yield from self.__iter__()

