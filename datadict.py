# -*- coding: utf-8 -*-
import settings
from collections import Counter
import numpy as np
from tokenizer import Tokenizer
from pathlib import Path
import os
import sys

class creatPreData:
    def __init__(self, animal_factory=None):
        """pet_factory is our abstract factory.  We can set it at will."""

        self.pet_factory = animal_factory

    def create_data(self):
        """Creates and shows a pet using the abstract factory"""

        pet = self.pet_factory()
        return pet.handle_data()

class poetryData:
    def handle_data(self):
        data_path = settings.DATASET_PATH
        with open(Path(path1) / data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.replace('：', ':') for line in lines]
        poetry = []
        for line in lines:
            if line.count(':') != 1:
                continue
            ignore_flag = False
            _, last_part = line.split(':')
            for dis_word in disallowed_words:
                if dis_word in last_part:
                    ignore_flag = True
                    break
            if ignore_flag:
                continue
            if len(last_part) > max_len - 2:
                continue
            poetry.append(last_part.replace('\n', ''))

        return poetry


class songData:
    def handle_data(self):
        data_path = settings.DATASET_SONG_PATH
        with open(Path(path1) / data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        poetry = []
        for line in lines:
            if line.count(':') == 1:
                continue
            ignore_flag = False
            for dis_word in disallowed_words:
                if dis_word in line:
                    ignore_flag = True
                    break
            if ignore_flag:
                continue
            if len(line) > max_len - 2:
                continue
            poetry.append(line.replace('\n', ''))

        return poetry

# 禁用词
disallowed_words = settings.DISALLOWED_WORDS
# 句子最大长度
max_len = settings.MAX_LEN
# 最小词频
min_word_frequency = settings.MIN_WORD_FREQUENCY
# mini batch 大小
batch_size = settings.BATCH_SIZE
path1 = os.path.abspath(os.path.dirname(__file__))
train_type = sys.argv
if train_type[1]:
    factory = creatPreData(songData)
else:
    factory = creatPreData(poetryData)
poetry = factory.create_data()
counter = Counter()
for line in poetry:
    counter.update(line)

_tokens = sorted(counter.items(), key=lambda x: -x[1])
_tokens = [token for token, count in _tokens]

_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + _tokens

token_id_dict = dict(zip(_tokens, range(len(_tokens))))
tokenizer = Tokenizer(token_id_dict)
np.random.shuffle(poetry)

