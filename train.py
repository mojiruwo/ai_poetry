# -*- coding: utf-8 -*-
import tensorflow as tf
import datadict

from model import model

import settings
import utils

from poetrydata import PoetryDataGenerator

class Evaluate(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lowest = 1e10

    def on_epoch_end(self,epoch, logs = None):
        if logs['loss'] < self.lowest:
            self.lowest = logs['loss']
            model.save(settings.BEST_MODEL_PATH)

        print()
        for i in range(settings.SHOW_NUM):
            print(utils.generate_random_poetry(datadict.poetry, model))

# 创建数据字典
data_generator = PoetryDataGenerator(datadict.poetry, random=True)
# 处理数据字典
#
model.fit_generator(data_generator.for_fit(), steps_per_epoch=data_generator.step, epochs=settings.TRAIN_EPOCHS, callbacks=[Evaluate()])