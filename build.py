# -*- coding: utf-8 -*-
import utils
import datadict
import tensorflow as tf
import settings
from pathlib import Path
import os

path1 = os.path.abspath(os.path.dirname(__file__))
model = tf.keras.models.load_model(Path(path1) / settings.BEST_MODEL_PATH)
polist = utils.generate_random_poetry(datadict.tokenizer,model)
print(polist)