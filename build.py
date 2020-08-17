import utils
import datadict
import tensorflow as tf
import settings

model = tf.keras.models.load_model(settings.BEST_MODEL_PATH)
polist = utils.generate_random_poetry(datadict.tokenizer,model)
print(polist)