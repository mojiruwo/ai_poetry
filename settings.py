# 禁用词，包含如下字符的唐诗将被忽略
DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']
# 句子最大长度
MAX_LEN = 64
# 最小词频
MIN_WORD_FREQUENCY = 1
# 训练的batch size
BATCH_SIZE = 1
# 数据集路径
DATASET_PATH = './poetry_test.txt'
# 每个epoch训练完成后，随机生成SHOW_NUM首古诗作为展示
SHOW_NUM = 5
# 共训练多少个epoch
TRAIN_EPOCHS = 20
# 最佳权重保存路径
BEST_MODEL_PATH = './best_model.h5'