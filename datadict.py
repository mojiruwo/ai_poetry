import settings
from collections import Counter
# 禁用词
disallowed_words = settings.DISALLOWED_WORDS
# 句子最大长度
max_len = settings.MAX_LEN
# 最小词频
min_word_frequency = settings.MIN_WORD_FREQUENCY
# mini batch 大小
batch_size = settings.BATCH_SIZE

with open(settings.DATASET_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = [line.replace('：',':') for line in lines]

poetry = []

for line in lines:
    if line.count(':') != 1:
        continue
    _, last_part = line.split(':')
    poetry.append(last_part.replace('\n', ''))

counter = Counter()

for line in poetry:
    counter.update(line)


