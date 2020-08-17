import settings
import numpy as np
def generate_random_poetry(tokenizer, model, s=''):
    token_ids = tokenizer.encode(s)
    token_ids = token_ids[:-1]
    while len(token_ids) < settings.MAX_LEN:
        output = model(np.array([token_ids, ],dtype=np.int32))
        _probas = output.numpy()[0, -1, 3:]
        del output
        p_args = _probas.argsort()[::-1][:100]

        p = _probas[p_args]

        p = p / sum(p)

        target_index = np.random.choice(len(p), p = p)
        target = p_args[target_index] + 3
        token_ids.append(target)
        if target == 3:
            break
    return tokenizer.decode(token_ids)

