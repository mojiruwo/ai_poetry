class Tokenizer:
    """
    分词器
    """

    def __init__(self, token_dict):
        self.token_dict = token_dict
        self.token_dict_rev = {value: key for key, value in self.token_dict.items()}
        self.vocab_size = len(self.token_dict)

    def token_to_id(self, token):
        return self.token_dict.get(token, self.token_dict['UNK'])

    def encode(self, tokens):
        print(tokens)
        exit()
        token_ids = [self.token_to_id['[CLS]']]

        for token in tokens:
            token_ids.append(self.token_to_id(token))

        token_ids.append(self.token_to_id('[SEP]'))

        return token_ids

