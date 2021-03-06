import numpy as np
import pandas as pd
from collections import Counter 


class DataLoader():
    def __init__(self, datapath, x_col, y_col):
        self.df = pd.read_csv(datapath)
        self.X = self.df[x_col]
        self.Y = self.df[y_col]
        self.MAX_INPUT_SEQ_LENGTH = 500
        self.MAX_TARGET_SEQ_LENGTH = 50
        self.MAX_INPUT_VOCAB_SIZE = 5000
        self.MAX_TARGET_VOCAB_SIZE = 2000

    def get_X_and_Y(self):
        return self.X, self.Y

    def fit_text(self):
        input_seq_max_length = self.MAX_INPUT_SEQ_LENGTH
        target_seq_max_length = self.MAX_TARGET_SEQ_LENGTH
        
        input_counter = Counter()
        target_counter = Counter()
        max_input_seq_length = 0
        max_target_seq_length = 0

        for line in self.X:
            text = [word.lower() for word in line.split(' ')]
            seq_length = len(text)
            if seq_length > input_seq_max_length:
                text = text[0:input_seq_max_length]
                seq_length = len(text)
            for word in text:
                input_counter[word] += 1
            max_input_seq_length = max(max_input_seq_length, seq_length)

        for line in self.Y:
            line2 = 'START ' + line.lower() + ' END'
            text = [word for word in line2.split(' ')]
            seq_length = len(text)
            if seq_length > target_seq_max_length:
                text = text[0:target_seq_max_length]
                seq_length = len(text)
            for word in text:
                target_counter[word] += 1
                max_target_seq_length = max(max_target_seq_length, seq_length)
            
        input_word2idx = {}
        for idx, word in enumerate(input_counter.most_common(self.MAX_INPUT_VOCAB_SIZE)):
            input_word2idx[word[0]] = idx + 2
        input_word2idx['PAD'] = 0
        input_word2idx['UNK'] = 1
        input_idx2word = dict([(idx, word) for word, idx in input_word2idx.items()])

        target_word2idx = {}
        for idx, word in enumerate(target_counter.most_common(self.MAX_TARGET_VOCAB_SIZE)):
            target_word2idx[word[0]] = idx + 1
        target_word2idx['UNK'] = 0

        target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

        num_input_tokens = len(input_word2idx)
        num_target_tokens = len(target_word2idx)

        config = dict()
        config['input_word2idx'] = input_word2idx
        config['input_idx2word'] = input_idx2word
        config['target_word2idx'] = target_word2idx
        config['target_idx2word'] = target_idx2word
        config['num_input_tokens'] = num_input_tokens
        config['num_target_tokens'] = num_target_tokens 
        config['max_input_seq_length'] = max_input_seq_length
        config['max_target_seq_length'] = max_target_seq_length

        return config 



