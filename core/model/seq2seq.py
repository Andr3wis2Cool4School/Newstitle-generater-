import numpy as np 
import pandas as pd
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Input
from tensorflow.keras.layers.recurrent import LSTM
from keras.utils import plot_model

HIDDEN_UNITS = 100
DEFAULT_BATCH_SIZE = 64
VERBOSE = 1
DEFAULT_EPOCHS = 10

class Seq2SeqSummarizer():

    model_name = 'seq2seq'

    def __init__(self, config):
        self.num_input_tokens = config['num_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.config = config

        self.version = 0
        if 'version' in config:
            self.version = config['version']

        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encode_embedding = Embedding(input_dim=self.num_input_tokens, output_dim=HIDDEN_UNITS,
                                     input_length=self.max_input_seq_length, name='encoder_embedding')

        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.num_target_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, 
                                                                         initial_state=encoder_states)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
                                                                        




