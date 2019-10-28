import os
import pickle

import tensorflow_datasets as tfds


# This module is prepared for preprocessing strings of data and screen position tuples for array of data,
# which can be actually used by TensorFlow

# Preparing encoder
def prepare_encoder(vocabulary_set):
    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
    encoder.save_to_file('encoder')
    return encoder


# Preprocessing data
def prepare_data(data):
    # acquring tokenizer, and tokenizing strings related to action made
    # TODO this probably can be reworked to dialog choice using TKinter, but I didn't find such need in my case
    if os.path.isfile('tokenizer.pickle'):
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    # if tokenizer isn't found, new token is created
    else:
        tokenizer = tfds.features.text.Tokenizer()
        # saving tokenizer for scoring
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    vocabulary_set = set()
    # creating additional copy of data to resolve problems with IndexErrors exceptions
    data_copy = []
    # tokenizing actions from data sets and creating an vocabulary set
    for i in data:
        some_tokens = tokenizer.tokenize(i[1][1])
        vocabulary_set.update(some_tokens)
    encoder = prepare_encoder(vocabulary_set)
    # saving
    for i in data:
        # i is in format [image array,[(x_postion,y_position), action]] before this processing
        # Preparing new set of processed data. Getting previous screen array, encoding string and
        # getting screen position data from (x,y) tuple to [x,y] array
        screen = i[0]
        # token is a 1 element list with int as tokenized string value.
        token = int(encoder.encode(i[1][1])[0])
        screen_position_tuple = i[1][0]
        x_value = screen_position_tuple[0]
        y_value = screen_position_tuple[1]
        # creating a row of preprocessed data and appending new array
        new_data = (screen, [x_value, y_value, token])
        data_copy.append(new_data)
    return data_copy
