"""
Provides training code for performing Machine translation of formation description
    to either transcription or grain size classification. Written as part of the 
    FORCE Hackathon 2019.

    The script can be run as a command line tool. Run the following command for
    further information about usage:
        python machine_translation.py --help


Bjørn Harald Fotland

Based on English to French machine translation example in NLP-with-python repository:
https://github.com/susanli2016/NLP-with-Python/blob/master/machine_translation.ipynb
"""

import sys
from importlib import import_module

# Required libraries. Tested with TensorFlow 1.14
libnames = ['pandas', 'xlsxwriter', 'tensorflow', 'xlrd']
for libname in libnames:
    try:
        lib = import_module(libname)
    except:
        print(sys.exc_info())
    else:
        globals()[libname] = lib

import argparse
import collections
import os

import numpy as np
import pandas as pd
import xlsxwriter
from tensorflow.keras.layers import (GRU, Activation, Bidirectional, Dense,
                                     Input, RepeatVector, TimeDistributed)
from tensorflow.keras.layers import Embedding
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.client import device_lib


def filter_extended(input_path, output_path):
    data = pd.read_excel(input_path)
    filtered = data[[
        'Well Name', 'Measured Depth', 'Formation description original',
        'Non sorted Transcription', 'clean lithology', 'color', 'grain size',
        'rounding', 'cement', 'sorting'
    ]]

    # Remove empty cells
    filtered.dropna(subset=['Formation description original'], inplace=True)
    filtered.dropna(subset=['grain size'], inplace=True)
    filtered.dropna(subset=['clean lithology'], inplace=True)
    filtered.dropna(subset=['rounding'], inplace=True)
    filtered.dropna(subset=['cement'], inplace=True)
    filtered.dropna(subset=['sorting'], inplace=True)

    print(len(filtered))

    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    filtered.to_excel(writer, sheet_name='Sheet1')
    writer.save()


def filter(input_path, output_path):
    data = pd.read_excel(input_path)

    # Remarks on the table: CORE number
    filtered = data[[
        'Well Name', 'Measured Depth', 'Formation description original',
        'Remarks on the table', 'Non sorted Transcription'
    ]]

    # Remove missing descriptions
    filtered.dropna(subset=['Formation description original'], inplace=True)

    print(filtered.head())

    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    filtered.to_excel(writer, sheet_name='Sheet1')
    writer.save()


def get_filtered_sentences(filename):
    data = pd.read_excel(filename)

    transcription_sentences = list(data['Non sorted Transcription'])
    formation_description_sentences = list(
        data['Formation description original'])

    print(len(transcription_sentences))
    print(len(formation_description_sentences))

    # Remove "as above" and ensure transcription is string
    transcription_filtered = []
    formation_description_filtered = []
    for i, (e, f) in enumerate(
            zip(formation_description_sentences, transcription_sentences)):
        if type(f) != str or "as above" in f:
            continue
        else:
            formation_description_filtered.append(
                e.replace('.', ' ').replace(',', ' '))
            transcription_filtered.append(f)

    print('Filtered counts {}/{}'.format(len(formation_description_filtered),
                                         len(transcription_filtered)))

    return (formation_description_filtered,
            transcription_filtered), ('Formation description', 'Transcription')


def get_filtered_extended_sentences(filename):
    data = pd.read_excel(filename)

    columns = [
        'Formation description original', 'Non sorted Transcription',
        'clean lithology', 'color', 'grain size', 'rounding', 'cement',
        'sorting'
    ]

    formation_desc = list(data['Formation description original'])
    transcription = list(data['Non sorted Transcription'])
    grain_size = list(data[columns[4]])

    formation_description_sentences = formation_desc
    grain_size_sentences = grain_size

    formation_description_filtered = []
    grain_size_filtered = []
    for i, (e, f, t) in enumerate(
            zip(formation_description_sentences, grain_size_sentences,
                transcription)):
        if type(t) != str or "as above" in t:
            # Skip as above for now ..
            continue
        else:
            # Prepare split by . and ,
            formation_description_filtered.append(
                e.replace('.', ' ').replace(',', ' '))
            grain_size_filtered.append(f)

    print('Filtered counts {}/{}'.format(len(formation_description_filtered),
                                         len(grain_size_filtered)))

    return (formation_description_filtered,
            grain_size_filtered), ('Formation description', 'Grain size')


def print_words_info(sentences, name='Language', split=None):
    words = [word for sentence in sentences for word in sentence.split(split)]
    print('{} words in {}.'.format(len(words), name))
    word_counter = collections.Counter(words)
    print('{} unique {} words.'.format(len(word_counter), name))
    print('10 Most common words in the {} dataset:'.format(name))
    # print('"' + '" "'.join(list(zip(*word_counter.most_common(10)))[0]) + '"')
    print(list(zip(*word_counter.most_common(10))))
    print()


def tokenize(sentences, split=' '):
    """
    Tokenizes sentences
    :param sentences: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized sentences data, tokenizer used to tokenize sentences)
    """
    tokenizer = Tokenizer(char_level=False, filters='', split=split)
    tokenizer.fit_on_texts(sentences)
    return tokenizer.texts_to_sequences(sentences), tokenizer


def pad(sentences, length=None):
    """
    Add padding to sentences.
    :param sentences: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in sentences.
    :return: Padded numpy array of sequences
    """
    if length is None:
        length = max([len(sentence) for sentence in sentences])
    return pad_sequences(sentences, maxlen=length, padding='post')


def preprocess(x, y, split_x, split_y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x, split_x)
    preprocess_y, y_tk = tokenize(y, split_y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = ''

    return ' '.join(
        [index_to_words[prediction] for prediction in np.argmax(logits, 1)])


def get_model(input_shape, output_sequence_length, from_language_vocab_size,
              to_language_vocab_size):
    """ Set up the ML model, loss function and optimizer """

    model = Sequential()
    model.add(
        Embedding(input_dim=from_language_vocab_size + 1,
                  output_dim=128,
                  input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256, return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(
        TimeDistributed(Dense(to_language_vocab_size + 1,
                              activation='softmax')))
    learning_rate = 0.005

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])

    return model


def main(dataset_path, target, samples, epochs):
    if target == 'transcription':
        output_path = 'filtered.xlsx'
        if not os.path.isfile(output_path):
            print('Extracting data needed from {} and storing it in {}'.format(
                dataset_path, output_path))
            filter(dataset_path, output_path)
        (sentences_a,
         sentences_b), (language_a,
                        language_b) = get_filtered_sentences(output_path)
    elif target == 'grain size':
        output_path = 'filtered_extended.xlsx'
        if not os.path.isfile(output_path):
            print('Extracting data needed from {} and storing it in {}'.format(
                dataset_path, output_path))
            filter_extended(dataset_path, output_path)
        (sentences_a, sentences_b), (
            language_a,
            language_b) = get_filtered_extended_sentences(output_path)
    else:
        raise NotImplementedError

    # Take n sentences from data for training and validation
    n = samples
    sentences_a = sentences_a[:n]
    sentences_b = sentences_b[:n]

    # Do not split grain size classes
    split_b = '_' if target == 'grain size' else ' '

    print_words_info(sentences_a, name=language_a)
    print_words_info(sentences_b, name=language_b, split=split_b)

    preproc_sentences_a, preproc_sentences_b, tokenizer_a, tokenizer_b =\
        preprocess(sentences_a, sentences_b, split_x = ' ', split_y=split_b)

    max_sequence_length_a = preproc_sentences_a.shape[1]
    max_sequence_length_b = preproc_sentences_b.shape[1]
    vocabulary_size_a = len(tokenizer_a.word_index)
    vocabulary_size_b = len(tokenizer_b.word_index)

    print('Data Preprocessed')
    print("Max {} sentence length: {}".format(language_a,
                                              max_sequence_length_a))
    print("Max {} sentence length: {}".format(language_b,
                                              max_sequence_length_b))
    print("{} vocabulary size: {}".format(language_a, vocabulary_size_a))
    print("{} vocabulary size: {}".format(language_b, vocabulary_size_b))

    tmp_x = pad(preproc_sentences_a)

    model = get_model(tmp_x.shape, max_sequence_length_b, vocabulary_size_a,
                      vocabulary_size_b)

    pretrained_model_name = 'pretrained_model_{}.h5'.format(target)

    # Uncomment below to pick up pretrained model for predictions
    # if os.path.isfile(pretrained_model_name):
    # print('Loading pretrained')
    # model.load_weights(pretrained_model_name)
    # else:
    print('Training')
    model.fit(tmp_x,
              preproc_sentences_b,
              batch_size=1024,
              epochs=epochs,
              validation_split=0.2)
    print('Saving model weights')
    model.save_weights(pretrained_model_name)

    # Output some predictions
    for i in range(5):
        print('{:>5}: {:25}: {}'.format(i, language_a, sentences_a[i]))
        print('{:>5}: {:25}: {}'.format(i, 'Original ' + language_b,
                                        sentences_b[i]))
        print('{:>5}: {:25}: {}'.format(
            i, 'Predicted ' + language_b,
            logits_to_text(model.predict(tmp_x[i:(i + 1)])[0], tokenizer_b)))


if '__main__' == __name__:
    dataset_path = r"data\\RealPore Por Perm Lithology data 1240 Wells Norway public.xlsx"

    parser = argparse.ArgumentParser()
    parser.add_argument('TARGET',
                        help='Valid options: transcription or "grain size"')
    parser.add_argument(
        '--samples',
        help='Number of training and validation samples. Validation split 0.2',
        type=int,
        default=20000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dataset', default=dataset_path)
    args = parser.parse_args()

    if args.TARGET == 'transcription' or args.TARGET == 'grain size':
        main(args.dataset, args.TARGET, args.samples, args.epochs)
    else:
        raise "Unknown TARGET"
