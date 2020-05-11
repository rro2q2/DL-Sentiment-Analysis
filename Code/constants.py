from __future__ import print_function
from __future__ import division
import os
import fasttext
import numpy as np
import pandas as pd
import string
from string import punctuation
import nltk
import csv
import timeit
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D


## CONSTANT PATHS ##
# char-CNN output path
CHARCNN_CONF_MATRIX_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        'output/charCNN_conf_matrix.png')

# char-CNN train file path
CHAR_TRAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/char_train.csv')

# char-CNN train file path
CHAR_TEST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/char_test.csv')

# fastText train file path
FASTTEXT_TRAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/fastText_train.csv')

# fastText train file path
FASTTEXT_TEST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/fastText_test.csv')

# fastText sentences train file path
FASTTEXT_SENT_TRAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fastText/sentences.train')

# fastText sentences train file path
FASTTEXT_SENT_TEST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fastText/sentences.test')
