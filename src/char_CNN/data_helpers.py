from constants import *


def load_ag_data():
    train = pd.read_csv(CHAR_TRAIN_PATH, header=None)
    train = train.dropna()
    x_train = train[1]
    x_train = np.array(x_train)

    y_train = train[0]
    y_train = to_categorical(y_train)

    test = pd.read_csv(CHAR_TEST_PATH, header=None)
    x_test = test[1]
    x_test = np.array(x_test)

    y_test = test[0]
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)


def encode_data(x, maxlen, vocab):
    # Iterate over the loaded data and create a matrix of size (len(x), maxlen)
    # Each character is encoded into a one-hot array later at the lambda layer.
    # Chars not in the vocab are encoded as -1, into an all zero vector.

    input_data = np.zeros((len(x), maxlen), dtype=np.int)
    for dix, sent in enumerate(x):
        counter = 0
        for c in sent:
            if counter >= maxlen:
                pass
            else:
                ix = vocab.get(c, -1)  # get index from vocab dictionary, if not in vocab, return -1
                input_data[dix, counter] = ix
                counter += 1
    return input_data


def create_vocab_set():
    # This alphabet is 69 chars vs. 70 reported in the paper since they include two
    # '-' characters. See https://github.com/zhangxiangxiao/Crepe#issues.

    alphabet = set(list(string.ascii_lowercase) + list(string.digits) +
                   list(string.punctuation) + ['\n'])
    vocab_size = len(alphabet)
    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, alphabet
