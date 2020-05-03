from constants import *

import src.char_CNN.py_crepe as crepe
import src.char_CNN.data_helpers as helpers


def plot_confusion_matrix(df_confusion, cmap=plt.cm.Blues):
    plt.matshow(df_confusion, cmap=cmap)  # imshow
    plt.title("char-CNN Confusion Matrix")
    plt.colorbar()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.savefig(CHARCNN_CONF_MATRIX_PATH)


def run_model():
    np.random.seed(123)  # for reproducibility
    # set parameters:
    # Maximum length. Longer gets chopped. Shorter gets padded.
    maxlen = 1014

    # Model params
    # Filters for conv layers
    nb_filter = 256
    # Number of units in the dense layer
    dense_outputs = 1024
    # Conv layer kernel size
    filter_kernels = [7, 7, 3, 3, 3, 3]
    # Number of units in the final output layer. Number of classes.
    cat_output = 2 # should be two: 1s and 0s

    # Compile/fit params
    batch_size = 80
    nb_epoch = 20

    # Expect x to be a list of sentences. Y to be index of the categories.
    (xt, yt), (x_test, y_test) = helpers.load_ag_data()

    vocab, reverse_vocab, vocab_size, alphabet = helpers.create_vocab_set()

    model = crepe.create_model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter, cat_output)
    # Encode data
    xt = helpers.encode_data(xt, maxlen, vocab)
    x_test = helpers.encode_data(x_test, maxlen, vocab)

    #model.summary()

    print('Fit model...')
    start = timeit.default_timer()
    H = model.fit(xt, yt, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, shuffle=True)
    stop = timeit.default_timer()
    print("[TRAINING TIME]:", stop - start)

    print("[INFO] evaluating network...")
    predIdxs = model.predict(x_test, batch_size=batch_size)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)

    # show a nicely formatted classification report
    print(classification_report(y_test.argmax(axis=1), predIdxs,
                                target_names=['negative', 'positive']))

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(y_test.argmax(axis=1), predIdxs)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    # show the confusion matrix, accuracy, sensitivity, and specificity
    y_true = pd.Series(y_test.argmax(axis=1), name='Actual')
    y_pred = pd.Series(predIdxs, name='Predicted')
    conf_matrix = pd.crosstab(y_true, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

    plot_confusion_matrix(conf_matrix)
