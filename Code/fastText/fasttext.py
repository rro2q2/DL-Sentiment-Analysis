from constants import *


def process_sentence(text):
    text = text.strip('1')
    text = text.strip('0')
    text = text.lower()
    text = ''.join([c for c in text if c not in punctuation])

    return text


def transform_instance(row):
    cur_row = []
    # Prefix the index-ed label with __label__
    label = "__label__" + row[1]
    cur_row.append(label)
    cur_row.extend(nltk.word_tokenize(process_sentence(row[0])))
    return cur_row


def preprocess(input_file, output_file):
    i = 0
    with open(output_file, 'w') as csvoutfile:
        csv_writer = csv.writer(csvoutfile, delimiter=' ', lineterminator='\n')
        with open(input_file, 'r', newline='', encoding='latin1') as csvinfile:  # ,encoding='latin1'
            csv_reader = csv.reader(csvinfile, delimiter=',', quotechar='"')
            for row in csv_reader:
                if row[1] in ['0', '1'] and row[0] != '':
                    row_output = transform_instance(row)

                    csv_writer.writerow(row_output)
                    # print(row_output)
                i = i + 1


def run_model():
    preprocess(FASTTEXT_TRAIN_PATH, FASTTEXT_SENT_TRAIN_PATH)
    preprocess(FASTTEXT_TEST_PATH, FASTTEXT_SENT_TEST_PATH)

    hyper_params = {"lr": 0.01,
                    "epoch": 20,
                    "wordNgrams": 2,
                    "dim": 20}

    # Train the model.
    start = timeit.default_timer()
    model = fasttext.train_supervised(input=FASTTEXT_SENT_TRAIN_PATH, **hyper_params)

    stop = timeit.default_timer()
    print("[TRAINING TIME]:", stop - start)
    # CHECK PERFORMANCE
    model_acc_training_set = model.test(FASTTEXT_SENT_TRAIN_PATH)
    model_acc_validation_set = model.test(FASTTEXT_SENT_TEST_PATH)

    # DISPLAY ACCURACY OF TRAINED MODEL
    print("Training Accuracy:", model_acc_training_set[1])
    print("Validation Accuracy:", model_acc_validation_set[1])
