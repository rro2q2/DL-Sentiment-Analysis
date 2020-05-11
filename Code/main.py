import char_CNN.model as char_model
import fastText.fasttext as ft_model
import Text_CNN.Text_CNN as text_cnn_model
import BERT.BERT as bert_model

def main():
    ## Running the fastTest model ##
    # NOTE: it reports the training time in the fasttext.py file
    print("**** RUNNING fastText MODEL ****")
    ft_model.run_model()

    ## Running the char CNN model ##
    # NOTE: it reports the training time in the model.py file
    print("**** RUNNING char-CNN MODEL ****")
    char_model.run_model()

    print("**** RUNNING Text-CNN MODEL ****")
    text_cnn_hyperparams = {
        "embedding_dimension": 100,
        "batch_size": 64,
        "epochs": 10,
        "dropout": 0.5,
        "learning_rate": .001
    }

    text_cnn_model.run_model(text_cnn_hyperparams,
                             "../data/HW5_training.csv",
                             "../data/HW5_test.csv",
                             "small")

    text_cnn_model.run_model(text_cnn_hyperparams,
                             "../data/Proble4Dataset_large_train.csv",
                             "../data/Proble4Dataset_large_test.csv",
                             "large")

    print("**** RUNNING BERT MODEL ****")
    bert_hyperparams = {
        "embedding_dimension": 256,
        "batch_size": 10,
        "epochs": 20,
        "dropout": 0.25,
        "learning_rate": .001
    }

    bert_model.run_model(bert_hyperparams,
                             "../data/HW5_training.csv",
                             "../data/HW5_test.csv",
                             "small")

    bert_model.run_model(bert_hyperparams,
                             "../data/Proble4Dataset_large_train.csv",
                             "../data/Proble4Dataset_large_test.csv",
                             "large")


if __name__ == '__main__':
    main()
