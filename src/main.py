import src.char_CNN.model as char_model
import src.fastText.fasttext as ft_model


def main():
    ## Running the fastTest model ##
    # NOTE: it reports the training time in the fasttext.py file
    print("**** RUNNING fastText MODEL ****")
    ft_model.run_model()

    ## Running the char CNN model ##
    # NOTE: it reports the training time in the model.py file
    print("**** RUNNING char-CNN MODEL ****")
    char_model.run_model()


if __name__ == '__main__':
    main()
