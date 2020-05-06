import torch
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F
import spacy
import src.Util.Common_Util as util
import time
import torch.optim as optim
import en_core_web_sm
import pandas as pd

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[0], embedding_dim))

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[1], embedding_dim))

        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[2], embedding_dim))

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)

        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        return self.fc(cat)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = util.binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = util.binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def predict_sentiment(model, sentence, text_field, min_len=5):
    nlp = en_core_web_sm.load()
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [text_field.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(0)
    prediction = torch.round(torch.sigmoid(model(tensor)).squeeze())
    return prediction.item()

def train_model(model, epochs, train_iterator, valid_iterator, optimizer, criterion, identifier):
    best_valid_loss = float('inf')

    train_losses_lst = []
    valid_losses_lst = []

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = util.get_time_diff(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "cnn-model-" + identifier + ".pt")

        train_losses_lst.append(train_loss)
        valid_losses_lst.append(valid_loss)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    return train_losses_lst, valid_losses_lst


def run_model(hyperparameters, train_path, test_path, identifier):
    spacy.load('en_core_web_sm')

    total_filters = 100
    filter_sizes = [3, 4, 5]
    output_dim = 1

    text_field = data.Field(tokenize='spacy', batch_first=True)
    label_field = data.LabelField(dtype=torch.float)

    train_dataset = data.TabularDataset(
        path=train_path, format='csv', skip_header=True,
        fields=[('text', text_field),
                ('label', label_field)])

    test_data = data.TabularDataset(
        path=test_path, format='csv', skip_header=True,
        fields=[('text', text_field),
                ('label', label_field)])

    train_data, valid_data = train_dataset.split(split_ratio=0.8)

    text_field.build_vocab(train_data,
                           max_size=25_000,
                           vectors="glove.6B.100d",
                           unk_init=torch.Tensor.normal_)
    label_field.build_vocab(train_data)

    train_iterator = data.BucketIterator(train_data, batch_size=hyperparameters["batch_size"])
    valid_iterator = data.BucketIterator(valid_data, batch_size=hyperparameters["batch_size"])
    test_iterator = data.BucketIterator(test_data, batch_size=hyperparameters["batch_size"])

    model = CNN(len(text_field.vocab), hyperparameters["embedding_dimension"], total_filters, filter_sizes, output_dim, hyperparameters["dropout"], text_field.vocab.stoi[text_field.pad_token])
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    start_time_total = time.time()
    train_losses, valid_losses = train_model(model, hyperparameters["epochs"], train_iterator, valid_iterator, optimizer, criterion, identifier)
    end_time_total = time.time()

    total_mins, total_secs = util.get_time_diff(start_time_total, end_time_total)
    print(f'Total Training Time: {total_mins}m {total_secs}s')

    util.plot_losses(train_losses, valid_losses, "Text-CNN-"+identifier+".png")

    train_data_input = pd.read_csv(train_path)
    train_prediction_lst = []
    for text, label in zip(train_data_input["Text"], train_data_input["Label"]):
        train_prediction_lst.append(predict_sentiment(model, text, text_field))

    true_positives, false_positives, true_negatives, false_negatives = util.confusion(
        torch.Tensor(train_prediction_lst), torch.tensor(train_data_input["Label"].to_numpy()))

    accuracy = (true_positives + true_negatives) / len(train_prediction_lst)
    print("Training Accuracy :: ", accuracy)
    print("Training Confusion Matrix :: \n \t Predicted:0 \t Predicted: 1 \n Actual 0: \t", true_positives, "\t", false_positives, "\n Actual 1: \t", false_negatives, "\t", true_negatives)

    test_data_input = pd.read_csv(test_path)
    test_prediction_lst = []
    for text, label in zip(test_data_input["Text"], test_data_input["Label"]):
        test_prediction_lst.append(predict_sentiment(model, text, text_field))

    true_positives, false_positives, true_negatives, false_negatives = util.confusion(
        torch.Tensor(test_prediction_lst), torch.Tensor(test_data_input["Label"].to_numpy()))
    accuracy = (true_positives + true_negatives) / len(test_prediction_lst)

    print("Testing Accuracy :: ", accuracy)
    print("Testing Confusion Matrix :: \n \t Predicted:0 \t Predicted: 1 \n Actual 0: \t", true_positives, "\t", false_positives, "\n Actual 1: \t", false_negatives, "\t", true_negatives)

