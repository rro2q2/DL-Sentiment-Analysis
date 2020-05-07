import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel
from torchtext import data
import src.Util.Common_Util as util
import time
import torch.optim as optim
import pandas as pd

tokenizer = None

class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 n_layers,
                 dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=True,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        with torch.no_grad():
            embedded = self.bert(text)[0]

        _, hidden = self.rnn(embedded)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        output = self.out(hidden)

        return output

def tokenize_and_cut(sentence):
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens

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
            torch.save(model.state_dict(), "bert-model-" + identifier + ".pt")

        train_losses_lst.append(train_loss)
        valid_losses_lst.append(valid_loss)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    return train_losses_lst, valid_losses_lst

def predict_sentiment(model, sentence):
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    indexed = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(0)
    prediction = torch.round(torch.sigmoid(model(tensor)).squeeze())
    return prediction.item()

def run_model(hyperparameters, train_path, test_path, identifier):
    layers = 2

    global tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    text_field = data.Field(batch_first = True,
                      use_vocab = False,
                      tokenize = tokenize_and_cut,
                      preprocessing = tokenizer.convert_tokens_to_ids,
                      init_token = tokenizer.cls_token_id,
                      eos_token = tokenizer.sep_token_id,
                      pad_token = tokenizer.pad_token_id,
                      unk_token = tokenizer.unk_token_id)

    label_field = data.LabelField(dtype = torch.float)

    train_dataset = data.TabularDataset(
        path=train_path, format='csv', skip_header=True,
        fields=[('text', text_field),
                ('label', label_field)])

    test_data = data.TabularDataset(
        path=test_path, format='csv', skip_header=True,
        fields=[('text', text_field),
                ('label', label_field)])

    train_data, valid_data = train_dataset.split(split_ratio=0.8)

    label_field.build_vocab(train_data)

    train_iterator = data.BucketIterator(train_data, batch_size=hyperparameters["batch_size"])
    valid_iterator = data.BucketIterator(valid_data, batch_size=hyperparameters["batch_size"])
    test_iterator = data.BucketIterator(test_data, batch_size=hyperparameters["batch_size"])

    bert = BertModel.from_pretrained('bert-base-uncased')
    model = BERTGRUSentiment(bert, hyperparameters["embedding_dimension"], layers, hyperparameters["dropout"])

    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    start_time_total = time.time()
    train_losses, valid_losses = train_model(model, hyperparameters["epochs"], train_iterator, valid_iterator,
                                             optimizer, criterion, identifier)
    end_time_total = time.time()

    total_mins, total_secs = util.get_time_diff(start_time_total, end_time_total)
    print(f'Total Training Time: {total_mins}m {total_secs}s')

    util.plot_losses(train_losses, valid_losses, "Text-CNN-" + identifier + ".png")

    train_data_input = pd.read_csv(train_path)
    train_prediction_lst = []
    for text, label in zip(train_data_input["Text"], train_data_input["Label"]):
        train_prediction_lst.append(predict_sentiment(model, text))

    true_positives, false_positives, true_negatives, false_negatives = util.confusion(
        torch.Tensor(train_prediction_lst), torch.tensor(train_data_input["Label"].to_numpy()))

    accuracy = (true_positives + true_negatives) / len(train_prediction_lst)
    print("Training Accuracy :: ", accuracy)
    print("Training Confusion Matrix :: \n \t Predicted:0 \t Predicted: 1 \n Actual 0: \t", true_positives, "\t",
          false_positives, "\n Actual 1: \t", false_negatives, "\t", true_negatives)

    test_data_input = pd.read_csv(test_path)
    test_prediction_lst = []
    for text, label in zip(test_data_input["Text"], test_data_input["Label"]):
        test_prediction_lst.append(predict_sentiment(model, text))

    true_positives, false_positives, true_negatives, false_negatives = util.confusion(
        torch.Tensor(test_prediction_lst), torch.Tensor(test_data_input["Label"].to_numpy()))
    accuracy = (true_positives + true_negatives) / len(test_prediction_lst)

    print("Testing Accuracy :: ", accuracy)
    print("Testing Confusion Matrix :: \n \t Predicted:0 \t Predicted: 1 \n Actual 0: \t", true_positives, "\t",
          false_positives, "\n Actual 1: \t", false_negatives, "\t", true_negatives)

